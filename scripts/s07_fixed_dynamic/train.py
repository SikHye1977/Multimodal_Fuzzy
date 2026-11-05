#!/usr/bin/env python3
"""Compare fixed and input-dependent 512-D tau with the unchanged C loss.

Success means pre-decoder Hamming distance <= t (default 32), not zero errors.
Uses frozen embeddings, matched batches and the original participant splits.
No BCH compilation/decoding, reliability fitting, feature extraction or Fusion.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import common
from scripts.common import data as protocol, losses as loss_ops, metrics as metric_ops, training as train_ops, provenance
from scripts.common.bch_evaluator import BCHEvaluator
from Thresholding_Model_512 import ThresholdingModel512
MODALITIES = common.MODALITIES


MODES = ("fixed", "dynamic")
PRIMARY = "mean_unique_within_radius_fraction"
DESCRIPTIONS = {
    "fixed": "Per modality: one global trainable 512-vector, tau=(base_bound+delta_bound)*tanh(theta)",
    "dynamic": "Per modality: global base + current-input residual, tau=base_bound*tanh(theta)+delta_bound*tanh(MLP(x))",
}


@torch.no_grad()
def initial_output_digest(model, cache, rows):
    """Different architectures must start with identical zero tau and hard bits."""
    outputs = train_ops.quantize_pair(model, cache, rows)
    inputs = common.model_inputs(cache, rows)
    digest = hashlib.sha256()
    for out, pair in zip(outputs, ((inputs[0], inputs[1]), (inputs[2], inputs[3]))):
        for m, x in zip(MODALITIES, pair):
            tau, bits = out[f"{m}_tau"], out[f"{m}_bits"]
            if bool((tau != 0).any()) or not torch.equal(bits, (x >= 0).to(torch.uint8)):
                raise RuntimeError("Both architectures must initialize at tau=0")
            digest.update(tau.cpu().numpy().tobytes())
            digest.update(bits.cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, cache, rows, users, plan, args, on_batch=None):
    e, q = train_ops.collect(model, cache, rows, args.batch_size, on_batch=on_batch)
    result = metric_ops.coordinate_bit_metrics(e, q, plan, args)
    result["unique_pairs"] = {}
    for m, cols in (("face", [common.EF, common.QF]), ("fingerprint", [common.EP, common.QP])):
        idx = metric_ops.unique_indices(rows[:, cols])
        d = (e[f"{m}_bits"][idx] != q[f"{m}_bits"][idx]).sum(1).numpy()
        et, qt = e[f"{m}_tau"][idx], q[f"{m}_tau"][idx]
        # Tau variability uses each observed image once, avoiding repeated pairs.
        sample_ids = np.r_[rows[:, cols[0]], rows[:, cols[1]]]
        observations = metric_ops.unique_indices(sample_ids)
        all_tau = torch.cat((e[f"{m}_tau"], q[f"{m}_tau"]))[observations]
        result["unique_pairs"][m] = {
            "genuine_pair_count": len(idx), "genuine_hamming_bits": float(d.mean()),
            "genuine_hamming_p95_bits": float(np.quantile(d, .95)),
            "within_radius_count": int((d <= args.bch_t).sum()),
            "within_radius_fraction": float((d <= args.bch_t).mean()),
            "within_target_count": int((d <= args.bch_t - args.safety_margin).sum()),
            "within_target_fraction": float((d <= args.bch_t - args.safety_margin).mean()),
            "zero_error_fraction": float((d == 0).mean()),
            "tau_mean_absolute": float(all_tau.abs().mean()),
            "tau_across_unique_images_std_mean": float(all_tau.std(dim=0, unbiased=False).mean()),
            "tau_genuine_mean_absolute_difference": float((et - qt).abs().mean()),
            "tau_bound_saturation_fraction": float((all_tau.abs() >= .98*model.config.tau_bound).float().mean()),
            "negative_scopes": metric_ops.negative_diagnostics(e, q, rows, users, m, args.bch_t)}
    for key in ("within_radius_fraction", "genuine_hamming_bits"):
        result["mean_unique_" + key] = sum(result["unique_pairs"][m][key] for m in MODALITIES) / 2
    return result, e, q


def better(candidate, best, args):
    """Inclusive <=t rate first; same fixed/dynamic rule, no decoder criterion."""
    if abs(candidate[PRIMARY] - best[PRIMARY]) > 1e-12:
        return candidate[PRIMARY] > best[PRIMARY]
    return candidate["mean_unique_genuine_hamming_bits"] < best["mean_unique_genuine_hamming_bits"] - args.min_bit_gain


def train_trial(model, seed, data, users, cache, rows, plan, baseline, args, directory, schedules, future):
    model.set_stage("threshold")
    log = common.Log(directory)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.threshold_lr)
    best, best_epoch, stale = baseline, 0, 0
    metadata = {"arm": "C", "mode": model.config.mode, "seed": seed, "stage": "threshold_only",
                "selection_metric": PRIMARY, "success_hamming_max_bits": args.bch_t,
                "requires_new_enrollment": True, "reliability_fitted": False}
    model.save_checkpoint(directory / "best_threshold.pt", {**metadata, "epoch": 0, "origin": "fixed_zero_fallback"})
    log.record({"epoch": 0, "validation": baseline, "selected": True})
    progress = common.Progress(f"C/{model.config.mode} seed={seed}", args.epochs, future, args.log_interval)
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches, padding = train_ops.make_schedule(data, users, args, seed, epoch, "threshold", schedules)
        steps = len(batches) + math.ceil(len(rows) / args.batch_size)
        sums, count = defaultdict(float), 0
        for step, (batch, _) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            e, q = train_ops.quantize_pair(model, cache, batch)
            losses = loss_ops.bch_threshold_loss(model, e, q, batch, users, args, "C")
            if not bool(torch.isfinite(losses["total"])):
                raise RuntimeError("Nonfinite C loss")
            losses["total"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            for key, value in losses.items():
                sums[key] += float(value.detach()) * len(batch)
            sums["gradient_norm"] += float(gradient) * len(batch)
            count += len(batch)
            progress.update(epoch, step, steps, "train", sums["total"] / count)
        metrics, _, _ = evaluate(model, cache, rows, users, plan, args,
            on_batch=lambda k: progress.update(epoch, len(batches)+k, steps, "validation", sums["total"]/count))
        eligible, reasons = metric_ops.bch_selection_guard(metrics, baseline, args)
        improved = eligible and better(metrics, best, args)
        if improved:
            best, best_epoch, stale = metrics, epoch, 0
            model.save_checkpoint(directory / "best_threshold.pt", {**metadata, "epoch": epoch, "origin": "trained_network"})
        else:
            stale += 1
        model.save_checkpoint(directory / "last_threshold.pt", {**metadata, "epoch": epoch, "origin": "last_epoch"})
        progress.finish_line()
        log.record({"epoch": epoch, "train_loss": {k: v/count for k, v in sums.items()}, "padding": padding,
                    "validation": metrics, "eligible": eligible, "reasons": reasons, "selected": improved})
        values = metrics["unique_pairs"]
        log.message(f"C/{model.config.mode} seed={seed} epoch {epoch} | "
            f"얼굴 {values['face']['genuine_hamming_bits']:.2f} bit / {args.bch_t}비트 이내 {100*values['face']['within_radius_fraction']:.2f}% | "
            f"지문 {values['fingerprint']['genuine_hamming_bits']:.2f} bit / {args.bch_t}비트 이내 {100*values['fingerprint']['within_radius_fraction']:.2f}% | "
            f"선택 epoch={best_epoch}, guard={'통과' if eligible else '탈락'}")
        if stale >= args.patience:
            break
    return {"best_epoch": best_epoch, "epochs_run": epoch, "validation": best,
            "origin": "trained_network" if best_epoch else "fixed_zero_fallback"}


def export_details(directory, split, rows, data, e, q, args):
    arrays = {f"{role}_{k}": v.numpy() for role, out in (("enrollment", e), ("query", q)) for k, v in out.items()}
    arrays["pair_rows"] = rows
    records = []
    for m, cols in (("face", [common.EF, common.QF]), ("fingerprint", [common.EP, common.QP])):
        distances = (e[f"{m}_bits"] != q[f"{m}_bits"]).sum(1).numpy()
        arrays[f"{m}_hamming_bits"] = distances
        arrays[f"{m}_within_radius"] = distances <= args.bch_t
        for i in metric_ops.unique_indices(rows[:, cols]):
            records.append({"modality": m, "enrollment": data.files[m][int(rows[i, cols[0]])],
                "query": data.files[m][int(rows[i, cols[1]])], "hamming_bits": int(distances[i]),
                "within_radius": int(distances[i] <= args.bch_t),
                "within_target": int(distances[i] <= args.bch_t - args.safety_margin),
                "tau_mean_absolute_difference": float((e[f"{m}_tau"][i] - q[f"{m}_tau"][i]).abs().mean())})
    np.savez_compressed(directory / f"{split}_bits_tau.npz", **arrays)
    common.write_csv(directory / f"{split}_unique_pairs.csv", list(records[0]), records)


def write_tables(directory, trials, baselines):
    rows = []
    for trial in trials:
        for split in ("validation", "test"):
            for m in MODALITIES:
                metric = trial[split]["unique_pairs"][m]
                reference = baselines[split]["unique_pairs"][m]
                row = {"mode": trial["mode"], "seed": trial["seed"], "split": split, "modality": m,
                       "best_epoch": trial["threshold"]["best_epoch"], "origin": trial["threshold"]["origin"]}
                row.update({k: v for k, v in metric.items() if k != "negative_scopes"})
                for scope in ("all_different_identities", "different_people", "same_person_different_fingers"):
                    values = metric["negative_scopes"].get(scope, {})
                    for key in ("pair_count", "mean_hamming_bits", "p05_hamming_bits", "FAR_h"):
                        row[f"{scope}_{key}"] = values.get(key)
                row["within_radius_percent"] = 100*metric["within_radius_fraction"]
                row["within_radius_delta_vs_zero_pp"] = 100*(metric["within_radius_fraction"]-reference["within_radius_fraction"])
                row["genuine_hamming_delta_vs_zero_bits"] = metric["genuine_hamming_bits"]-reference["genuine_hamming_bits"]
                rows.append(row)
    common.write_csv(directory / "comparison.csv", list(rows[0]), rows)
    keys = [k for k in rows[0] if k not in ("mode", "seed", "split", "modality", "origin")]
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ("mode", "split", "modality"))].append(row)
    aggregate = []
    for group, values in grouped.items():
        item = dict(zip(("mode", "split", "modality"), group))
        item["seed_count"] = len(values)
        for key in keys:
            numbers = [v[key] for v in values if v[key] is not None]
            item[key+"_mean"] = float(np.mean(numbers)) if numbers else None
            item[key+"_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        aggregate.append(item)
    common.write_csv(directory / "aggregate.csv", list(aggregate[0]), aggregate)
    lookup = {(r["mode"], r["seed"], r["split"], r["modality"]): r for r in rows}
    deltas = []
    for row in rows:
        if row["mode"] != "dynamic":
            continue
        fixed = lookup[("fixed", row["seed"], row["split"], row["modality"])]
        item = {"comparison": "dynamic-fixed", **{k: row[k] for k in ("seed", "split", "modality")}}
        for key in keys:
            item[key+"_delta"] = row[key]-fixed[key] if row[key] is not None and fixed[key] is not None else None
        deltas.append(item)
    common.write_csv(directory / "dynamic_minus_fixed.csv", list(deltas[0]), deltas)
    return aggregate, deltas


def run(args, data, users, pairs, source, source_hashes, *, synthetic=False):
    started = time.perf_counter()
    args.temperature, args.data_seed = float(source["temperature"]), int(source["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log, trials, schedules, initial_hashes, current = common.Log(args.output_dir), [], {}, {}, None
    try:
        common.write_json(args.output_dir / "status.json", {"state": "running", "synthetic_test_only": synthetic})
        if not synthetic:
            for name in protocol.MANIFESTS:
                shutil.copy2(args.source_run / name, args.output_dir / name)
        code_hashes = provenance.code_hashes(__file__)
        config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "loss_arm": "C", "modes": list(MODES), "mode_descriptions": DESCRIPTIONS,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "torch": str(torch.__version__),
            "synthetic_test_only": synthetic, "input_sha256": data.hashes, "source_sha256": source_hashes, "code_sha256": code_hashes,
            "primary_success_definition": "pre-decoder Hamming(enrollment_bits, query_bits) <= bch_t",
            "selection": "Original zero-baseline guards; equal-modality mean unique within-radius rate, then lower mean Hamming",
            "test_used_for_selection": False, "BCH_decoding": False, "FE_key_evaluation": False, "reliability_trained": False,
            "requires_new_enrollment": True, "checkpoint_resuming_supported": False,
            "initialization": "Both modes start at tau=0. Architecture weights differ; initial outputs and batch schedules are matched.",
            "C_loss": "Unmodified train_thresholding_bch_losses.bch_threshold_loss(..., arm='C'), including all regularizers"}
        common.write_json(args.output_dir / "config.json", config)
        log.message(f"결과: {args.output_dir}\nC 손실 고정 | fixed / dynamic | 성공 기준: {args.bch_t}비트 이내 | C 목표: {args.bch_t-args.safety_margin}비트")
        log.message("두 구조 모두 tau=0에서 시작합니다. 같은 시드는 같은 학습 쌍과 배치 순서를 사용합니다.")
        log.message("ETA는 남은 최대 epoch와 관측 속도를 사용합니다. 조기 종료와 최종 평가에 따라 달라집니다.")
        cache = {m: torch.tensor(x, device=args.device) for m, x in data.arrays.items()}
        plans = {s: metric_ops.evaluation_plan(pairs[s], users, "cpu") for s in ("validation", "test")}
        zero = train_ops.make_coordinate_model(args, "zero", args.seeds[0])
        baseline, _, _ = evaluate(zero, cache, pairs["validation"], users, plans["validation"], args)
        baselines = {"validation": baseline}
        common.write_json(args.output_dir / "zero_validation.json", baseline)
        jobs = [(seed, mode) for seed in args.seeds for mode in MODES]
        log.message(f"{len(jobs)}회 순차 실행 | train={len(pairs['train'])}, validation={len(pairs['validation'])}, test={len(pairs['test'])} 쌍")
        for j, (seed, mode) in enumerate(jobs):
            current = args.output_dir / f"{mode}_seed{seed}"
            current.mkdir()
            common.write_json(current / "status.json", {"state": "training"})
            model = train_ops.make_coordinate_model(args, mode, seed)
            digest = initial_output_digest(model, cache, pairs["train"][:args.batch_size])
            if str(seed) in initial_hashes and initial_hashes[str(seed)] != digest:
                raise RuntimeError("Initial fixed/dynamic outputs differ")
            initial_hashes[str(seed)] = digest
            training = train_trial(model, seed, data, users, cache, pairs["validation"], plans["validation"], baseline,
                args, current, schedules, (len(jobs)-j-1)*args.epochs)
            trial = {"mode": mode, "seed": seed, "loss_arm": "C", "threshold": training,
                "checkpoint": str(current / "best_threshold.pt"), "initial_outputs_sha256": digest,
                "threshold_parameter_count": sum(p.numel() for p in model.thresholds.parameters())}
            trials.append(trial)
            common.write_json(current / "training_summary.json", trial)
            common.write_json(current / "status.json", {"state": "trained_awaiting_test"})
            common.write_json(args.output_dir / "training_summary.json", trials)
            common.write_json(args.output_dir / "shared_schedule_sha256.json", schedules)
            log.message(f"학습 완료 {j+1}/{len(jobs)} | {mode} seed={seed} | 선택 epoch={training['best_epoch']}")
        log.message("모든 체크포인트 선택 완료. 선택된 모델의 비트·tau를 저장하고 시험 분할을 평가합니다.")
        baselines["test"], e, q = evaluate(zero, cache, pairs["test"], users, plans["test"], args)
        common.write_json(args.output_dir / "zero_test.json", baselines["test"])
        export_details(args.output_dir, "zero_test", pairs["test"], data, e, q, args)
        for j, trial in enumerate(trials):
            current = Path(trial["checkpoint"]).parent
            model, _ = ThresholdingModel512.load_checkpoint(trial["checkpoint"], args.device)
            progress = common.Progress(f"최종평가 {trial['mode']} seed={trial['seed']}", 2, 2*(len(trials)-j-1), args.log_interval)
            for epoch, split in enumerate(("validation", "test"), 1):
                steps = math.ceil(len(pairs[split])/args.batch_size)
                trial[split], e, q = evaluate(model, cache, pairs[split], users, plans[split], args,
                    on_batch=lambda k: progress.update(epoch, k, steps, split, None))
                export_details(current, split, pairs[split], data, e, q, args)
            progress.finish_line()
            if trial["validation"] != trial["threshold"]["validation"]:
                raise RuntimeError("Reloaded checkpoint does not reproduce selected validation metrics")
            common.write_json(current / "report.json", trial)
            common.write_json(current / "status.json", {"state": "completed"})
        aggregate, deltas = write_tables(args.output_dir, trials, baselines)
        data.assert_unchanged()
        for mapping in (source_hashes, code_hashes):
            if any(common.sha256(Path(path)) != digest for path, digest in mapping.items()):
                raise RuntimeError("Input source or code changed during the experiment")
        report = {"trials": trials, "baselines": baselines, "aggregate": aggregate, "paired_deltas": deltas,
            "seconds": time.perf_counter()-started, "synthetic_test_only": synthetic, "input_and_source_unchanged": True,
            "primary_success_definition": config["primary_success_definition"], "shared_schedule_sha256": schedules,
            "initial_outputs_sha256": initial_hashes,
            "limitations": ["Within-radius rate measures pre-decoder biometric mismatch, not exact biometric or key recovery.",
                "FAR_h is a distance acceptance proxy, not FE false acceptance. No different-person estimate for one-person splits.",
                "Original fingerprint split has 5/1/1 real participants. Seeds do not increase participant count.",
                "Face/fingerprint pairing is virtual, not genuine same-person multimodal data.",
                "Previously inspected test split gives exploratory evidence only.",
                "Threshold updates require new enrollment. Reliability heads remain uncalibrated.",
                "Fixed and dynamic differ in capacity and tau parameterization; same total bound does not imply identical optimization."]}
        common.write_json(args.output_dir / "report.json", report)
        common.write_json(args.output_dir / "status.json", {"state": "completed", "seconds": report["seconds"], "synthetic_test_only": synthetic})
        log.message(f"완료 | {common.duration(report['seconds'])} | {args.output_dir / 'comparison.csv'}")
        return report
    except BaseException as error:
        status = {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)}
        common.write_json(args.output_dir / "status.json", status)
        if current is not None:
            common.write_json(current / "status.json", status)
        (args.output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-run", type=Path, default=ROOT/"thresholding/results/train_20261029_162319_486882")
    p.add_argument("--embedding-dir", type=Path, help="Relocated identical embedding archives only")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    for name, value in {"epochs": 100, "patience": 15, "batch_size": 64, "hidden_dim": 32,
                        "threads": 2, "log_interval": 10, "bch_t": 32, "safety_margin": 4}.items():
        p.add_argument("--"+name.replace("_", "-"), type=int, default=value)
    for name, value in {"threshold_lr": 1e-4, "tau_base_bound": .01, "tau_delta_bound": .01, "grad_clip": 5.,
            "min_bit_gain": .1, "tau_stability_weight": .01, "tau_residual_weight": .001,
            "max_impostor_drop_bits": 5., "max_impostor_p05_drop_bits": 5., "max_auroc_drop": .001,
            "max_genuine_increase_bits": .5, "max_genuine_p95_increase_bits": 5., "max_row_outside_increase": .01,
            "max_constant_increase": .02, "max_coordinate_balance_increase": .005, "max_far_h_increase": 0.}.items():
        p.add_argument("--"+name.replace("_", "-"), type=float, default=value)
    for name, value in asdict(loss_ops.LossSettings()).items():
        p.add_argument("--"+name.replace("_", "-"), type=type(value), default=value)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    for key in ("epochs", "patience", "hidden_dim", "threads", "log_interval"):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.batch_size < 2 or not 1 <= args.bch_t <= 255 or not 0 <= args.safety_margin < args.bch_t:
        p.error("batch-size >= 2, 1 <= bch-t <= 255 and 0 <= safety-margin < bch-t required")
    if len(set(args.seeds)) != len(args.seeds) or not all(0 <= seed < 2**32 for seed in args.seeds):
        p.error("Unique seeds in [0, 2**32) required")
    for key, value in vars(args).items():
        if isinstance(value, float) and (not math.isfinite(value) or value < 0 or
                (key in {"threshold_lr", "tau_base_bound", "tau_delta_bound", "grad_clip"} and value == 0)):
            p.error(f"Invalid {key}")
    if args.max_far_h_increase > 1:
        p.error("max-far-h-increase must be <= 1")
    try:
        loss_ops.loss_settings(args)
    except ValueError as error:
        p.error(str(error))
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
        ROOT/"thresholding/results"/datetime.now().strftime("c_fixed_dynamic_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT/name for name in ("scripts", "muct-master", "UareU", "BCH", "feature_extractors")] + [args.source_run]
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    nested = any((parent/"config.json").is_file() and (parent/"status.json").is_file() for parent in args.output_dir.parents)
    if args.output_dir.exists() or nested or any(args.output_dir.is_relative_to(path.resolve()) for path in protected):
        p.error("Output must be a new directory outside code, input data and existing experiments")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print("기존 임베딩·사람 단위 분할·등록/질의 쌍을 확인합니다.", flush=True)
    data, users, pairs, source, hashes = provenance.prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output cannot be inside input embeddings")
    if args.check_only:
        data.assert_unchanged()
        print(json.dumps({"check_only": True, "training_started": False, "outputs_created": False,
            "loss_arm": "C", "modes": list(MODES), "seeds": args.seeds, "trial_count": 2*len(args.seeds),
            "success_hamming_max_bits": args.bch_t, "C_target_bits": args.bch_t-args.safety_margin,
            "temperature": source["temperature"], "BCH_decoding": False,
            "pairs": {s: len(rows) for s, rows in pairs.items()},
            "people": {s: {m: len({u.face_person if m == "face" else u.fingerprint_person for u in users if u.split == s})
                           for m in MODALITIES} for s in common.SPLITS},
            "cuda_available": torch.cuda.is_available()}, ensure_ascii=False, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Explicitly use --device cpu if desired.")
    run(args, data, users, pairs, source, hashes)


if __name__ == "__main__":
    main()
