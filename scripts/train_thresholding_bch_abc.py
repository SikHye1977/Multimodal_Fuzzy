#!/usr/bin/env python3
"""A: soft bits; B: sigmoid STE + mean Hamming; C: STE + BCH-radius loss.

Uses frozen cached embeddings and the original participant-disjoint split. The
existing ThresholdingModel512 is unchanged. No reliability or Fusion training.
Training is opt-in by invoking this script; --check-only never writes or trains.
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_thresholding_512 as base
from scripts.bch_evaluator import BCHEvaluator
from Thresholding_Model_512 import ThresholdingModel512

common, previous, contrastive = base.common, base.previous, base.contrastive
MODALITIES = common.MODALITIES
ARMS = ("A", "B", "C")
DESCRIPTIONS = {
    "A": "Original soft sigmoid bits, mean absolute positive distance",
    "B": "Hard forward / sigmoid backward STE, mean positive Hamming / 512",
    "C": "Same STE, mean squared relu(Hamming - (t - safety_margin)) / t**2",
}


def ste_bits(soft, hard):
    """Exactly 0/1 in forward; sigmoid derivative in backward, not identity."""
    return hard.to(soft.dtype).detach() + (soft - soft.detach())


def xor_distance(e, q, matrix=False):
    """Polynomial XOR equals Hamming for hard STE values, with gradients."""
    if matrix:
        return e.sum(1)[:, None] + q.sum(1)[None, :] - 2 * (e @ q.T)
    return (e + q - 2 * e * q).sum(1)


def bch_positive(distances, t, safety_margin):
    return (F.relu(distances - (t - safety_margin)) / t).square().mean()


def mined_negative(distances, labels, margin, hard_k, hard_mix):
    """Same identity-balanced mining as the original, accepting Hamming / 512."""
    identities, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    if len(identities) < 2:
        raise ValueError("At least two identities are needed in each training batch")
    penalty = (F.relu(margin - distances) / margin).square()
    membership = F.one_hot(inverse, num_classes=len(identities)).to(distances.dtype)
    group_mean = penalty @ (membership / counts.to(distances.dtype))
    valid = labels[:, None] != identities[None, :]
    ordinary_rows = (group_mean * valid).sum(1) / valid.sum(1)
    minimum = torch.full_like(group_mean, float("inf")).scatter_reduce(
        1, inverse[None, :].expand(len(labels), -1), distances, reduce="amin", include_self=True)
    nearest = minimum.masked_fill(~valid, float("inf")).topk(
        min(hard_k, len(identities) - 1), dim=1, largest=False).values
    hard_rows = (F.relu(margin - nearest) / margin).square().mean(1)
    weights = contrastive.balanced_weights(labels)
    ordinary, hard = (ordinary_rows * weights).sum(), (hard_rows * weights).sum()
    return (1 - hard_mix) * ordinary + hard_mix * hard, ordinary, hard


def threshold_loss(model, e, q, rows, users, args, arm):
    if arm == "A":
        # Exact original implementation, including regularizers and mining.
        return base.threshold_loss(model, e, q, rows, users, args)
    if arm not in ("B", "C"):
        raise ValueError(f"Unknown arm {arm}")
    settings = contrastive.loss_settings(args)
    labels = contrastive.identity_codes(rows, users, args.device)
    terms = defaultdict(list)
    for m in MODALITIES:
        eb = ste_bits(e[f"{m}_soft_bits"], e[f"{m}_bits"])
        qb = ste_bits(q[f"{m}_soft_bits"], q[f"{m}_bits"])
        d = xor_distance(eb, qb)
        positive = (d / 512).mean() if arm == "B" else bch_positive(d, args.bch_t, args.safety_margin)
        neg, ordinary, hard = mined_negative(xor_distance(eb, qb, matrix=True) / 512,
            labels[m], getattr(settings, m + "_margin"), settings.hard_k, settings.hard_mix)
        weights = contrastive.balanced_weights(labels[m])
        balance = sum(((weights[:, None] * b).sum(0) - .5).square().mean() for b in (eb, qb)) / 2
        row = sum(contrastive.row_penalty(b, weights, settings.row_lower) for b in (eb, qb)) / 2
        for key, value in {"positive": positive, "negative": neg, "negative_all": ordinary,
                           "negative_hard": hard, "balance": balance, "row": row}.items():
            terms[key].append(value)
        terms[f"{m}_hard_hamming_bits"].append(d.detach().mean())
        terms[f"{m}_above_target_fraction"].append((d.detach() > args.bch_t - args.safety_margin).float().mean())
    loss = {key: torch.stack(values).mean() for key, values in terms.items()}
    loss["tau_stability"] = torch.stack([
        ((e[f"{m}_tau"] - q[f"{m}_tau"]) / model.config.tau_bound).square().mean() for m in MODALITIES]).mean()
    loss["tau_residual"] = torch.stack([
        (out[f"{m}_delta_tau"] / model.config.tau_delta_bound).square().mean()
        for m in MODALITIES for out in (e, q)]).mean()
    loss["total"] = (loss["positive"] + settings.negative_weight * loss["negative"]
        + settings.balance_weight * loss["balance"] + settings.row_weight * loss["row"]
        + args.tau_stability_weight * loss["tau_stability"] + args.tau_residual_weight * loss["tau_residual"])
    return loss


def quantize_pair(model, cache, rows):
    ef, ep, qf, qp = common.model_inputs(cache, rows)
    return tuple({f"{m}_{key}": value for m, x in zip(MODALITIES, inputs)
                  for key, value in model.quantize(m, x).items()} for inputs in ((ef, ep), (qf, qp)))


def unique_indices(values):
    return np.sort(np.unique(values, axis=0, return_index=True)[1])


def negative_diagnostics(e, q, rows, users, modality, t):
    """Unique image pairs, excluding genuine identity; keep participant scopes."""
    ec, qc = (common.EF, common.QF) if modality == "face" else (common.EP, common.QP)
    ei, qi = unique_indices(rows[:, ec]), unique_indices(rows[:, qc])
    eb, qb = e[f"{modality}_bits"][ei].float(), q[f"{modality}_bits"][qi].float()
    distances = xor_distance(eb, qb, matrix=True).numpy()
    eu, qu = [users[int(rows[i, common.USER])] for i in ei], [users[int(rows[i, common.USER])] for i in qi]
    if modality == "face":
        same_person = np.array([[a.face_person == b.face_person for b in qu] for a in eu])
        different_identity = ~same_person
    else:
        same_person = np.array([[a.fingerprint_person == b.fingerprint_person for b in qu] for a in eu])
        different_identity = np.array([[a.finger != b.finger for b in qu] for a in eu])
    scopes = {"all_different_identities": different_identity, "different_people": ~same_person}
    if modality == "fingerprint":
        scopes["same_person_different_fingers"] = same_person & different_identity
    result = {}
    for name, mask in scopes.items():
        values = distances[mask]
        result[name] = {"pair_count": len(values), "mean_hamming_bits": float(values.mean()) if len(values) else None,
            "p05_hamming_bits": float(np.quantile(values, .05)) if len(values) else None,
            "within_t_count": int((values <= t).sum()),
            "FAR_h": float((values <= t).mean()) if len(values) else None}
    return result


def evaluation_steps(rows, batch_size, bch_batch_size):
    return math.ceil(len(rows) / batch_size) + sum(math.ceil(len(unique_indices(rows[:, cols])) / bch_batch_size)
        for cols in ([common.EF, common.QF], [common.EP, common.QP]))


@torch.no_grad()
def evaluate(model, cache, rows, users, plan, args, bch, on_step=None):
    e, q = previous.collect(model, cache, rows, args.batch_size, on_batch=on_step)
    metrics = base.bit_metrics(e, q, plan, args)
    metrics["unique_pairs"], outcomes = {}, {}
    step = math.ceil(len(rows) / args.batch_size)
    for m, cols in (("face", [common.EF, common.QF]), ("fingerprint", [common.EP, common.QP])):
        indices = unique_indices(rows[:, cols])
        eb, qb = e[f"{m}_bits"].numpy(), q[f"{m}_bits"].numpy()
        unique_outcomes = []
        for offset in range(0, len(indices), args.bch_batch_size):
            idx = indices[offset:offset + args.bch_batch_size]
            unique_outcomes.append(bch.recover(eb[idx], qb[idx]))
            step += 1
            if on_step:
                on_step(step)
        decoded = np.concatenate(unique_outcomes)
        lookup = {tuple(rows[i, cols]): int(code) for i, code in zip(indices, decoded)}
        outcomes[m] = np.array([lookup[tuple(row[cols])] for row in rows], dtype=np.uint8)
        d = np.count_nonzero(eb[indices] != qb[indices], axis=1)
        if np.any((d <= args.bch_t) & (decoded != 1)):
            raise RuntimeError(f"{m}: BCH failed inside its claimed correction radius")
        metrics["unique_pairs"][m] = {
            "genuine_pair_count": len(indices), "genuine_hamming_bits": float(d.mean()),
            "genuine_hamming_p95_bits": float(np.quantile(d, .95)),
            "within_radius_fraction": float((d <= args.bch_t).mean()),
            "within_target_fraction": float((d <= args.bch_t - args.safety_margin).mean()),
            "bch_exact_recovery_fraction": float((decoded == 1).mean()),
            "bch_no_candidate_fraction": float((decoded == 0).mean()),
            "bch_wrong_candidate_fraction": float((decoded == 2).mean()),
            "negative_scopes": negative_diagnostics(e, q, rows, users, m, args.bch_t)}
    for key in ("bch_exact_recovery_fraction", "within_radius_fraction", "genuine_hamming_bits"):
        metrics["mean_unique_" + key] = sum(metrics["unique_pairs"][m][key] for m in MODALITIES) / 2
    metrics["virtual_pair_both_bch_exact_fraction"] = float(((outcomes["face"] == 1) & (outcomes["fingerprint"] == 1)).mean())
    return metrics, e, q, outcomes


def selection_guard(metrics, baseline, args):
    _, reasons = base.selection_guard(metrics, baseline, args)
    for m in MODALITIES:
        for scope, values in metrics["unique_pairs"][m]["negative_scopes"].items():
            original = baseline["unique_pairs"][m]["negative_scopes"][scope]["FAR_h"]
            if values["FAR_h"] is not None and values["FAR_h"] > original + args.max_far_h_increase + 1e-12:
                reasons.append(f"{m}.{scope}.FAR_h increased by more than {args.max_far_h_increase}")
    return not reasons, reasons


def better(candidate, best, args):
    for key in ("mean_unique_bch_exact_recovery_fraction", "mean_unique_within_radius_fraction"):
        if abs(candidate[key] - best[key]) > 1e-12:
            return candidate[key] > best[key]
    return candidate["mean_unique_genuine_hamming_bits"] < best["mean_unique_genuine_hamming_bits"] - args.min_bit_gain


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_trial(model, arm, seed, data, users, cache, rows, plan, baseline, args, bch,
                directory, schedules, future):
    log = common.Log(directory)
    model.set_stage("threshold")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.threshold_lr)
    best, best_epoch, stale = baseline, 0, 0
    metadata = {"arm": arm, "mode": model.config.mode, "seed": seed, "stage": "threshold_only",
                "reliability_fitted": False, "requires_new_enrollment": True}
    model.save_checkpoint(directory / "best_threshold.pt", {**metadata, "epoch": 0, "origin": "fixed_zero_fallback"})
    log.record({"epoch": 0, "validation": baseline, "selected": True})
    progress = common.Progress(f"{arm}/{model.config.mode} seed={seed}", args.epochs, future, args.log_interval)
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches, padding = previous.make_schedule(data, users, args, seed, epoch, "threshold", schedules)
        steps = len(batches) + evaluation_steps(rows, args.batch_size, args.bch_batch_size)
        sums, count = defaultdict(float), 0
        for step, (batch, _) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            e, q = quantize_pair(model, cache, batch)
            losses = threshold_loss(model, e, q, batch, users, args, arm)
            if not bool(torch.isfinite(losses["total"])):
                raise RuntimeError("Nonfinite training loss")
            losses["total"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            for key, value in losses.items():
                sums[key] += float(value.detach()) * len(batch)
            sums["gradient_norm"] += float(gradient) * len(batch)
            count += len(batch)
            progress.update(epoch, step, steps, "train", sums["total"] / count)
        metrics, _, _, _ = evaluate(model, cache, rows, users, plan, args, bch,
            on_step=lambda k: progress.update(epoch, len(batches) + k, steps, "validation+BCH", sums["total"] / count))
        eligible, reasons = selection_guard(metrics, baseline, args)
        improved = eligible and better(metrics, best, args)
        if improved:
            best, best_epoch, stale = metrics, epoch, 0
            model.save_checkpoint(directory / "best_threshold.pt", {**metadata, "epoch": epoch, "origin": "trained_network"})
        else:
            stale += 1
        model.save_checkpoint(directory / "last_threshold.pt", {**metadata, "epoch": epoch, "origin": "last_epoch"})
        progress.finish_line()
        log.record({"epoch": epoch, "train_loss": {key: value / count for key, value in sums.items()},
                    "padding": padding, "validation": metrics, "eligible": eligible, "reasons": reasons, "selected": improved})
        bits = metrics["unique_pairs"]
        log.message(f"{arm}/{model.config.mode} seed={seed} epoch {epoch} | "
            f"얼굴 {bits['face']['genuine_hamming_bits']:.2f} bit / BCH {100*bits['face']['bch_exact_recovery_fraction']:.2f}% | "
            f"지문 {bits['fingerprint']['genuine_hamming_bits']:.2f} bit / BCH {100*bits['fingerprint']['bch_exact_recovery_fraction']:.2f}% | "
            f"선택 epoch={best_epoch}, guard={'통과' if eligible else '탈락'}")
        if stale >= args.patience:
            break
    return {"best_epoch": best_epoch, "epochs_run": epoch,
            "origin": "trained_network" if best_epoch else "fixed_zero_fallback", "validation": best}


def export_details(directory, split, rows, data, e, q, outcomes):
    arrays = {f"{role}_{key}": value.numpy() for role, out in (("enrollment", e), ("query", q)) for key, value in out.items()}
    arrays.update({"pair_rows": rows, **{f"{m}_bch_outcome": values for m, values in outcomes.items()}})
    np.savez_compressed(directory / f"{split}_bits_tau.npz", **arrays)
    records = []
    for m, cols in (("face", [common.EF, common.QF]), ("fingerprint", [common.EP, common.QP])):
        for i in unique_indices(rows[:, cols]):
            records.append({"modality": m, "enrollment": data.files[m][int(rows[i, cols[0]])],
                "query": data.files[m][int(rows[i, cols[1]])],
                "hamming_bits": int((e[f"{m}_bits"][i] != q[f"{m}_bits"][i]).sum()),
                "tau_mean_abs_difference": float((e[f"{m}_tau"][i] - q[f"{m}_tau"][i]).abs().mean()),
                "bch_outcome": int(outcomes[m][i])})
    common.write_csv(directory / f"{split}_unique_pairs.csv", list(records[0]), records)


def write_tables(directory, trials, baselines):
    rows = []
    for trial in trials:
        for split in ("validation", "test"):
            for m in MODALITIES:
                metric = trial[split]["unique_pairs"][m]
                reference = baselines[split]["unique_pairs"][m]
                row = {"arm": trial["arm"], "mode": trial["mode"], "seed": trial["seed"], "split": split,
                       "modality": m, "best_epoch": trial["threshold"]["best_epoch"], "origin": trial["threshold"]["origin"]}
                for key, value in metric.items():
                    if key != "negative_scopes":
                        row[key] = value
                for scope in ("all_different_identities", "different_people", "same_person_different_fingers"):
                    values = metric["negative_scopes"].get(scope, {})
                    for key in ("pair_count", "mean_hamming_bits", "p05_hamming_bits", "FAR_h"):
                        row[f"{scope}_{key}"] = values.get(key)
                for key in ("genuine_hamming_bits", "within_radius_fraction", "bch_exact_recovery_fraction"):
                    row[key + "_delta_vs_zero"] = metric[key] - reference[key]
                rows.append(row)
    common.write_csv(directory / "comparison.csv", list(rows[0]), rows)
    keys = [key for key in rows[0] if key not in ("arm", "mode", "seed", "split", "modality", "origin")]
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ("arm", "mode", "split", "modality"))].append(row)
    aggregate = []
    for group, values in grouped.items():
        item = dict(zip(("arm", "mode", "split", "modality"), group))
        item["seeds"] = len(values)
        for key in keys:
            numbers = [v[key] for v in values if v[key] is not None]
            item[key + "_mean"] = float(np.mean(numbers)) if numbers else None
            item[key + "_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        aggregate.append(item)
    common.write_csv(directory / "aggregate.csv", list(aggregate[0]), aggregate)
    lookup = {(r["arm"], r["mode"], r["seed"], r["split"], r["modality"]): r for r in rows}
    deltas = []
    for target, reference in (("B", "A"), ("C", "B"), ("C", "A")):
        for row in rows:
            old = lookup.get((reference, row["mode"], row["seed"], row["split"], row["modality"]))
            if row["arm"] != target or old is None:
                continue
            item = {"comparison": target + "-" + reference, **{k: row[k] for k in ("mode", "seed", "split", "modality")}}
            for key in keys:
                item[key + "_delta"] = row[key] - old[key] if row[key] is not None and old[key] is not None else None
            deltas.append(item)
    if deltas:
        common.write_csv(directory / "paired_deltas.csv", list(deltas[0]), deltas)
    return aggregate, deltas


def run(args, data, users, pairs, source, source_hashes, *, synthetic=False):
    started = time.perf_counter()
    args.temperature, args.data_seed = float(source["temperature"]), int(source["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log, trials, schedules, initial_hashes, current = common.Log(args.output_dir), [], {}, {}, None
    try:
        common.write_json(args.output_dir / "status.json", {"state": "running", "synthetic_test_only": synthetic})
        if not synthetic:
            for name in previous.MANIFESTS:
                shutil.copy2(args.source_run / name, args.output_dir / name)
        paths = [Path(__file__), Path(base.__file__), Path(previous.__file__), Path(common.__file__),
                 Path(contrastive.__file__), Path(base.architectures.__file__), ROOT / "Thresholding_Model_512.py",
                 ROOT / "thresholding/variability_model.py", ROOT / "Thresholding Model.py",
                 ROOT / "scripts/train_reliability_distance.py", ROOT / "scripts/train_thresholding_ablation.py",
                 ROOT / "scripts/bch_evaluator.py", ROOT / "scripts/fusion_bch_bridge.cpp",
                 ROOT / "BCH/BCH.cpp", ROOT / "BCH/BCH.hpp"]
        code_hashes = {str(path): common.sha256(path) for path in paths}
        common.write_json(args.output_dir / "config.json", {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "descriptions": DESCRIPTIONS,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "torch": str(torch.__version__),
            "synthetic_test_only": synthetic, "input_sha256": data.hashes, "source_sha256": source_hashes, "code_sha256": code_hashes,
            "test_used_for_selection": False, "reliability_trained": False, "FE_key_evaluation": False,
            "selection": "Baseline separation/balance/FAR_h guards, then mean unique exact BCH recovery, within-t rate, lower mean Hamming",
            "positive_loss_scaling": "B divides by 512; C divides excess distance by t before squaring. C changes both shape and gradient scale.",
            "requires_new_enrollment": True, "checkpoint_resuming_supported": False})
        log.message(f"결과: {args.output_dir}\nBCH t={args.bch_t}, C 목표={args.bch_t-args.safety_margin} bit | device={args.device}")
        log.message("A=soft / B=hard STE 평균거리 / C=hard STE 정정범위 손실. 신뢰도 학습은 포함하지 않습니다.")
        log.message("ETA는 남은 최대 epoch 기준 추정이며 조기 종료·마지막 시험 평가에 따라 달라집니다.")
        bch = BCHEvaluator(args.output_dir / "bch_evaluator", args.bch_t)
        cache = {m: torch.tensor(values, device=args.device) for m, values in data.arrays.items()}
        plans = {s: contrastive.evaluation_plan(pairs[s], users, "cpu") for s in ("validation", "test")}
        zero = base.make_model(args, "zero", args.seeds[0])
        baseline_progress = common.Progress("tau=0 검증", 1, 0, args.log_interval)
        steps = evaluation_steps(pairs["validation"], args.batch_size, args.bch_batch_size)
        baseline, _, _, _ = evaluate(zero, cache, pairs["validation"], users, plans["validation"], args, bch,
            on_step=lambda k: baseline_progress.update(1, k, steps, "validation+BCH", None))
        baseline_progress.finish_line()
        baselines = {"validation": baseline}
        common.write_json(args.output_dir / "zero_validation.json", baseline)
        jobs = [(seed, mode, arm) for seed in args.seeds for mode in args.modes for arm in args.arms]
        log.message(f"{len(jobs)}회 순차 실행 | train={len(pairs['train'])}, validation={len(pairs['validation'])}, test={len(pairs['test'])} 쌍")
        for j, (seed, mode, arm) in enumerate(jobs):
            current = args.output_dir / f"{mode}_{arm}_seed{seed}"
            current.mkdir()
            common.write_json(current / "status.json", {"state": "training"})
            model = base.make_model(args, mode, seed)
            digest, key = state_digest(model), f"{mode}:seed{seed}"
            if key in initial_hashes and initial_hashes[key] != digest:
                raise RuntimeError("A/B/C initial states differ")
            initial_hashes[key] = digest
            training = train_trial(model, arm, seed, data, users, cache, pairs["validation"], plans["validation"], baseline,
                args, bch, current, schedules, (len(jobs) - j - 1) * args.epochs)
            trial = {"arm": arm, "mode": mode, "seed": seed, "threshold": training,
                     "checkpoint": str(current / "best_threshold.pt"), "initial_state_sha256": digest}
            trials.append(trial)
            common.write_json(current / "training_summary.json", trial)
            common.write_json(current / "status.json", {"state": "trained_awaiting_test"})
            common.write_json(args.output_dir / "training_summary.json", trials)
            common.write_json(args.output_dir / "shared_schedule_sha256.json", schedules)
            log.message(f"학습 완료 {j+1}/{len(jobs)} | {mode}/{arm} seed={seed} | 선택 epoch={training['best_epoch']}")
        log.message("모든 체크포인트 선택 완료. 선택된 모델의 비트·tau를 저장하고 시험 분할을 평가합니다.")
        baseline_test, e, q, outcomes = evaluate(zero, cache, pairs["test"], users, plans["test"], args, bch)
        baselines["test"] = baseline_test
        common.write_json(args.output_dir / "zero_test.json", baseline_test)
        export_details(args.output_dir, "zero_test", pairs["test"], data, e, q, outcomes)
        for index, trial in enumerate(trials):
            current = Path(trial["checkpoint"]).parent
            model, _ = ThresholdingModel512.load_checkpoint(trial["checkpoint"], args.device)
            progress = common.Progress(f"최종평가 {trial['mode']}/{trial['arm']} seed={trial['seed']}", 2,
                                       2 * (len(trials) - index - 1), args.log_interval)
            for epoch, split in enumerate(("validation", "test"), 1):
                steps = evaluation_steps(pairs[split], args.batch_size, args.bch_batch_size)
                result, e, q, outcomes = evaluate(model, cache, pairs[split], users, plans[split], args, bch,
                    on_step=lambda k: progress.update(epoch, k, steps, split + "+BCH", None))
                trial[split] = result
                export_details(current, split, pairs[split], data, e, q, outcomes)
            progress.finish_line()
            if trial["validation"] != trial["threshold"]["validation"]:
                raise RuntimeError("Reloaded checkpoint differs from selected validation metrics")
            common.write_json(current / "report.json", trial)
            common.write_json(current / "status.json", {"state": "completed"})
        aggregate, deltas = write_tables(args.output_dir, trials, baselines)
        data.assert_unchanged()
        for mapping in (source_hashes, code_hashes):
            if any(common.sha256(Path(path)) != digest for path, digest in mapping.items()):
                raise RuntimeError("Source or code changed during the experiment")
        report = {"trials": trials, "baselines": baselines, "aggregate": aggregate, "paired_deltas": deltas,
            "seconds": time.perf_counter() - started, "synthetic_test_only": synthetic, "input_and_source_unchanged": True,
            "shared_schedule_sha256": schedules, "initial_state_sha256": initial_hashes,
            "limitations": ["Exact BCH recovery compares recovered biometric bits, not FE cryptographic keys.",
                "FAR_h is the empirical fraction of negative pairs within t, not measured FE false acceptance.",
                "Different-person fingerprint FAR is unavailable when a split contains only one participant.",
                "Only 7 fingerprint participants (5/1/1); random seeds do not add independent participants.",
                "Previously inspected test split: exploratory comparison, not independent confirmation.",
                "Virtual face/finger correspondences are not actual same-person multimodal samples.",
                "Selected checkpoints require new enrollment; reliability heads remain uncalibrated.",
                "B/C use a biased surrogate gradient; convergence or within-t recovery is not guaranteed."]}
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
    p.add_argument("--source-run", type=Path, default=ROOT / "thresholding/results/train_20261029_162319_486882")
    p.add_argument("--embedding-dir", type=Path, help="Relocated identical embedding archives only")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--modes", nargs="+", choices=("fixed", "dynamic"), default=["fixed"])
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    for name, value in {"epochs": 100, "patience": 15, "batch_size": 64, "bch_batch_size": 32,
                         "hidden_dim": 32, "threads": 2, "log_interval": 10, "bch_t": 32, "safety_margin": 4}.items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=value)
    for name, value in {"threshold_lr": 1e-4, "tau_base_bound": .01, "tau_delta_bound": .01, "grad_clip": 5.,
            "min_bit_gain": .1, "tau_stability_weight": .01, "tau_residual_weight": .001,
            "max_impostor_drop_bits": 5., "max_impostor_p05_drop_bits": 5., "max_auroc_drop": .001,
            "max_genuine_increase_bits": .5, "max_genuine_p95_increase_bits": 5., "max_row_outside_increase": .01,
            "max_constant_increase": .02, "max_coordinate_balance_increase": .005, "max_far_h_increase": 0.}.items():
        p.add_argument("--" + name.replace("_", "-"), type=float, default=value)
    for name, value in asdict(contrastive.LossSettings()).items():
        p.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    for key in ("epochs", "patience", "bch_batch_size", "hidden_dim", "threads", "log_interval"):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.batch_size < 2 or not 1 <= args.bch_t <= 255 or not 0 <= args.safety_margin < args.bch_t:
        p.error("batch-size >= 2, 1 <= bch-t <= 255, and 0 <= safety-margin < bch-t required")
    for key in ("seeds", "arms", "modes"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            p.error(f"Duplicate {key}")
    if not all(0 <= seed < 2**32 for seed in args.seeds):
        p.error("Seeds must be in [0, 2**32)")
    positive = {"threshold_lr", "tau_base_bound", "tau_delta_bound", "grad_clip"}
    for key, value in vars(args).items():
        if isinstance(value, float) and (not math.isfinite(value) or value < 0 or (key in positive and value == 0)):
            p.error(f"Invalid {key}")
    if args.max_far_h_increase > 1:
        p.error("max-far-h-increase must be <= 1")
    try:
        contrastive.loss_settings(args)
    except ValueError as error:
        p.error(str(error))
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
        ROOT / "thresholding/results" / datetime.now().strftime("bch_abc_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT / name for name in ("scripts", "muct-master", "UareU", "BCH", "feature_extractors")] + [args.source_run]
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    nested = any((parent / "config.json").is_file() and (parent / "status.json").is_file() for parent in args.output_dir.parents)
    if args.output_dir.exists() or nested or any(args.output_dir.is_relative_to(path.resolve()) for path in protected):
        p.error("Output must be a new directory outside code, inputs and existing experiments")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print("기존 임베딩·사람 단위 분할·등록/질의 쌍을 확인합니다.", flush=True)
    data, users, pairs, source, hashes = previous.prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output cannot be inside the input embeddings")
    if args.check_only:
        data.assert_unchanged()
        print(json.dumps({"check_only": True, "training_started": False, "outputs_created": False,
            "arms": args.arms, "modes": args.modes, "seeds": args.seeds, "bch_t": args.bch_t,
            "C_target_bits": args.bch_t - args.safety_margin, "temperature": source["temperature"],
            "pairs": {s: len(rows) for s, rows in pairs.items()},
            "people": {s: {m: len({u.face_person if m == "face" else u.fingerprint_person for u in users if u.split == s})
                           for m in MODALITIES} for s in common.SPLITS},
            "cuda_available": torch.cuda.is_available(), "g++_available": shutil.which("g++") is not None}, ensure_ascii=False, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Explicitly use --device cpu if desired.")
    run(args, data, users, pairs, source, hashes)


if __name__ == "__main__":
    main()
