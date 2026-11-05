#!/usr/bin/env python3
"""Compare zero, globally learned 512-D, and input-dependent 512-D thresholds.

Frozen 512-D embeddings; original person-disjoint splits; train tau then freeze
it and fit genuine-bit reliability. --check-only performs no training/writing.
No feature extraction, image modification, Fusion, BCH decoding or key recovery.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
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
from scripts.common import (data as protocol, losses as loss_ops, metrics as metric_ops,
                            training as train_ops, reporting, provenance)
from scripts.common.bch_evaluator import BCHEvaluator
from Thresholding_Model_512 import ThresholdingModel512
MODALITIES = common.MODALITIES
make_model = train_ops.make_coordinate_model
threshold_loss = loss_ops.coordinate_threshold_loss
bit_metrics = metric_ops.coordinate_bit_metrics
selection_guard = metric_ops.coordinate_selection_guard

from Thresholding_Model_512 import ThresholdingModel512, MODES, MODALITIES

DESCRIPTIONS = {"zero": "tau=0; same reliability head as learned modes",
                "fixed": "512 globally learned coordinate thresholds per modality; shared by all users",
                "dynamic": "512 global base thresholds + input-dependent 512-D residual per modality"}


def train_stage(model, stage, data, users, cache, validation, plan, baseline, args, seed,
                directory, log, schedules, future):
    is_tau = stage == "threshold"
    model.set_stage(stage)
    frozen = model.fixed_reliability_state() if not is_tau else None
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.threshold_lr if is_tau else args.reliability_lr,
                                 weight_decay=0 if is_tau else args.reliability_weight_decay)
    epochs = args.threshold_epochs if is_tau else args.reliability_epochs
    e, q = train_ops.collect(model, cache, validation, args.batch_size, reliability=not is_tau)
    best = bit_metrics(e, q, plan, args) if is_tau else metric_ops.evaluate_reliability(model, e, q, detailed=False)
    initial, best_epoch, stale = best, 0, 0
    best_path = directory / ("best_threshold.pt" if is_tau else "best_model.pt")
    meta = {"mode": model.config.mode, "seed": seed, "stage": stage, "epoch": 0,
            "origin": "fixed_zero_fallback" if is_tau else "initial_prior"}
    model.save_checkpoint(best_path, meta)
    log.record({"stage": stage, "epoch": 0, "validation": best, "selected": True})
    progress = common.Progress(f"{model.config.mode} seed={seed} {stage}", epochs, future, args.log_interval)
    for epoch in range(1, epochs + 1):
        model.train()
        batches, padding = train_ops.make_schedule(data, users, args, seed, epoch, stage, schedules)
        steps, sums, count = len(batches)+math.ceil(len(validation)/args.batch_size), defaultdict(float), 0
        for step, (batch, _) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            e, q = common.forward_pair(model, cache, batch)
            losses = (threshold_loss(model, e, q, batch, users, args) if is_tau
                      else {"total": loss_ops.reliability_loss(e, q)})
            if not bool(torch.isfinite(losses["total"])):
                raise RuntimeError(f"Nonfinite {stage} loss")
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            for key, value in losses.items():
                sums[key] += float(value.detach())*len(batch)
            sums["gradient_norm"] += float(gradient_norm)*len(batch)
            count += len(batch)
            progress.update(epoch, step, steps, "train", sums["total"]/count)
        if frozen is not None:
            train_ops.assert_fixed(model, frozen)
        e, q = train_ops.collect(model, cache, validation, args.batch_size, reliability=not is_tau,
                                on_batch=lambda k: progress.update(epoch, len(batches)+k, steps, "validation", sums["total"]/count))
        metrics = bit_metrics(e, q, plan, args) if is_tau else metric_ops.evaluate_reliability(model, e, q, detailed=False)
        eligible, reasons = selection_guard(metrics, baseline, args) if is_tau else (True, [])
        improved = eligible and (metric_ops.threshold_better(metrics, best, args) if is_tau else
                                metrics["selection_bce"] < best["selection_bce"] - args.min_delta)
        if improved:
            best, best_epoch, stale = metrics, epoch, 0
            model.save_checkpoint(best_path, {**meta, "epoch": epoch, "origin": "trained_network"})
        else:
            stale += 1
        model.save_checkpoint(directory / f"last_{stage}.pt", {**meta, "epoch": epoch, "origin": "last_epoch"})
        progress.finish_line()
        log.record({"stage": stage, "epoch": epoch, "train_loss": {k: v/count for k, v in sums.items()},
                    "padding": padding, "validation": metrics, "eligible": eligible, "reasons": reasons,
                    "selected": improved, "threshold_frozen": not is_tau})
        score = (f"얼굴 {metrics['bits']['face']['genuine_hamming_bits']:.3f}, "
                 f"지문 {metrics['bits']['fingerprint']['genuine_hamming_bits']:.3f} bit" if is_tau
                 else f"val BCE {metrics['selection_bce']:.6f}")
        log.message(f"{model.config.mode} seed={seed} {stage} epoch {epoch} | {score} | best {best_epoch}")
        if stale >= args.patience:
            break
    selected, chosen_meta = ThresholdingModel512.load_checkpoint(best_path, args.device)
    if frozen is not None:
        train_ops.assert_fixed(selected, frozen)
        selected.reliability_fitted.fill_(best_epoch > 0)
        selected.save_checkpoint(best_path, {**chosen_meta, "reliability_fitted": best_epoch > 0})
    return selected, {"best_epoch": best_epoch, "epochs_run": epoch, "origin": chosen_meta["origin"],
                      "initial_validation": initial, "validation": best}


def evaluate_trial(model, cache, rows, plan, args, directory, split, data):
    e, q = train_ops.collect(model, cache, rows, args.batch_size, reliability=True)
    result = {"bits": bit_metrics(e, q, plan, args), "reliability": metric_ops.evaluate_reliability(model, e, q),
              "reliability_fitted": bool(model.reliability_fitted), "unique_pair_diagnostics": {}}
    for m, columns in (("face", [common.EF, common.QF]), ("fingerprint", [common.EP, common.QP])):
        _, unique = np.unique(rows[:, columns], axis=0, return_index=True)
        y = (e[f"{m}_bits"][unique] == q[f"{m}_bits"][unique]).numpy()
        logits = q[f"{m}_reliability_logits"][unique].numpy()
        aucs = [metric_ops.roc_auc(a, b) for a, b in zip(y, logits)]
        aucs = [v for v in aucs if v is not None]
        result["unique_pair_diagnostics"][m] = {
            "pair_count": len(unique), "genuine_hamming_bits": float((~y).sum(1).mean()),
            "query_within_pair_auroc_mean": float(np.mean(aucs)) if aucs else None,
            "query_within_pair_auroc_defined_pairs": len(aucs),
            "query_lowest10_error_precision_mean": float(np.mean([
                metric_ops.low_confidence_precision(~a, b, .1) for a, b in zip(y, logits)]))}
    records = []
    for i, row in enumerate(rows):
        item = {"pair_index": i, "virtual_user_index": int(row[common.USER])}
        for m, ec, qc in (("face", common.EF, common.QF), ("fingerprint", common.EP, common.QP)):
            item.update({f"{m}_enrollment_file": data.files[m][int(row[ec])],
                         f"{m}_query_file": data.files[m][int(row[qc])],
                         f"{m}_hamming_bits": int((e[f"{m}_bits"][i] != q[f"{m}_bits"][i]).sum()),
                         f"{m}_tau_pair_mae": float((e[f"{m}_tau"][i]-q[f"{m}_tau"][i]).abs().mean())})
            for role, out in (("enrollment", e), ("query", q)):
                for key, value in (("mean", out[f"{m}_tau"][i].mean()), ("min", out[f"{m}_tau"][i].min()),
                                   ("max", out[f"{m}_tau"][i].max()), ("std", out[f"{m}_tau"][i].std(unbiased=False))):
                    item[f"{m}_{role}_tau_{key}"] = float(value)
        records.append(item)
    common.write_csv(directory/f"{split}_pairs_metrics.csv", list(records[0]), records)
    np.savez_compressed(directory/f"{split}_outputs.npz", **{
        f"{role}_{key}": value.numpy() for role, out in (("enrollment", e), ("query", q)) for key, value in out.items()})
    common.write_json(directory/f"{split}_tau_coordinates.json", {
        m: {role: {"mean": out[f"{m}_tau"].mean(0).tolist(),
                   "std_across_observations": out[f"{m}_tau"].std(0, unbiased=False).tolist()}
            for role, out in (("enrollment", e), ("query", q))} for m in MODALITIES})
    return result


def run(args, data, users, pairs, source, source_hashes, *, synthetic=False):
    started = time.perf_counter()
    args.temperature, args.data_seed = float(source["temperature"]), int(source["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log, trials, schedules, directory = common.Log(args.output_dir), [], {}, None
    try:
        common.write_json(args.output_dir/"status.json", {"state": "running", "synthetic_test_only": synthetic})
        if not synthetic:
            for name in protocol.MANIFESTS:
                shutil.copy2(args.source_run/name, args.output_dir/name)
        code_hashes = provenance.code_hashes(__file__)
        common.write_json(args.output_dir/"config.json", {
            **{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "torch": str(torch.__version__), "descriptions": DESCRIPTIONS, "synthetic_test_only": synthetic,
            "input_sha256": data.hashes, "source_sha256": source_hashes, "code_sha256": code_hashes,
            "test_used_for_selection": False, "checkpoint_resuming_supported": False, "FE_key_evaluation": False,
            "threshold_shape": ["batch",512], "reliability_architecture": "previous ABC B head for BOTH modalities",
            "selection": "guarded hard validation radius rates then mean Hamming; reliability: pooled genuine BCE",
            "loss": "preceding D loss + normalized pair tau consistency + dynamic residual penalty"})
        cache = {m: torch.tensor(v, device=args.device) for m,v in data.arrays.items()}
        plans = {s: metric_ops.evaluation_plan(pairs[s], users, "cpu") for s in ("validation", "test")}
        zero = make_model(args, "zero", args.seeds[0])
        e, q = train_ops.collect(zero, cache, pairs["validation"], args.batch_size)
        baseline = bit_metrics(e, q, plans["validation"], args)
        common.write_json(args.output_dir/"zero_validation.json", baseline)
        jobs = [(seed, mode) for seed in args.seeds for mode in args.modes]
        log.message(f"결과: {args.output_dir}\n{len(jobs)}회 순차 실행 | device={args.device}")
        log.message("zero: 0 고정 / fixed: 공통 512차원 학습 / dynamic: 입력별 512차원 예측. 얼굴·지문은 독립입니다.")
        log.message("ETA는 현재 단계 속도와 남은 최대 epoch 기준 추정입니다. 조기 종료·최종 평가 시간에 따라 달라집니다.")
        log.message(f"쌍 수: train={len(pairs['train'])}, validation={len(pairs['validation'])}, test={len(pairs['test'])}")
        for j, (seed, mode) in enumerate(jobs):
            directory = args.output_dir/f"{mode}_seed{seed}"
            directory.mkdir()
            trial_log = common.Log(directory)
            common.write_json(directory/"status.json", {"state": "running"})
            future = sum(args.reliability_epochs + (args.threshold_epochs if m != "zero" else 0) for _,m in jobs[j+1:])
            model = make_model(args, mode, seed)
            threshold = {"best_epoch": 0, "epochs_run": 0, "origin": "fixed_zero", "validation": baseline}
            if mode != "zero":
                model, threshold = train_stage(model, "threshold", data, users, cache, pairs["validation"],
                                               plans["validation"], baseline, args, seed, directory, trial_log,
                                               schedules, future+args.reliability_epochs)
            stats = train_ops.fit_statistics(model, cache, data, users, pairs["train"], args)
            for m in MODALITIES:
                model.reliability[m].set_statistics(stats[m])
            common.write_json(directory/"train_statistics.json", stats)
            common.write_json(directory/"threshold_summary.json", threshold)
            model, reliability = train_stage(model, "reliability", data, users, cache, pairs["validation"],
                                             plans["validation"], baseline, args, seed, directory, trial_log, schedules, future)
            validation = evaluate_trial(model, cache, pairs["validation"], plans["validation"], args, directory, "validation", data)
            if mode == "zero" and validation["bits"] != baseline:
                raise RuntimeError("Reliability training changed the zero-threshold baseline")
            trial = {"arm": mode, "seed": seed, "checkpoint": str(directory/"best_model.pt"),
                     "threshold": threshold, "reliability_training": reliability, "validation": validation,
                     "parameter_counts": {"threshold": sum(p.numel() for p in model.thresholds.parameters()) if mode != 'zero' else 0,
                                          "reliability": sum(p.numel() for p in model.reliability.parameters())}}
            trials.append(trial)
            common.write_json(directory/"summary.json", trial)
            common.write_json(directory/"status.json", {"state": "trained_awaiting_test"})
            common.write_json(args.output_dir/"training_summary.json", trials)
            log.message(f"학습 완료 {j+1}/{len(jobs)} | {mode} seed={seed} | tau best={threshold['best_epoch']}, r best={reliability['best_epoch']}")
        log.message("모든 체크포인트 선택 완료. 이제 시험 분할을 평가합니다.")
        for trial in trials:
            directory = Path(trial["checkpoint"]).parent
            model, _ = ThresholdingModel512.load_checkpoint(trial["checkpoint"], args.device)
            trial["test"] = evaluate_trial(model, cache, pairs["test"], plans["test"], args, directory, "test", data)
            common.write_json(directory/"report.json", trial)
            common.write_json(directory/"status.json", {"state": "completed"})
            log.message(f"시험 완료: {trial['arm']} seed={trial['seed']}")
        rows, aggregate, _ = reporting.reliability_tables(trials)
        for name, values in (("comparison.csv",rows), ("aggregate.csv",aggregate)):
            common.write_csv(args.output_dir/name, list(values[0]), values)
        deltas = []
        lookup = {(r['arm'],r['seed'],r['split'],r['modality']):r for r in rows}
        for mode, reference in (("fixed","zero"), ("dynamic","zero"), ("dynamic","fixed")):
            for row in rows:
                base = lookup.get((reference,row['seed'],row['split'],row['modality']))
                if row['arm'] != mode or base is None:
                    continue
                item = {"comparison": f"{mode}-{reference}", **{k:row[k] for k in ('seed','split','modality')}}
                for key in ('genuine_hamming_bits','within_radius_fraction','impostor_hamming_bits',
                            'query_bce','query_brier','query_agreement_auroc','query_error_average_precision'):
                    item[f"{key}_delta"] = row[key]-base[key] if row[key] is not None and base[key] is not None else None
                deltas.append(item)
        if deltas:
            common.write_csv(args.output_dir/'paired_deltas.csv', list(deltas[0]), deltas)
        data.assert_unchanged()
        for mapping in (source_hashes, code_hashes):
            if any(common.sha256(path) != digest for path,digest in mapping.items()):
                raise RuntimeError("Source/code changed during the experiment")
        report = {"trials": trials, "aggregate": aggregate, "paired_deltas": deltas,
                  "seconds": time.perf_counter()-started, "synthetic_test_only": synthetic,
                  "input_and_source_unchanged": True, "shared_schedule_sha256": schedules,
                  "limitations": ["7 actual fingerprint people; 5 train / 1 validation / 1 test. Seeds do not add people.",
                                  "Face/fingerprint correspondences are virtual. Fingerprint captures repeat across virtual users.",
                                  "Previously inspected test split: exploratory evidence, not independent confirmation.",
                                  "Each mode requires new enrollment; old registered bits and keys are not preserved.",
                                  "Within-t rate measures bit errors only; no actual BCH decoding, FE, cryptographic key or helper evaluation.",
                                  "Reliability is marginal bit agreement, not a helper-conditioned flip probability.",
                                  "All modes use the previous B reliability head, including the zero baseline.",
                                  "A 512-D threshold changes binary labels; compare reliability metrics with hard-bit errors."]}
        common.write_json(args.output_dir/'report.json', report)
        common.write_json(args.output_dir/'status.json', {"state":"completed", "seconds":report['seconds'], "synthetic_test_only":synthetic})
        log.message(f"완료 | {common.duration(report['seconds'])} | {args.output_dir/'comparison.csv'}")
        return report
    except BaseException as error:
        status = {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)}
        common.write_json(args.output_dir/'status.json', status)
        if directory is not None:
            common.write_json(directory/'status.json', status)
        (args.output_dir/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source-run', type=Path, default=ROOT/'thresholding/results/train_20261029_162319_486882')
    p.add_argument('--embedding-dir', type=Path, help='Relocated identical archives only')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--seeds', nargs='+', type=int, default=[17,29,43])
    for name, value in {'threshold_epochs':100, 'reliability_epochs':60, 'patience':15, 'batch_size':64,
                        'hidden_dim':32, 'threads':2, 'log_interval':10, 'bch_t':8}.items():
        p.add_argument('--'+name.replace('_','-'), type=int, default=value)
    for name, value in {'threshold_lr':1e-4, 'reliability_lr':1e-3, 'tau_base_bound':.01, 'tau_delta_bound':.01,
                        'variance_floor':1e-4, 'grad_clip':5., 'min_bit_gain':.1, 'min_delta':1e-5,
                        'tau_stability_weight':.01, 'tau_residual_weight':.001, 'variance_shrinkage':.25,
                        'reliability_weight_decay':0., 'max_impostor_drop_bits':5., 'max_impostor_p05_drop_bits':5.,
                        'max_auroc_drop':.001, 'max_genuine_increase_bits':.5, 'max_genuine_p95_increase_bits':5.,
                        'max_row_outside_increase':.01, 'max_constant_increase':.02,
                        'max_coordinate_balance_increase':.005}.items():
        p.add_argument('--'+name.replace('_','-'), type=float, default=value)
    for name, value in asdict(loss_ops.LossSettings()).items():
        p.add_argument('--'+name.replace('_','-'), type=type(value), default=value)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args(argv)
    for key in ('threshold_epochs','reliability_epochs','patience','hidden_dim','threads','log_interval'):
        if getattr(args,key) < 1:
            p.error(f'{key} must be positive')
    if args.batch_size < 2 or not 1 <= args.bch_t <= 255:
        p.error('batch-size >= 2 and 1 <= bch-t <= 255 required')
    if len(set(args.seeds)) != len(args.seeds) or not all(0 <= s < 2**32 for s in args.seeds) or len(set(args.modes)) != len(args.modes):
        p.error('Unique modes and unique seeds in [0,2**32) required')
    positive = {'threshold_lr','reliability_lr','tau_base_bound','tau_delta_bound','variance_floor','grad_clip'}
    for key, value in vars(args).items():
        if isinstance(value,float) and (not math.isfinite(value) or value < 0 or (key in positive and value == 0)):
            p.error(f'Invalid {key}')
    if args.variance_shrinkage > 1:
        p.error('variance-shrinkage must be <= 1')
    try:
        loss_ops.loss_settings(args)
    except ValueError as error:
        p.error(str(error))
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT/'thresholding/results'/datetime.now().strftime('threshold512_%Y%m%d_%H%M%S_%f'))
    protected = [ROOT/name for name in ('scripts','muct-master','UareU','BCH','feature_extractors')]+[args.source_run]
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    nested_experiment = any((parent/'config.json').is_file() and (parent/'status.json').is_file()
                            for parent in args.output_dir.parents)
    if args.output_dir.exists() or nested_experiment or any(args.output_dir.is_relative_to(x.resolve()) for x in protected):
        p.error('Output must be a new directory outside input data, code and existing experiments')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available() and not args.check_only:
        raise RuntimeError('CUDA unavailable. Use --device cpu or run in an environment with GPU access.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print('기존 임베딩·사람 단위 분할·등록/질의 쌍을 검증합니다.', flush=True)
    data, users, pairs, source, hashes = provenance.prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError('Output cannot be inside input embeddings')
    if args.check_only:
        data.assert_unchanged()
        print(json.dumps({'check_only':True, 'training_started':False, 'outputs_created':False,
                          'modes':args.modes, 'seeds':args.seeds, 'threshold_shape':['batch',512],
                          'pairs':{s:len(rows) for s,rows in pairs.items()}}, ensure_ascii=False, indent=2))
        return
    run(args, data, users, pairs, source, hashes)


if __name__ == '__main__':
    main()
