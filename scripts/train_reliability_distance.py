#!/usr/bin/env python3
"""Compare monotone distance reliability with the previous frozen MLP experiment.

Reuses a completed source run's exact splits, virtual users and evaluation pairs.
Only four reliability parameters are trained with BCE; tau and hard bits stay
unchanged. This is a controlled reliability experiment, not a revised tau loss.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
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
from scripts import train_thresholding as common
from thresholding.distance_reliability import DistanceThresholdingModel

MODALITIES = common.MODALITIES


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def prepare_source(args):
    """Refuse incompatible sources rather than silently change the comparison."""
    source = args.source_run
    config = json.loads((source / "config.json").read_text())
    if json.loads((source / "status.json").read_text()).get("state") != "completed":
        raise ValueError("--source-run must be a completed original Thresholding experiment")
    for path, key in ((ROOT / "Thresholding Model.py", "model_sha256"),
                      (ROOT / "scripts/train_thresholding.py", "trainer_sha256")):
        if common.sha256(path) != config[key]:
            raise ValueError(f"Source experiment code has changed: {path}")
    embedding_dir = args.embedding_dir or Path(config["embedding_dir"])
    data = common.Embeddings(embedding_dir)
    expected_hashes = {Path(path).name: digest for path, digest in config["input_sha256"].items()}
    if {Path(path).name: digest for path, digest in data.hashes.items()} != expected_hashes:
        raise ValueError("Embeddings differ from the source experiment")
    users, partitions = common.make_users(data, int(config["split_seed"]))
    expected_users = [{"virtual_user_index": str(i), **asdict(user)} for i, user in enumerate(users)]
    if read_csv(source / "virtual_users.csv") != expected_users:
        raise ValueError("Source virtual-user mapping cannot be reproduced exactly")
    saved_splits = json.loads((source / "splits.json").read_text())
    if any(saved_splits[key] != value for key, value in partitions.items()):
        raise ValueError("Source participant splits differ")
    pairs = {}
    for split in common.SPLITS:
        rows = common.make_pairs(data, users, split, int(config["seed"]), epoch=1 if split == "train" else 0)
        expected = [{"virtual_user_index": str(int(row[common.USER])),
                     "enrollment_face": data.files["face"][int(row[common.EF])],
                     "query_face": data.files["face"][int(row[common.QF])],
                     "enrollment_fingerprint": data.files["fingerprint"][int(row[common.EP])],
                     "query_fingerprint": data.files["fingerprint"][int(row[common.QP])]}
                    for row in rows]
        if read_csv(source / f"{split}_pairs.csv") != expected:
            raise ValueError(f"Source {split} pairs cannot be reproduced exactly")
        pairs[split] = rows
    filenames = ("config.json", "status.json", "splits.json", "virtual_users.csv", "samples.csv",
                 "train_pairs.csv", "validation_pairs.csv", "test_pairs.csv", "best_threshold.pt", "final_model.pt")
    source_hashes = {str(source / name): common.sha256(source / name) for name in filenames}
    return data, users, pairs, config, source_hashes


@torch.no_grad()
def training_statistics(model, cache, rows, batch_size):
    """Fit conditioning scale and constant-probability baseline from TRAIN only."""
    sums = {m: {"distance": 0.0, "agreement": 0.0, "bits": 0} for m in MODALITIES}
    for offset in range(0, len(rows), batch_size):
        e, q = common.forward_pair(model, cache, rows[offset:offset + batch_size])
        for m in MODALITIES:
            y = (e[f"{m}_bits"] == q[f"{m}_bits"])
            sums[m]["distance"] += float(e[f"{m}_distance"].sum() + q[f"{m}_distance"].sum())
            sums[m]["agreement"] += 2 * int(y.sum())
            sums[m]["bits"] += 2 * y.numel()
    return {m: {"mean_distance": values["distance"] / values["bits"],
                "agreement_rate": values["agreement"] / values["bits"],
                "labeled_bit_observations": values["bits"]} for m, values in sums.items()}


def roc_auc(labels, scores):
    """Exact empirical ROC AUC with average ranks for ties; no sklearn required."""
    labels, scores = np.asarray(labels, dtype=np.bool_), np.asarray(scores)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    mean_ranks = (starts + ends + 1) / 2
    positive_counts = np.add.reduceat(sorted_labels.astype(np.int64), starts)
    return float(((mean_ranks * positive_counts).sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


@torch.no_grad()
def evaluate(model, cache, rows, stats, batch_size, old_model=None, detailed=False, on_batch=None):
    if len(rows) == 0:
        raise ValueError("Evaluation requires at least one enrollment/query pair")
    model.eval()
    totals = {m: {"bce": 0.0, "brier": 0.0, "agreements": 0.0, "bits": 0,
                  "old_bce": 0.0, "old_brier": 0.0, "tau_equal": True, "bits_equal": True}
              for m in MODALITIES}
    arrays = {m: {key: [] for key in ("label", "logit", "probability", "distance", "old_logit")}
              for m in MODALITIES}
    for step, offset in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[offset:offset + batch_size]
        outputs = common.forward_pair(model, cache, batch)
        reference = common.forward_pair(old_model, cache, batch) if old_model is not None else None
        e, q = outputs
        for m in MODALITIES:
            target = (e[f"{m}_bits"] == q[f"{m}_bits"]).float()
            for role, output in enumerate(outputs):
                logits, probability = output[f"{m}_reliability_logits"], output[f"{m}_reliability"]
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Nonfinite reliability logits")
                values = totals[m]
                values["bce"] += float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
                values["brier"] += float((probability - target).square().sum())
                values["agreements"] += float(target.sum())
                values["bits"] += target.numel()
                if detailed:
                    for name, value in (("label", target.bool()), ("logit", logits),
                                        ("probability", probability), ("distance", output[f"{m}_distance"])):
                        arrays[m][name].append(value.flatten().cpu().numpy())
                if reference is not None:
                    old = reference[role]
                    values["tau_equal"] &= torch.equal(output["tau"], old["tau"])
                    values["bits_equal"] &= torch.equal(output[f"{m}_bits"], old[f"{m}_bits"])
                    values["old_bce"] += float(F.binary_cross_entropy_with_logits(
                        old[f"{m}_reliability_logits"], target, reduction="sum"))
                    values["old_brier"] += float((old[f"{m}_reliability"] - target).square().sum())
                    if detailed:
                        arrays[m]["old_logit"].append(old[f"{m}_reliability_logits"].flatten().cpu().numpy())
        if on_batch:
            on_batch(step)
    result = {"pair_count": len(rows), "modalities": {}}
    for m, values in totals.items():
        count, positive = values["bits"], values["agreements"]
        constant = stats[m]["agreement_rate"]
        metrics = {"bce": values["bce"] / count, "brier": values["brier"] / count,
                   "agreement_rate": positive / count, "train_constant_probability": constant,
                   "constant_bce": -(positive * math.log(constant) + (count - positive) * math.log1p(-constant)) / count,
                   "constant_brier": (positive * (1 - constant) ** 2 + (count - positive) * constant ** 2) / count}
        if old_model is not None:
            if not values["tau_equal"] or not values["bits_equal"]:
                raise RuntimeError("Tau or hard bits differ from the source final model; comparison is invalid")
            metrics.update({"old_mlp_bce": values["old_bce"] / count,
                            "old_mlp_brier": values["old_brier"] / count,
                            "tau_and_hard_bits_equal_to_source": True})
        if detailed:
            vectors = {name: np.concatenate(items) for name, items in arrays[m].items() if items}
            metrics.update({"reliability_logit_auroc": roc_auc(vectors["label"], vectors["logit"]),
                            "reliability_probability_auroc": roc_auc(vectors["label"], vectors["probability"]),
                            "raw_distance_auroc": roc_auc(vectors["label"], vectors["distance"]),
                            "probability_mean": float(vectors["probability"].mean()),
                            "probability_std": float(vectors["probability"].std())})
            if "old_logit" in vectors:
                metrics["old_mlp_auroc"] = roc_auc(vectors["label"], vectors["old_logit"])
        result["modalities"][m] = metrics
    result["reliability_bce"] = sum(result["modalities"][m]["bce"] for m in MODALITIES) / 2
    result["constant_bce"] = sum(result["modalities"][m]["constant_bce"] for m in MODALITIES) / 2
    if old_model is not None:
        result["old_mlp_bce"] = sum(result["modalities"][m]["old_mlp_bce"] for m in MODALITIES) / 2
    return result


def assert_fixed_state(model, fixed_state):
    current = model.state_dict()
    if any(not torch.equal(current[name], value) for name, value in fixed_state.items()):
        raise RuntimeError("Frozen tau parameters or training distance scales changed")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("A frozen parameter received a gradient")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed original experiment directory")
    parser.add_argument("--embedding-dir", type=Path, help="Optional relocated archives with identical hashes")
    parser.add_argument("--output-dir", type=Path, help="New directory only; default is timestamped")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.00001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17, help="Batch-order seed; source split/pair seeds remain fixed")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    for name in ("batch_size", "epochs", "patience", "threads", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >= 1")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    for name in ("lr", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("min-delta must be nonnegative and finite")
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else ROOT / "thresholding/results" /
                       datetime.now().strftime("reliability_distance_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT / name for name in ("muct-master", "UareU", "feature_extractors", "BCH", "scripts")]
    protected += [args.source_run]
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    if any(args.output_dir.is_relative_to(path.resolve()) for path in protected):
        parser.error("Output must be separate from source data/code and the source experiment")
    return args


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the existing launcher or explicitly pass --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log, started = common.Log(args.output_dir), time.perf_counter()
    model, progress = None, None
    try:
        common.write_json(args.output_dir / "status.json", {"state": "running"})
        log.message(f"결과 경로: {args.output_dir}\n기존 분할·쌍 확인 → τ 고정 → 거리 기반 신뢰도 학습 → 비교 평가")
        data, users, pairs, source_config, source_hashes = prepare_source(args)
        for name in ("splits.json", "virtual_users.csv", "samples.csv", "train_pairs.csv", "validation_pairs.csv", "test_pairs.csv"):
            shutil.copy2(args.source_run / name, args.output_dir / name)
        cache = {m: torch.tensor(array, device=args.device) for m, array in data.arrays.items()}
        model = DistanceThresholdingModel.from_threshold_checkpoint(args.source_run / "best_threshold.pt", args.device)
        old_module = common.load_model_module()
        old_model = old_module.ThresholdingModel.load_checkpoint(args.source_run / "final_model.pt", args.device)
        for name, parameter in model.named_parameters():
            if "reliability_head" not in name and not torch.equal(parameter, old_model.state_dict()[name]):
                raise ValueError("Source best-threshold and final-model tau parameters differ")
        stats = training_statistics(model, cache, pairs["train"], args.batch_size)
        for modality in MODALITIES:
            getattr(model, f"{modality}_reliability_head").initialize_from_training(
                stats[modality]["mean_distance"], stats[modality]["agreement_rate"])
        model.set_training_stage("reliability")
        trainable_count = sum(p.numel() for p in model.trainable_parameters())
        if trainable_count != 4:
            raise RuntimeError("Expected exactly four trainable calibration scalars")
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        fixed_state = {name: value.detach().clone() for name, value in model.state_dict().items()
                       if name not in trainable_names}
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"started_at_utc": datetime.now(timezone.utc).isoformat(), "torch": str(torch.__version__),
                       "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                       "trainable_parameters": trainable_count, "training_statistics": stats,
                       "source_pair_seed": source_config["seed"], "source_split_seed": source_config["split_seed"],
                       "source_file_sha256": source_hashes, "input_sha256": data.hashes,
                       "trainer_sha256": common.sha256(__file__),
                       "distance_model_sha256": common.sha256(ROOT / "thresholding/distance_reliability.py"),
                       "initial_heads": model.head_settings(),
                       "formula": "sigmoid((softplus(raw_slope)+1e-6)*abs(x-tau)/train_mean_distance + intercept)",
                       "objective": "unweighted genuine bit-agreement BCE, mean over two modalities and two roles",
                       "checkpoint_format": model.checkpoint_format,
                       "checkpoint_resuming_supported": False,
                       "test_interpretation": "comparison on an already inspected test set, not fresh independent confirmation"})
        common.write_json(args.output_dir / "config.json", config)
        log.message(f"학습 파라미터 {trainable_count}개 | train {len(pairs['train'])}쌍 / "
                    f"validation {len(pairs['validation'])}쌍 / test {len(pairs['test'])}쌍")
        log.message("거리의 순위는 유지하고 확률 변환을 학습합니다. ETA는 조기 종료 전 최대 epoch 기준입니다.")
        initial = evaluate(model, cache, pairs["validation"], stats, args.batch_size, old_model, detailed=True)
        best_score, best_epoch, stale = initial["reliability_bce"], 0, 0
        common.checkpoint(model, args.output_dir / "best_reliability.pt")
        log.record({"stage": "distance_reliability", "epoch": 0, "validation": initial, "best": True})
        log.message(f"초기 val BCE {best_score:.6f} | 기존 MLP {initial['old_mlp_bce']:.6f} | "
                    f"고정 확률 {initial['constant_bce']:.6f}")
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
        progress = common.Progress("distance_reliability", args.epochs, 0, args.log_interval)
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            rows = common.make_pairs(data, users, "train", int(source_config["seed"]), epoch)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch, 9001]))
            order = rng.permutation(len(rows))
            batches = [order[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
            # BCE uses genuine pairs only; no negative sampling or padding needed.
            steps = len(batches) + math.ceil(len(pairs["validation"]) / args.batch_size)
            total_loss, total_pairs = 0.0, 0
            for step, indices in enumerate(batches, start=1):
                optimizer.zero_grad(set_to_none=True)
                e, q = common.forward_pair(model, cache, rows[indices])
                targets = old_module.reliability_targets(e, q)
                loss = old_module.reliability_loss(e, q, targets)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training BCE")
                loss.backward()
                if any(p.grad is None for p in model.trainable_parameters()):
                    raise RuntimeError("Missing reliability gradient")
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                total_pairs += len(indices)
                progress.update(epoch, step, steps, "train", total_loss / total_pairs,
                                force=step == 1 or step == len(batches))
            validation = evaluate(model, cache, pairs["validation"], stats, args.batch_size,
                                  on_batch=lambda n: progress.update(epoch, len(batches) + n, steps,
                                                                     "validation", total_loss / total_pairs))
            assert_fixed_state(model, fixed_state)
            improved = validation["reliability_bce"] < best_score - args.min_delta
            if improved:
                best_score, best_epoch, stale = validation["reliability_bce"], epoch, 0
                common.checkpoint(model, args.output_dir / "best_reliability.pt")
            else:
                stale += 1
            common.checkpoint(model, args.output_dir / "last_reliability.pt")
            log.record({"stage": "distance_reliability", "epoch": epoch, "train_loss": total_loss / total_pairs,
                        "validation": validation, "best": improved, "best_epoch": best_epoch,
                        "heads": model.head_settings(), "seconds": time.perf_counter() - epoch_start})
            progress.finish_line()
            log.message(f"[epoch {epoch}/{args.epochs}] train BCE {total_loss / total_pairs:.6f} | "
                        f"val BCE {validation['reliability_bce']:.6f} | best epoch {best_epoch} | "
                        f"patience {stale}/{args.patience}")
            if stale >= args.patience:
                log.message("검증 BCE 개선이 멈춰 조기 종료합니다.")
                break
        model = DistanceThresholdingModel.load_checkpoint(args.output_dir / "best_reliability.pt", args.device)
        assert_fixed_state(model, fixed_state)
        common.checkpoint(model, args.output_dir / "final_model.pt")
        log.message("선택된 모델을 기존 MLP·거리·고정 확률 기준과 비교합니다. 시험 결과로 모델을 고르지 않습니다.")
        best_validation = evaluate(model, cache, pairs["validation"], stats, args.batch_size, old_model, detailed=True)
        test = evaluate(model, cache, pairs["test"], stats, args.batch_size, old_model, detailed=True)
        data.assert_unchanged()
        if any(common.sha256(path) != digest for path, digest in source_hashes.items()):
            raise RuntimeError("Source experiment files changed during the run")
        report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
                  "epochs_run": epoch, "best_epoch": best_epoch, "best_selection_bce": best_score,
                  "initial_validation": initial, "best_validation": best_validation, "test": test,
                  "heads": model.head_settings(), "tau_path_and_distance_scales_unchanged": True,
                  "source_files_and_embeddings_unchanged": True,
                  "limitations": ["Four-parameter monotone calibration preserves raw-distance ranking; it cannot improve its ideal AUROC.",
                                  "This reuses an already inspected test set for comparison, not independent confirmation.",
                                  "Synthetic multimodal pairing; 7 actual fingerprint participants, not 276 independent finger participants.",
                                  "Hard bits and tau stay identical; this does not repair the failed tau training or measure FE key recovery."]}
        common.write_json(args.output_dir / "report.json", report)
        common.write_json(args.output_dir / "status.json", {"state": "completed", "seconds": report["seconds"]})
        log.message(f"완료 | {common.duration(report['seconds'])} | 선택 epoch {best_epoch} | "
                    f"test BCE {test['reliability_bce']:.6f} (기존 MLP {test['old_mlp_bce']:.6f})")
        log.message(f"최종 모델: {args.output_dir / 'final_model.pt'}\n결과: {args.output_dir / 'report.json'}")
    except BaseException as error:
        if progress is not None:
            progress.finish_line()
        if model is not None:
            try:
                common.checkpoint(model, args.output_dir / "interrupted.pt")
            except Exception:
                pass
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        common.write_json(args.output_dir / "status.json", {"state": state, "error": str(error)})
        with (args.output_dir / "error.log").open("w", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        log.message(f"\n{state}: 저장된 최적 체크포인트와 로그를 유지합니다.")
        raise


if __name__ == "__main__":
    main()
