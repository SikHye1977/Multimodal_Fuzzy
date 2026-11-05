#!/usr/bin/env python3
"""Controlled tau-loss ablation on the existing frozen 512-D embeddings.

Three independent arms: DSH-style pair loss, + bit balance, + VICReg-style
variance. This adapts their principles; it does not reproduce either paper.
No feature-extractor, reliability-head, FE, or joint-helper training occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from scripts import train_thresholding as common
from scripts.train_reliability_distance import prepare_source, roc_auc

MODALITIES = common.MODALITIES
ARMS = ("dsh", "dsh_balance", "dsh_balance_variance")
MANIFESTS = ("splits.json", "virtual_users.csv", "samples.csv", "train_pairs.csv",
             "validation_pairs.csv", "test_pairs.csv")
REFERENCES = {
    "dsh": "https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf",
    "balance": "https://papers.nips.cc/paper_files/paper/2008/file/d58072be2820e8682c0a27c0518e805e-Paper.pdf",
    "vicreg": "https://arxiv.org/abs/2105.04906",
}


@dataclass(frozen=True)
class LossSettings:
    face_margin: float = 0.25
    fingerprint_margin: float = 0.25
    negative_weight: float = 1.0
    balance_weight: float = 1.0
    variance_weight: float = 1.0
    std_target: float = 0.1
    std_epsilon: float = 1e-6

    def __post_init__(self):
        for name in ("face_margin", "fingerprint_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("negative_weight", "balance_weight", "variance_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.std_target) or not 0 < self.std_target < 0.5:
            raise ValueError("std_target must be in (0, 0.5) for probabilities")
        if not math.isfinite(self.std_epsilon) or not 0 < self.std_epsilon < self.std_target ** 2:
            raise ValueError("std_epsilon must be positive and smaller than std_target squared")


def identity_weights(rows, users, device):
    """Equal total weight per observed identity for balance/variance only.

    A face identity is a person; a fingerprint identity is (person, finger).
    Repeated virtual assignments do not create additional fingerprint identities.
    IDs only determine statistical weights, never model inputs.
    """
    result = {}
    for m in MODALITIES:
        labels = [users[int(row[common.USER])].face_person if m == "face"
                  else users[int(row[common.USER])].finger for row in rows]
        counts = Counter(labels)
        result[m] = torch.tensor([1 / (len(counts) * counts[x]) for x in labels],
                                 dtype=torch.float32, device=device)
    return result


def probability_statistics(p, weights, settings):
    mean = (weights[:, None] * p).sum(0)
    variance = (weights[:, None] * (p - mean).square()).sum(0)
    std = torch.sqrt(variance + settings.std_epsilon)
    return (mean - 0.5).square().mean(), F.relu(settings.std_target - std).mean()


def compute_losses(enrollment, query, negatives, weights, settings, arm):
    """L1 soft distance equals normalized Hamming on hard bits.

    DSH-inspired, NOT the original squared-Euclidean DSH + quantization loss.
    Both genuine and negative pairs use the SAME distance. Negative hinge is
    divided by its margin before squaring, so identical impostor codes cost 1.
    Balance and population-STD penalties are calculated separately per role,
    then averaged; opposite enrollment/query biases cannot cancel each other.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    terms = {key: [] for key in ("positive", "negative", "balance", "variance", "negative_active_fraction")}
    for m in MODALITIES:
        e, q = enrollment[f"{m}_soft_bits"], query[f"{m}_soft_bits"]
        if e.ndim != 2 or e.shape != q.shape or e.shape[1] != 512 or len(e) < 2:
            raise ValueError("Expected matching soft bit tensors (B, 512), B >= 2")
        if negatives.dtype != torch.long or negatives.shape != (len(e),):
            raise ValueError("Expected one negative index per enrollment")
        if bool((negatives == torch.arange(len(e), device=e.device)).any()):
            raise ValueError("Self pairs cannot be negatives")
        w = weights[m]
        if w.shape != (len(e),) or not bool((w > 0).all()) or not torch.isclose(w.sum(), w.new_tensor(1.0)):
            raise ValueError("Identity weights must be positive and sum to one")
        genuine = (e - q).abs().mean(1)
        different = (e - q[negatives]).abs().mean(1)
        margin = getattr(settings, f"{m}_margin")
        terms["positive"].append(genuine.mean())
        terms["negative"].append((F.relu(margin - different) / margin).square().mean())
        terms["negative_active_fraction"].append((different < margin).float().mean())
        eb, ev = probability_statistics(e, w, settings)
        qb, qv = probability_statistics(q, w, settings)
        terms["balance"].append((eb + qb) / 2)
        terms["variance"].append((ev + qv) / 2)
    result = {key: torch.stack(value).mean() for key, value in terms.items()}
    result["weighted_negative"] = settings.negative_weight * result["negative"]
    result["weighted_balance"] = (settings.balance_weight if arm != "dsh" else 0) * result["balance"]
    result["weighted_variance"] = (settings.variance_weight if arm == "dsh_balance_variance" else 0) * result["variance"]
    result["total"] = (result["positive"] + result["weighted_negative"]
                       + result["weighted_balance"] + result["weighted_variance"])
    return result


def selection_score(metrics):
    return sum(metrics["bits"][m]["genuine_hamming_fraction"] for m in MODALITIES) / 2


def selection_guard(metrics, initial, args):
    """Validation only. A better genuine mean cannot excuse severe separation loss."""
    reasons = []
    for m in MODALITIES:
        current, reference = metrics["bits"][m], initial["bits"][m]
        if current["impostor_hamming_bits"] < reference["impostor_hamming_bits"] - args.max_impostor_drop_bits:
            reasons.append(f"{m}: impostor mean dropped by more than {args.max_impostor_drop_bits:g} bits")
        if current["extreme_code_fraction"] > reference["extreme_code_fraction"] + args.max_extreme_increase + 1e-9:
            reasons.append(f"{m}: excessive increase in codes with <5% or >95% ones")
    return not reasons, reasons


@torch.no_grad()
def evaluate(model, cache, rows, users, args, settings, arm, on_batch=None, fixed_tau=False):
    model.eval()
    collected = ({}, {})
    for step, offset in enumerate(range(0, len(rows), args.batch_size), 1):
        batch = rows[offset:offset + args.batch_size]
        if fixed_tau:
            ef, ep, qf, qp = common.model_inputs(cache, batch)
            outputs = []
            for face, finger in ((ef, ep), (qf, qp)):
                out = {"tau": torch.zeros((len(batch), 1), device=face.device)}
                for m, x in (("face", face), ("fingerprint", finger)):
                    out[f"{m}_bits"] = (x >= 0).to(torch.uint8)
                    out[f"{m}_soft_bits"] = torch.sigmoid(x / model.config.temperature)
                outputs.append(out)
        else:
            outputs = common.forward_pair(model, cache, batch)
        for destination, output in zip(collected, outputs):
            for key in ("tau", "face_bits", "fingerprint_bits", "face_soft_bits", "fingerprint_soft_bits"):
                destination.setdefault(key, []).append(output[key])
        if on_batch:
            on_batch(step)
    e, q = ({key: torch.cat(value) for key, value in role.items()} for role in collected)
    if not all(bool(torch.isfinite(x).all()) for role in (e, q) for x in role.values()):
        raise RuntimeError("Nonfinite evaluation output")
    negative = torch.as_tensor(common.negative_permutation(rows, users), device=e["tau"].device)
    weights = identity_weights(rows, users, e["tau"].device)
    losses = compute_losses(e, q, negative, weights, settings, arm)
    bits = common.bit_statistics(e, q, negative)
    for m in MODALITIES:
        eh, qh = e[f"{m}_bits"], q[f"{m}_bits"]
        hard = torch.cat((eh, qh)).float()
        soft = torch.cat((e[f"{m}_soft_bits"], q[f"{m}_soft_bits"]))
        row_ones, coordinate_ones = hard.mean(1), hard.mean(0)
        genuine = (eh != qh).float().sum(1).cpu().numpy()
        impostor = (eh != qh[negative]).float().sum(1).cpu().numpy()
        bits[m].update({
            "impostor_hamming_p05_bits": float(np.quantile(impostor, 0.05)),
            "genuine_zero_error_fraction": float(np.mean(genuine == 0)),
            "extreme_code_fraction": float(((row_ones < 0.05) | (row_ones > 0.95)).float().mean()),
            "per_code_ones_min": float(row_ones.min()), "per_code_ones_max": float(row_ones.max()),
            "constant_coordinate_fraction": float(((coordinate_ones == 0) | (coordinate_ones == 1)).float().mean()),
            "soft_coordinate_std_mean": float(soft.std(0, unbiased=False).mean()),
            "soft_coordinate_std_below_target_fraction": float((soft.std(0, unbiased=False) < settings.std_target).float().mean()),
            "genuine_vs_impostor_auroc": roc_auc(np.r_[np.ones(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)],
                                               -np.r_[genuine, impostor]),
        })
    tau = {role: {"mean": float(out["tau"].mean()), "std": float(out["tau"].std(unbiased=False)),
                  "min": float(out["tau"].min()), "max": float(out["tau"].max())}
           for role, out in (("enrollment", e), ("query", q))}
    tau["genuine_mean_absolute_difference"] = float((e["tau"] - q["tau"]).abs().mean())
    result = {"pair_count": len(rows), "loss": {key: float(value) for key, value in losses.items()},
              "bits": bits, "tau": tau}
    result["selection_genuine_fraction"] = selection_score(result)
    return result


def save_checkpoint(model, path, arm, seed, epoch):
    """Original loader compatible, but reliability heads are explicitly untrained."""
    payload = {"format": model.checkpoint_format, "config": asdict(model.config),
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "purpose": "threshold_only_ablation; reliability_heads_untrained; not_for_enrollment",
               "experiment": {"arm": arm, "seed": seed, "epoch": epoch, "reliability_calibrated": False}}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def schedule_digest(batches):
    digest = hashlib.sha256()
    for rows, negative in batches:
        digest.update(rows.astype("<i8").tobytes())
        digest.update(negative.astype("<i8").tobytes())
    return digest.hexdigest()


def train_arm(module, initial_state, cache, data, users, validation, args, settings,
              arm, seed, directory, future_trials, schedules):
    directory.mkdir()
    log = common.Log(directory)
    common.write_json(directory / "status.json", {"state": "running"})
    model = module.ThresholdingModel(module.ThresholdingConfig(args.temperature, args.tau_bound))
    model.load_state_dict(initial_state, strict=True)
    if state_digest(model.state_dict()) != state_digest(initial_state):
        raise RuntimeError("Ablation arms must start from identical weights")
    model.to(args.device).set_training_stage("threshold")
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    parameters = list(model.trainable_parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    initial = evaluate(model, cache, validation, users, args, settings, arm)
    best, best_epoch, best_metrics, stale = selection_score(initial), 0, initial, 0
    best_path = directory / "best_threshold.pt"
    save_checkpoint(model, best_path, arm, seed, 0)
    log.record({"epoch": 0, "validation": initial, "eligible": True, "best": True, "best_epoch": 0})
    log.message(f"\n[{arm}, seed={seed}] 시작 | 초기 genuine 평균 {best * 512:.3f} bit")
    progress = common.Progress(f"{arm} seed={seed}", args.epochs, future_trials * args.epochs, args.log_interval)
    gradient_blocks, epoch = {}, 0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            # Initialization seeds do not alter the shared data schedule.
            rows = common.make_pairs(data, users, "train", args.data_seed, epoch)
            rng = np.random.default_rng(np.random.SeedSequence([args.data_seed, epoch, 9001]))
            batches, padded = common.training_batches(rows, users, args.batch_size, rng)
            digest = schedule_digest(batches)
            if schedules.setdefault(epoch, digest) != digest:
                raise RuntimeError("Training pairs/batch order/negatives differ between trials")
            val_steps = math.ceil(len(validation) / args.batch_size)
            steps, sums, processed = len(batches) + val_steps, {}, 0
            for step, (batch, negative) in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                e, q = common.forward_pair(model, cache, batch)
                negatives = torch.as_tensor(negative, dtype=torch.long, device=args.device)
                losses = compute_losses(e, q, negatives, identity_weights(batch, users, args.device), settings, arm)
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Nonfinite loss: {arm} epoch={epoch} step={step}")
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
                if epoch == step == 1:
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and parameter.grad is None:
                            raise RuntimeError(f"Missing gradient: {name}")
                        if parameter.grad is not None:
                            block = name.split(".")[0]
                            gradient_blocks[block] = gradient_blocks.get(block, 0.0) + float(parameter.grad.norm())
                    if not math.isfinite(float(norm)):
                        raise RuntimeError("Nonfinite gradient")
                optimizer.step()
                for name, value in losses.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(batch)
                processed += len(batch)
                progress.update(epoch, step, steps, "train", sums["total"] / processed,
                                force=step == 1 or step == len(batches))
            metrics = evaluate(model, cache, validation, users, args, settings, arm,
                               on_batch=lambda n: progress.update(epoch, len(batches) + n, steps, "validation", sums["total"] / processed))
            for name, parameter in model.named_parameters():
                if name in frozen and (parameter.grad is not None or not torch.equal(parameter.detach(), frozen[name])):
                    raise RuntimeError(f"Frozen reliability parameter changed: {name}")
            eligible, reasons = selection_guard(metrics, initial, args)
            score = selection_score(metrics)
            improved = eligible and score < best - args.min_delta
            if improved:
                best, best_epoch, best_metrics, stale = score, epoch, metrics, 0
                save_checkpoint(model, best_path, arm, seed, epoch)
            else:
                stale += 1
            save_checkpoint(model, directory / "last_threshold.pt", arm, seed, epoch)
            log.record({"epoch": epoch, "train_loss": {k: v / processed for k, v in sums.items()},
                        "validation": metrics, "eligible": eligible, "ineligible_reasons": reasons,
                        "best": improved, "best_epoch": best_epoch, "original_pair_count": len(rows),
                        "processed_pairs": processed, "supplementary_pairs": padded,
                        "schedule_sha256": digest, "epoch_seconds": time.perf_counter() - epoch_started})
            progress.finish_line()
            detail = " | ".join(f"{m}: 동일 {metrics['bits'][m]['genuine_hamming_bits']:.2f} / "
                                f"상이 {metrics['bits'][m]['impostor_hamming_bits']:.2f} bit / "
                                f"1 비율 {metrics['bits'][m]['ones_fraction']:.3f}" for m in MODALITIES)
            log.message(f"[{arm} seed={seed} {epoch}/{args.epochs}] loss {sums['total'] / processed:.5f} | "
                        f"선택 가능 {eligible} | best epoch {best_epoch} | patience {stale}/{args.patience}\n  {detail}")
            if reasons:
                log.message("  선택 제외: " + "; ".join(reasons))
            if stale >= args.patience:
                log.message("검증 개선이 없어 이 조건의 학습을 조기 종료합니다.")
                break
    except BaseException as error:
        progress.finish_line()
        try:
            save_checkpoint(model, directory / "interrupted.pt", arm, seed, epoch)
        except Exception:
            pass
        common.write_json(directory / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                      "error": str(error)})
        raise
    finally:
        progress.finish_line()
    if best_epoch == 0:
        log.message("초기 모델보다 선택 조건을 만족하는 개선이 없어 epoch 0을 유지합니다.")
    result = {"arm": arm, "seed": seed, "epochs_run": epoch, "best_epoch": best_epoch,
              "initial_state_sha256": state_digest(initial_state), "initial_validation": initial,
              "best_validation": best_metrics, "best_selection_score": best,
              "first_step_gradient_norms": gradient_blocks, "seconds": time.perf_counter() - progress.start,
              "checkpoint": str(best_path), "reliability_heads_trained": False}
    common.write_json(directory / "summary.json", result)
    common.write_json(directory / "status.json", {"state": "trained_awaiting_test"})
    return result


def comparison_rows(trials, baselines):
    result = []
    for split in ("validation", "test"):
        sources = [("tau_zero", "", 0, baselines[split])]
        for trial in trials:
            sources.append((trial["arm"], trial["seed"], trial["best_epoch"],
                            trial["best_validation"] if split == "validation" else trial["test"]))
        for arm, seed, epoch, metrics in sources:
            for m in MODALITIES:
                values = metrics["bits"][m]
                result.append({"split": split, "arm": arm, "seed": seed, "best_epoch": epoch, "modality": m,
                               **{key: values[key] for key in ("genuine_hamming_bits", "impostor_hamming_bits",
                                   "genuine_hamming_p95_bits", "impostor_hamming_p05_bits", "ones_fraction",
                                   "extreme_code_fraction", "constant_coordinate_fraction", "soft_coordinate_std_mean",
                                   "genuine_vs_impostor_auroc")},
                               "tau_enrollment_mean": metrics["tau"]["enrollment"]["mean"],
                               "tau_query_mean": metrics["tau"]["query"]["mean"],
                               "tau_genuine_mean_absolute_difference": metrics["tau"]["genuine_mean_absolute_difference"]})
    return result


def aggregate_results(rows):
    result = []
    keys = ("genuine_hamming_bits", "impostor_hamming_bits", "ones_fraction", "genuine_vs_impostor_auroc")
    for split in ("validation", "test"):
        for arm in ARMS:
            for m in MODALITIES:
                selected = [r for r in rows if r["split"] == split and r["arm"] == arm and r["modality"] == m]
                result.append({"split": split, "arm": arm, "modality": m, "initialization_seed_count": len(selected),
                               **{f"{key}_{stat}": float(getattr(np, stat)([r[key] for r in selected]))
                                  for key in keys for stat in ("mean", "std")}})
    # Matched-seed differences isolate the effect of adding each regularizer.
    deltas = []
    for split in ("validation", "test"):
        for before, after in zip(ARMS[:-1], ARMS[1:]):
            for m in MODALITIES:
                previous = {r["seed"]: r for r in rows if r["arm"] == before and r["split"] == split and r["modality"] == m}
                current = {r["seed"]: r for r in rows if r["arm"] == after and r["split"] == split and r["modality"] == m}
                deltas.append({"split": split, "change": f"{after} minus {before}", "modality": m,
                               **{f"{key}_delta_mean": float(np.mean([current[s][key] - previous[s][key] for s in previous]))
                                  for key in keys}})
    return result, deltas


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-run", type=Path, default=ROOT / "thresholding/results/train_20261029_162319_486882")
    parser.add_argument("--embedding-dir", type=Path, help="Optional relocated archives, identical hashes required")
    parser.add_argument("--output-dir", type=Path, help="New directory only; default is timestamped")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43], help="Paired model initialization seeds; data split stays fixed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum improvement in mean genuine Hamming fraction")
    parser.add_argument("--face-margin", type=float, default=0.25)
    parser.add_argument("--fingerprint-margin", type=float, default=0.25)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--std-target", type=float, default=0.1)
    parser.add_argument("--max-impostor-drop-bits", type=float, default=5.0, help="Per-modality validation guard relative to epoch 0")
    parser.add_argument("--max-extreme-increase", type=float, default=0.01, help="Allowed increase in extreme-code fraction versus epoch 0")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args(argv)
    for name in ("epochs", "patience", "threads", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >= 1")
    if args.batch_size < 2 or min(args.seeds) < 0 or max(args.seeds) >= 2 ** 32 or len(set(args.seeds)) != len(args.seeds):
        parser.error("batch_size >= 2 and unique seeds in [0, 2**32) are required")
    for name in ("lr", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    for name in ("min_delta", "max_impostor_drop_bits", "max_extreme_increase"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative and finite")
    if args.max_extreme_increase > 1 or args.max_impostor_drop_bits > 512:
        parser.error("Invalid selection-guard range")
    try:
        LossSettings(**{key: getattr(args, key) for key in LossSettings.__dataclass_fields__ if hasattr(args, key)})
    except ValueError as error:
        parser.error(str(error))
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT / "thresholding/results" / datetime.now().strftime("tau_ablation_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT / name for name in ("muct-master", "UareU", "feature_extractors", "BCH", "scripts")]
    protected += [args.source_run]
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    if any(args.output_dir.is_relative_to(p.resolve()) for p in protected):
        parser.error("Output must be separate from source data, embeddings, code and source run")
    if args.output_dir.exists():
        parser.error("Output directory already exists; existing experiments are never overwritten")
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the existing environment or explicitly pass --device cpu")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print("원본 실험의 전체 임베딩·분할·가상 대응·평가 쌍을 확인합니다.", flush=True)
    data, users, pairs, source_config, source_hashes = prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output must not be inside the input embedding directory")
    args.data_seed = int(source_config["seed"])
    args.temperature = float(source_config["temperature"])
    args.tau_bound = float(source_config["tau_bound"])
    settings = LossSettings(**{key: getattr(args, key) for key in LossSettings.__dataclass_fields__ if hasattr(args, key)})
    module = common.load_model_module()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = common.Log(args.output_dir)
    try:
        common.write_json(args.output_dir / "status.json", {"state": "running"})
        for name in MANIFESTS:
            shutil.copy2(args.source_run / name, args.output_dir / name)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"started_at_utc": datetime.now(timezone.utc).isoformat(), "arms": list(ARMS),
                       "loss_settings": asdict(settings), "source_sha256": source_hashes,
                       "input_sha256": data.hashes, "trainer_sha256": common.sha256(__file__),
                       "source_validator_sha256": common.sha256(ROOT / "scripts/train_reliability_distance.py"),
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                       "references": REFERENCES, "paper_reproduction": False,
                       "selection": "min mean genuine hard-Hamming fraction subject to per-modality validation guards versus epoch 0",
                       "loss_statistics": "identity-weighted population statistics separately per modality and enrollment/query role",
                       "pair_weighting": "uniform observed pairs; same negative permutation as source",
                       "reliability_training": False, "checkpoint_resuming_supported": False,
                       "test_used_for_selection": False})
        common.write_json(args.output_dir / "config.json", config)
        cache = {m: torch.tensor(value, device=args.device) for m, value in data.arrays.items()}
        log.message(f"결과: {args.output_dir}\n3조건 × {len(args.seeds)}초기화 seed = {3 * len(args.seeds)}회 독립 학습")
        log.message("각 조건은 같은 초기 가중치와 학습 순서를 사용합니다. 신뢰도 학습은 포함하지 않습니다.")
        log.message("ETA는 남은 최대 epoch 기준이며 조기 종료·초기화·최종 시험 시간에 따라 달라집니다.")
        partitions = json.loads((args.source_run / "splits.json").read_text())
        for split in common.SPLITS:
            people = partitions["fingerprint_people"][split]
            log.message(f"{split}: 얼굴 {len(partitions['face_people'][split])}명 / 지문 참여자 {len(people)}명 {people} / {len(pairs[split])}쌍")
        trials, schedules, baselines = [], {}, {}
        total_trials = len(args.seeds) * len(ARMS)
        for seed in args.seeds:
            torch.manual_seed(seed)
            initial_model = module.ThresholdingModel(module.ThresholdingConfig(args.temperature, args.tau_bound))
            initial_state = {name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()}
            if not baselines:
                baselines["validation"] = evaluate(initial_model.to(args.device), cache, pairs["validation"], users,
                                                   args, settings, "dsh", fixed_tau=True)
                common.write_json(args.output_dir / "tau_zero_validation.json", baselines["validation"])
            for arm in ARMS:
                result = train_arm(module, initial_state, cache, data, users, pairs["validation"], args,
                                   settings, arm, seed, args.output_dir / f"{arm}_seed{seed}",
                                   total_trials - len(trials) - 1, schedules)
                trials.append(result)
                common.write_json(args.output_dir / "training_summary.json", trials)
        # No test inference until ALL arms have completed validation selection.
        log.message("\n모든 조건의 모델 선택 완료. 선택된 모델마다 시험 분할을 한 번 평가합니다.")
        for trial in trials:
            model = module.ThresholdingModel.load_checkpoint(Path(trial["checkpoint"]), args.device)
            trial["test"] = evaluate(model, cache, pairs["test"], users, args, settings, trial["arm"])
            directory = Path(trial["checkpoint"]).parent
            common.write_json(directory / "report.json", trial)
            common.write_json(directory / "status.json", {"state": "completed"})
            log.message(f"시험 완료: {trial['arm']} seed={trial['seed']} | best epoch={trial['best_epoch']}")
        baselines["test"] = evaluate(model, cache, pairs["test"], users, args, settings, "dsh", fixed_tau=True)
        data.assert_unchanged()
        if any(common.sha256(path) != digest for path, digest in source_hashes.items()):
            raise RuntimeError("Source experiment changed during this comparison")
        rows = comparison_rows(trials, baselines)
        aggregates, deltas = aggregate_results(rows)
        common.write_csv(args.output_dir / "comparison.csv", list(rows[0]), rows)
        common.write_csv(args.output_dir / "aggregate.csv", list(aggregates[0]), aggregates)
        common.write_csv(args.output_dir / "paired_deltas.csv", list(deltas[0]), deltas)
        report = {"trials": trials, "tau_zero": baselines, "aggregate": aggregates, "paired_deltas": deltas,
                  "seconds": time.perf_counter() - started, "input_and_source_unchanged": True,
                  "shared_training_schedule_sha256": schedules,
                  "limitations": ["DSH-inspired soft-L1 margin loss, not original DSH reproduction; no quantization or covariance term.",
                                  "Frozen extractors, one scalar tau shared by two modalities; restricted representational capacity.",
                                  "Only 7 real fingerprint participants. Same person's fingers never cross splits.",
                                  "Validation/test each contain one fingerprint person; impostors there are different fingers, not different people.",
                                  "Synthetic face/finger mappings, with repeated fingers; initialization seeds are not independent participant folds.",
                                  "Previously inspected test split is reused; results are exploratory, not a new untouched test.",
                                  "No reliability calibration, FE key-recovery evaluation or cryptographic entropy claim.",
                                  "Guard defaults and loss weights are starting choices, not established optimal hyperparameters."]}
        common.write_json(args.output_dir / "report.json", report)
        common.write_json(args.output_dir / "status.json", {"state": "completed", "seconds": report["seconds"]})
        log.message(f"완료 | 경과 {common.duration(report['seconds'])} | 비교표: {args.output_dir / 'comparison.csv'}")
        for row in aggregates:
            if row["split"] == "test":
                log.message(f"{row['arm']} / {row['modality']}: 동일 {row['genuine_hamming_bits_mean']:.2f} ± "
                            f"{row['genuine_hamming_bits_std']:.2f}, 상이 {row['impostor_hamming_bits_mean']:.2f} ± "
                            f"{row['impostor_hamming_bits_std']:.2f} bit (초기화 seed 간 평균 ± 모집단 표준편차)")
    except BaseException as error:
        common.write_json(args.output_dir / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                          "error": str(error)})
        (args.output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
