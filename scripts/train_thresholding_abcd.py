#!/usr/bin/env python3
"""Run four independent, matched-seed scalar-tau experiments: A / B / C / D.

A: pair distance + coordinate balance.
B: A + within-code occupancy penalty.
C: B + identity-balanced multi-negative / nearest-identity mining.
D: C + a narrower scalar tau bound.

Uses complete frozen embedding archives and the original person's splits.
Training starts only in main(); --help and imports never run an experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from scripts import train_thresholding_ablation as previous
from scripts.train_reliability_distance import prepare_source, roc_auc

ARMS = ("A", "B", "C", "D")
MODALITIES = common.MODALITIES
DESCRIPTIONS = {
    "A": "pair_distance_and_coordinate_balance",
    "B": "A_plus_within_code_occupancy",
    "C": "B_plus_multi_negative_and_nearest_identity_mining",
    "D": "C_plus_narrow_scalar_tau_bound",
}
REFERENCES = {
    "dsh_principle_not_reproduction": "https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf",
    "batch_negative_mining_principle": "https://openaccess.thecvf.com/content_cvpr_2016/html/Song_Deep_Metric_Learning_CVPR_2016_paper.html",
}


@dataclass(frozen=True)
class LossSettings:
    face_margin: float = 0.25
    fingerprint_margin: float = 0.25
    negative_weight: float = 1.0
    balance_weight: float = 1.0
    row_weight: float = 1.0
    row_lower: float = 0.3
    hard_mix: float = 0.5
    hard_k: int = 4

    def __post_init__(self):
        for name in ("face_margin", "fingerprint_margin"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("negative_weight", "balance_weight", "row_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.row_lower) or not 0 < self.row_lower < 0.5:
            raise ValueError("row_lower must be in (0,0.5)")
        if not math.isfinite(self.hard_mix) or not 0 < self.hard_mix < 1:
            raise ValueError("hard_mix must be in (0,1), preserving both ordinary and hard negatives")
        if not isinstance(self.hard_k, int) or self.hard_k < 1:
            raise ValueError("hard_k must be a positive integer")


def loss_settings(args):
    return LossSettings(**{key: getattr(args, key) for key in LossSettings.__dataclass_fields__})


def identity_codes(rows, users, device):
    """IDs only define labels and weights; they never enter the network."""
    result = {}
    for m in MODALITIES:
        labels = [users[int(i)].face_person if m == "face" else users[int(i)].finger
                  for i in rows[:, common.USER]]
        lookup = {value: i for i, value in enumerate(sorted(set(labels)))}
        result[m] = torch.tensor([lookup[value] for value in labels], dtype=torch.long, device=device)
    return result


def balanced_weights(labels):
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse].float())


def row_penalty(p, weights, lower):
    occupancy = p.mean(1)
    values = (F.relu(lower - occupancy) / lower).square()
    values = values + (F.relu(occupancy - (1 - lower)) / lower).square()
    return (values * weights).sum()


def mined_negative(e, q, labels, margin, hard_k, hard_mix):
    """All negative identities + top-k distinct negative identities per anchor.

    Ordinary term: average captures inside each query identity, then identities.
    Hard term: nearest capture of each identity, then nearest k identities.
    Anchors are identity balanced too. Repeated observations never occupy
    several top-k identity slots. This is an adaptation, not a paper replica.
    """
    identities, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    if len(identities) < 2:
        raise ValueError("Multi-negative training requires at least two identities per modality")
    distances = torch.cdist(e, q, p=1) / e.shape[1]
    penalty = (F.relu(margin - distances) / margin).square()
    membership = F.one_hot(inverse, num_classes=len(identities)).to(e.dtype)
    group_mean = penalty @ (membership / counts.to(e.dtype))
    valid = labels[:, None] != identities[None, :]
    ordinary_rows = (group_mean * valid).sum(1) / valid.sum(1)
    # Out-of-place scatter_reduce preserves gradients to the selected distances.
    minimum = torch.full((len(e), len(identities)), float("inf"), device=e.device, dtype=e.dtype)
    minimum = minimum.scatter_reduce(1, inverse[None, :].expand(len(e), -1), distances,
                                     reduce="amin", include_self=True)
    k = min(hard_k, len(identities) - 1)
    nearest = minimum.masked_fill(~valid, float("inf")).topk(k, dim=1, largest=False).values
    hard_rows = ((F.relu(margin - nearest) / margin).square()).mean(1)
    weights = balanced_weights(labels)
    ordinary, hard = (ordinary_rows * weights).sum(), (hard_rows * weights).sum()
    return (1 - hard_mix) * ordinary + hard_mix * hard, ordinary, hard


def compute_losses(enrollment, query, negative, labels, settings, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    values = {key: [] for key in ("positive", "negative", "negative_all", "negative_hard", "balance", "row")}
    for m in MODALITIES:
        e, q = enrollment[f"{m}_soft_bits"], query[f"{m}_soft_bits"]
        if e.ndim != 2 or e.shape != q.shape or e.shape[1] != 512 or len(e) < 2:
            raise ValueError("Expected matching (B,512) soft bits with B >= 2")
        lab = labels[m]
        if lab.dtype != torch.long or lab.shape != (len(e),) or lab.device != e.device:
            raise ValueError("Invalid modality identity labels")
        weights = balanced_weights(lab)
        margin = getattr(settings, f"{m}_margin")
        values["positive"].append((e - q).abs().mean())
        if arm in ("A", "B"):
            if negative.dtype != torch.long or negative.shape != (len(e),):
                raise ValueError("Expected one negative index per anchor")
            if bool((lab == lab[negative]).any()):
                raise ValueError("A same-identity pair cannot be a negative")
            distance = (e - q[negative]).abs().mean(1)
            neg = (F.relu(margin - distance) / margin).square().mean()
            ordinary, hard = neg, neg.new_zeros(())
        else:
            neg, ordinary, hard = mined_negative(e, q, lab, margin, settings.hard_k, settings.hard_mix)
        values["negative"].append(neg)
        values["negative_all"].append(ordinary)
        values["negative_hard"].append(hard)
        # Keep the same coordinate-balance definition as the preceding ablation.
        balance = sum(((weights[:, None] * p).sum(0) - 0.5).square().mean() for p in (e, q)) / 2
        values["balance"].append(balance)
        values["row"].append(sum(row_penalty(p, weights, settings.row_lower) for p in (e, q)) / 2)
    result = {key: torch.stack(val).mean() for key, val in values.items()}
    result["weighted_negative"] = settings.negative_weight * result["negative"]
    result["weighted_balance"] = settings.balance_weight * result["balance"]
    result["weighted_row"] = (0 if arm == "A" else settings.row_weight) * result["row"]
    result["total"] = result["positive"] + result["weighted_negative"] + result["weighted_balance"] + result["weighted_row"]
    return result


def bound_for_arm(args, arm):
    return args.narrow_tau_bound if arm == "D" else args.tau_bound


def make_model(module, state, args, arm):
    model = module.ThresholdingModel(module.ThresholdingConfig(args.temperature, bound_for_arm(args, arm)))
    model.load_state_dict(state, strict=True)
    return model.to(args.device)


def zero_tau_model(model):
    with torch.no_grad():
        model.threshold_head.weight.zero_()
        model.threshold_head.bias.zero_()
    return model.set_training_stage("inference")


def evaluation_plan(rows, users, device):
    """One fixed enrollment context per virtual user, against every query.

    Face negatives exclude same face PERSON, including other faces that share
    the virtual finger. Finger negatives exclude same (person, finger).
    """
    if len(rows) < 2:
        raise ValueError("Evaluation needs at least two pairs")
    _, anchors = np.unique(rows[:, common.USER], return_index=True)
    for index in anchors:
        group = rows[rows[:, common.USER] == rows[index, common.USER]]
        if not np.all(group[:, [common.EF, common.EP]] == rows[index, [common.EF, common.EP]]):
            raise ValueError("Evaluation requires fixed enrollment images within each virtual user")
    labels = identity_codes(rows, users, device)
    anchors = torch.as_tensor(anchors, dtype=torch.long, device=device)
    masks = {m: labels[m][anchors, None] != labels[m][None, :] for m in MODALITIES}
    if any(not bool(mask.any()) for mask in masks.values()):
        raise ValueError("No valid evaluation impostors")
    return {"anchors": anchors, "masks": masks,
            "legacy_negative": torch.as_tensor(common.negative_permutation(rows, users), dtype=torch.long, device=device)}


@torch.no_grad()
def evaluate(model, cache, rows, args, plan, on_batch=None):
    """Use hard bits and exhaustive modality-specific negatives for ALL arms.

    No training loss is used for selection. Its definition differs across arms
    and its batch statistics would not match this full-split evaluation anyway.
    """
    model.eval()
    collected = ({}, {})
    for step, offset in enumerate(range(0, len(rows), args.batch_size), 1):
        outputs = common.forward_pair(model, cache, rows[offset:offset + args.batch_size])
        for dest, output in zip(collected, outputs):
            for key in ("tau", "face_bits", "fingerprint_bits"):
                dest.setdefault(key, []).append(output[key])
        if on_batch:
            on_batch(step)
    e, q = ({key: torch.cat(val) for key, val in role.items()} for role in collected)
    if not all(bool(torch.isfinite(value).all()) for role in (e, q) for value in role.values()):
        raise RuntimeError("Nonfinite validation/test output")
    metrics = {}
    for m in MODALITIES:
        eh, qh = e[f"{m}_bits"].float(), q[f"{m}_bits"].float()
        genuine = (eh != qh).sum(1).float()
        anchor = eh[plan["anchors"]]
        # Exact integer Hamming counts: each float32 dot product is <= 512.
        matrix = anchor.sum(1)[:, None] + qh.sum(1)[None, :] - 2 * (anchor @ qh.T)
        impostor = matrix[plan["masks"][m]]
        legacy = (eh != qh[plan["legacy_negative"]]).sum(1).float()
        hard = torch.cat((eh, qh))
        row_mean, column_mean = hard.mean(1), hard.mean(0)
        g, imp = genuine.cpu().numpy(), impostor.cpu().numpy()
        metrics[m] = {
            "genuine_hamming_bits": float(genuine.mean()), "genuine_hamming_fraction": float(genuine.mean() / 512),
            "genuine_hamming_p95_bits": float(torch.quantile(genuine, 0.95)),
            "genuine_zero_error_fraction": float((genuine == 0).float().mean()),
            "impostor_hamming_bits": float(impostor.mean()),
            "impostor_hamming_p05_bits": float(torch.quantile(impostor, 0.05)),
            "impostor_pair_count": len(impostor),
            "genuine_vs_impostor_auroc": roc_auc(np.r_[np.ones(len(g), dtype=bool), np.zeros(len(imp), dtype=bool)], -np.r_[g, imp]),
            "legacy_fixed_negative_mean_bits": float(legacy.mean()),
            "ones_fraction": float(hard.mean()),
            "outside_row_band_fraction": float(((row_mean < args.row_lower) | (row_mean > 1 - args.row_lower)).float().mean()),
            "extreme_code_fraction": float(((row_mean < 0.05) | (row_mean > 0.95)).float().mean()),
            "per_code_ones_min": float(row_mean.min()), "per_code_ones_max": float(row_mean.max()),
            "constant_coordinate_fraction": float(((column_mean == 0) | (column_mean == 1)).float().mean()),
            "coordinate_balance_error": float((column_mean - 0.5).square().mean()),
        }
    tau = {name: {"mean": float(out["tau"].mean()), "std": float(out["tau"].std(unbiased=False)),
                  "min": float(out["tau"].min()), "max": float(out["tau"].max())}
           for name, out in (("enrollment", e), ("query", q))}
    tau["genuine_mean_absolute_difference"] = float((e["tau"] - q["tau"]).abs().mean())
    return {"pair_count": len(rows), "enrollment_contexts": len(plan["anchors"]), "bits": metrics, "tau": tau,
            "selection_score": sum(metrics[m]["genuine_hamming_fraction"] for m in MODALITIES) / 2}


def selection_guard(metrics, baseline, args):
    """Same fixed tau=0 validation reference and tolerances for every seed/arm."""
    reasons = []
    for m in MODALITIES:
        current, reference = metrics["bits"][m], baseline["bits"][m]
        lower_checks = (("impostor_hamming_bits", args.max_impostor_drop_bits),
                        ("impostor_hamming_p05_bits", args.max_impostor_p05_drop_bits),
                        ("genuine_vs_impostor_auroc", args.max_auroc_drop))
        upper_checks = (("genuine_hamming_bits", args.max_genuine_increase_bits),
                        ("genuine_hamming_p95_bits", args.max_genuine_p95_increase_bits),
                        ("outside_row_band_fraction", args.max_row_outside_increase))
        for key, tolerance in lower_checks:
            if current[key] < reference[key] - tolerance - 1e-9:
                reasons.append(f"{m}.{key} below tau=0 by more than {tolerance:g}")
        for key, tolerance in upper_checks:
            if current[key] > reference[key] + tolerance + 1e-9:
                reasons.append(f"{m}.{key} above tau=0 by more than {tolerance:g}")
    return not reasons, reasons


def save_checkpoint(model, path, arm, seed, epoch, origin):
    payload = {"format": model.checkpoint_format, "config": asdict(model.config),
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "purpose": "ABCD_tau_research_only; reliability_heads_untrained; not_for_enrollment",
               "experiment": {"arm": arm, "seed": seed, "epoch": epoch, "origin": origin,
                              "reliability_calibrated": False}}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train_arm(module, state, cache, data, users, validation, plan, baseline, args, settings,
              arm, seed, directory, remaining_trials, schedules):
    directory.mkdir()
    log = common.Log(directory)
    common.write_json(directory / "status.json", {"state": "running"})
    model = make_model(module, state, args, arm).set_training_stage("threshold")
    if previous.state_digest(model.state_dict()) != previous.state_digest(state):
        raise RuntimeError("Initial parameter tensors differ between conditions")
    initial = evaluate(model, cache, validation, args, plan)
    eligible, initial_reasons = selection_guard(initial, baseline, args)
    # The actual tau=0 model is an explicit candidate; a worse model need not be
    # retained merely because it was the initial network for this seed.
    best_metrics, best_epoch, origin = baseline, None, "tau_zero_fallback"
    best = baseline["selection_score"]
    best_path = directory / "best_threshold.pt"
    fallback = zero_tau_model(make_model(module, state, args, arm))
    save_checkpoint(fallback, best_path, arm, seed, None, origin)
    if eligible and initial["selection_score"] < best - args.min_delta:
        best_metrics, best_epoch, origin, best = initial, 0, "initial_network", initial["selection_score"]
        save_checkpoint(model, best_path, arm, seed, 0, origin)
    del fallback
    log.record({"epoch": 0, "validation": initial, "eligible": eligible, "ineligible_reasons": initial_reasons,
                "selected_origin": origin, "best_epoch": best_epoch})
    params = list(model.trainable_parameters())
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    optimizer = torch.optim.Adam(params, lr=args.lr)
    progress = common.Progress(f"{arm} seed={seed}", args.epochs, remaining_trials * args.epochs, args.log_interval)
    log.message(f"\n[{arm}, seed={seed}] {DESCRIPTIONS[arm]} | tau bound ±{bound_for_arm(args, arm):g}")
    stale, epoch, gradient_blocks, term_gradient_norms = 0, 0, {}, {}
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            rows = common.make_pairs(data, users, "train", args.data_seed, epoch)
            rng = np.random.default_rng(np.random.SeedSequence([args.data_seed, epoch, 9001]))
            batches, padded = common.training_batches(rows, users, args.batch_size, rng)
            digest = previous.schedule_digest(batches)
            if schedules.setdefault(epoch, digest) != digest:
                raise RuntimeError("Shared training rows/order changed across seeds or conditions")
            steps = len(batches) + math.ceil(len(validation) / args.batch_size)
            sums, processed = {}, 0
            for step, (batch, negative) in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                e, q = common.forward_pair(model, cache, batch)
                losses = compute_losses(e, q, torch.as_tensor(negative, device=args.device),
                                        identity_codes(batch, users, args.device), settings, arm)
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Nonfinite loss, arm={arm}, seed={seed}, epoch={epoch}")
                if epoch == step == 1:
                    for key in ("positive", "weighted_negative", "weighted_balance", "weighted_row"):
                        gradients = torch.autograd.grad(losses[key], tuple(model.threshold_head.parameters()),
                                                        retain_graph=True, allow_unused=True)
                        term_gradient_norms[key] = math.sqrt(sum(float(g.detach().square().sum())
                                                                for g in gradients if g is not None))
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
                if epoch == step == 1:
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and parameter.grad is None:
                            raise RuntimeError(f"Missing gradient: {name}")
                        if parameter.grad is not None:
                            block = name.split(".")[0]
                            gradient_blocks[block] = gradient_blocks.get(block, 0.0) + float(parameter.grad.norm())
                optimizer.step()
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach()) * len(batch)
                processed += len(batch)
                progress.update(epoch, step, steps, "train", sums["total"] / processed,
                                force=step == 1 or step == len(batches))
            metrics = evaluate(model, cache, validation, args, plan,
                               on_batch=lambda n: progress.update(epoch, len(batches) + n, steps, "validation", sums["total"] / processed))
            for name, parameter in model.named_parameters():
                if name in frozen and (parameter.grad is not None or not torch.equal(parameter, frozen[name])):
                    raise RuntimeError(f"Frozen reliability parameter changed: {name}")
            eligible, reasons = selection_guard(metrics, baseline, args)
            improved = eligible and metrics["selection_score"] < best - args.min_delta
            if improved:
                best_metrics, best_epoch, best, origin, stale = metrics, epoch, metrics["selection_score"], "trained_network", 0
                save_checkpoint(model, best_path, arm, seed, epoch, origin)
            else:
                stale += 1
            save_checkpoint(model, directory / "last_threshold.pt", arm, seed, epoch, "last_epoch")
            log.record({"epoch": epoch, "train_loss": {key: value / processed for key, value in sums.items()},
                        "validation": metrics, "eligible": eligible, "ineligible_reasons": reasons,
                        "best": improved, "best_epoch": best_epoch, "selected_origin": origin,
                        "schedule_sha256": digest, "original_pairs": len(rows), "processed_pairs": processed,
                        "supplementary_pairs": padded, "seconds": time.perf_counter() - epoch_started})
            progress.finish_line()
            log.message(f"[{arm} seed={seed} {epoch}/{args.epochs}] loss={sums['total']/processed:.5f} | "
                        f"적격={eligible} | best={origin}:{best_epoch} | patience={stale}/{args.patience}")
            for m in MODALITIES:
                b = metrics["bits"][m]
                log.message(f"  {m}: 동일 {b['genuine_hamming_bits']:.2f}, 상이 {b['impostor_hamming_bits']:.2f}, "
                            f"상이 P05 {b['impostor_hamming_p05_bits']:.2f} bit | AUC {b['genuine_vs_impostor_auroc']:.5f} | "
                            f"코드 쏠림 {100*b['outside_row_band_fraction']:.2f}%")
            if reasons:
                log.message("  선택 제외: " + "; ".join(reasons))
            if stale >= args.patience:
                log.message("검증에서 적격 개선이 없어 이 조건의 학습을 조기 종료합니다.")
                break
    except BaseException as error:
        progress.finish_line()
        try:
            save_checkpoint(model, directory / "interrupted.pt", arm, seed, epoch, "interrupted")
        except Exception:
            pass
        common.write_json(directory / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                      "error": str(error)})
        raise
    finally:
        progress.finish_line()
    result = {"arm": arm, "seed": seed, "epochs_run": epoch, "best_epoch": best_epoch,
              "selected_origin": origin, "best_selection_score": best,
              "initial_validation": initial, "best_validation": best_metrics,
              "initial_state_sha256": previous.state_digest(state), "tau_bound": bound_for_arm(args, arm),
              "first_step_gradient_norms": gradient_blocks, "first_step_head_gradient_norms_by_loss": term_gradient_norms,
              "seconds": time.perf_counter() - progress.start,
              "checkpoint": str(best_path), "reliability_trained": False}
    common.write_json(directory / "summary.json", result)
    common.write_json(directory / "status.json", {"state": "trained_awaiting_test"})
    if origin != "trained_network":
        log.message(f"학습 후 적격 개선 없음: {origin} 유지")
    return result


TABLE_METRICS = ("genuine_hamming_bits", "genuine_hamming_p95_bits", "impostor_hamming_bits",
                 "impostor_hamming_p05_bits", "genuine_vs_impostor_auroc", "ones_fraction",
                 "outside_row_band_fraction", "per_code_ones_min", "per_code_ones_max",
                 "constant_coordinate_fraction", "impostor_pair_count", "legacy_fixed_negative_mean_bits")


def comparison_tables(trials, baselines):
    rows, aggregate, deltas = [], [], []
    for split in ("validation", "test"):
        sources = [("tau_zero", "", None, "fixed_zero", baselines[split])]
        sources += [(t["arm"], t["seed"], t["best_epoch"], t["selected_origin"],
                     t["best_validation"] if split == "validation" else t["test"]) for t in trials]
        for arm, seed, epoch, origin, metrics in sources:
            for m in MODALITIES:
                rows.append({"split": split, "arm": arm, "seed": seed, "best_epoch": epoch,
                             "selected_origin": origin, "modality": m,
                             **{key: metrics["bits"][m][key] for key in TABLE_METRICS},
                             "tau_enrollment_mean": metrics["tau"]["enrollment"]["mean"],
                             "tau_enrollment_std": metrics["tau"]["enrollment"]["std"],
                             "tau_query_mean": metrics["tau"]["query"]["mean"],
                             "tau_genuine_absolute_difference": metrics["tau"]["genuine_mean_absolute_difference"]})
        keys = TABLE_METRICS[:7]
        for arm in ARMS:
            for m in MODALITIES:
                selected = [r for r in rows if r["split"] == split and r["arm"] == arm and r["modality"] == m]
                aggregate.append({"split": split, "arm": arm, "modality": m, "seed_count": len(selected),
                                  "trained_model_count": sum(r["selected_origin"] == "trained_network" for r in selected),
                                  **{f"{key}_{stat}": float(getattr(np, stat)([r[key] for r in selected]))
                                     for key in keys for stat in ("mean", "std")}})
        for before, after in zip(ARMS[:-1], ARMS[1:]):
            for m in MODALITIES:
                old = {r["seed"]: r for r in rows if r["split"] == split and r["arm"] == before and r["modality"] == m}
                new = {r["seed"]: r for r in rows if r["split"] == split and r["arm"] == after and r["modality"] == m}
                for seed in sorted(old):
                    deltas.append({"split": split, "change": f"{after} minus {before}", "modality": m, "seed": seed,
                                   **{f"{key}_delta": new[seed][key] - old[seed][key] for key in keys}})
    return rows, aggregate, deltas


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-run", type=Path, default=ROOT / "thresholding/results/train_20261029_162319_486882")
    parser.add_argument("--embedding-dir", type=Path, help="Relocated identical archives only")
    parser.add_argument("--output-dir", type=Path, help="New output directory; never overwrite an existing run")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--face-margin", type=float, default=0.25)
    parser.add_argument("--fingerprint-margin", type=float, default=0.25)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--row-weight", type=float, default=1.0)
    parser.add_argument("--row-lower", type=float, default=0.3, help="Permitted occupancy band [rho,1-rho]")
    parser.add_argument("--hard-mix", type=float, default=0.5)
    parser.add_argument("--hard-k", type=int, default=4, help="Nearest DISTINCT negative identities per anchor in C/D")
    parser.add_argument("--narrow-tau-bound", type=float, default=0.02, help="D uses tau=bound*tanh(logit), centered on zero")
    parser.add_argument("--max-impostor-drop-bits", type=float, default=5.0)
    parser.add_argument("--max-impostor-p05-drop-bits", type=float, default=5.0)
    parser.add_argument("--max-auroc-drop", type=float, default=0.001)
    parser.add_argument("--max-genuine-increase-bits", type=float, default=0.5)
    parser.add_argument("--max-genuine-p95-increase-bits", type=float, default=5.0)
    parser.add_argument("--max-row-outside-increase", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args(argv)
    for name in ("epochs", "patience", "threads", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >= 1")
    if args.batch_size < 2 or min(args.seeds) < 0 or max(args.seeds) >= 2**32 or len(args.seeds) != len(set(args.seeds)):
        parser.error("batch_size >= 2 and unique initialization seeds in [0,2**32) are required")
    for name in ("lr", "grad_clip", "narrow_tau_bound"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    limits = {"min_delta": 1, "max_impostor_drop_bits": 512, "max_impostor_p05_drop_bits": 512,
              "max_auroc_drop": 1, "max_genuine_increase_bits": 512,
              "max_genuine_p95_increase_bits": 512, "max_row_outside_increase": 1}
    for name, upper in limits.items():
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= upper:
            parser.error(f"{name} must be finite and in [0,{upper}]")
    try:
        loss_settings(args)
    except ValueError as error:
        parser.error(str(error))
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT / "thresholding/results" / datetime.now().strftime("tau_abcd_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT / name for name in ("muct-master", "UareU", "feature_extractors", "BCH", "scripts")]
    protected += [args.source_run] + ([args.embedding_dir] if args.embedding_dir else [])
    if any(args.output_dir.is_relative_to(p.resolve()) for p in protected):
        parser.error("Output must be separate from input datasets, archives, source code, and the source run")
    if args.output_dir.exists():
        parser.error("Output directory already exists")
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the existing environment or explicitly pass --device cpu")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print("기존 전체 임베딩·사람 단위 분할·평가 쌍의 일치를 확인합니다.", flush=True)
    data, users, pairs, source, source_hashes = prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output cannot be inside the embedding directory")
    args.temperature, args.tau_bound = float(source["temperature"]), float(source["tau_bound"])
    args.data_seed = int(source["seed"])
    if args.narrow_tau_bound >= args.tau_bound:
        raise ValueError("D's --narrow-tau-bound must be strictly smaller than the source tau bound")
    settings = loss_settings(args)
    module = common.load_model_module()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = common.Log(args.output_dir)
    try:
        common.write_json(args.output_dir / "status.json", {"state": "running"})
        for name in previous.MANIFESTS:
            shutil.copy2(args.source_run / name, args.output_dir / name)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"started_at_utc": datetime.now(timezone.utc).isoformat(), "arms": DESCRIPTIONS,
                       "loss_settings": asdict(settings), "input_sha256": data.hashes, "source_sha256": source_hashes,
                       "code_sha256": {str(path): common.sha256(path) for path in
                                       (Path(__file__), Path(previous.__file__), ROOT / "scripts/train_reliability_distance.py")},
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                       "references": REFERENCES, "variance_loss": False, "reliability_training": False,
                       "test_used_for_selection": False, "checkpoint_resuming_supported": False,
                       "D_initialization": "identical parameter tensors; only tau bound differs, so initial tau outputs/gradients differ",
                       "evaluation": "unique enrollment context per virtual user versus all queries; separate modality identity masks",
                       "selection": "min mean genuine hard-Hamming under fixed tau=0 validation guards; retain tau=0 if no eligible improvement"})
        common.write_json(args.output_dir / "config.json", config)
        cache = {m: torch.tensor(value, device=args.device) for m, value in data.arrays.items()}
        # Preparing test indices is metadata validation, not model evaluation.
        plans = {split: evaluation_plan(pairs[split], users, args.device) for split in ("validation", "test")}
        torch.manual_seed(args.seeds[0])
        reference = zero_tau_model(module.ThresholdingModel(module.ThresholdingConfig(args.temperature, args.tau_bound)).to(args.device))
        baselines = {"validation": evaluate(reference, cache, pairs["validation"], args, plans["validation"])}
        common.write_json(args.output_dir / "tau_zero_validation.json", baselines["validation"])
        log.message(f"결과: {args.output_dir}\n4조건 × {len(args.seeds)} seed = {4*len(args.seeds)}회 독립 학습")
        log.message(f"코드 허용 범위 {args.row_lower:.0%}~{1-args.row_lower:.0%} | C/D: 가까운 {args.hard_k}개 신원, "
                    f"비중 {args.hard_mix:g} | D tau bound ±{args.narrow_tau_bound:g}")
        log.message("ETA는 남은 최대 epoch 기준 추정치입니다. C/D는 A/B보다 계산량이 많으며 조기 종료 시 줄어듭니다.")
        log.message("신뢰도 학습은 포함하지 않습니다. 평가의 상이 쌍은 모달리티별 모든 유효 조합을 사용합니다.")
        partitions = json.loads((args.source_run / "splits.json").read_text())
        for split in common.SPLITS:
            log.message(f"{split}: 얼굴 {len(partitions['face_people'][split])}명 / 지문 참여자 "
                        f"{partitions['fingerprint_people'][split]} / 등록·질의 {len(pairs[split])}쌍")
        trials, schedules = [], {}
        for seed in args.seeds:
            torch.manual_seed(seed)
            initial = module.ThresholdingModel(module.ThresholdingConfig(args.temperature, args.tau_bound))
            state = {key: val.detach().clone() for key, val in initial.state_dict().items()}
            for arm in ARMS:
                trials.append(train_arm(module, state, cache, data, users, pairs["validation"], plans["validation"],
                                        baselines["validation"], args, settings, arm, seed,
                                        args.output_dir / f"{arm}_seed{seed}", 4*len(args.seeds)-len(trials)-1, schedules))
                common.write_json(args.output_dir / "training_summary.json", trials)
        log.message("\n모든 학습·검증 선택 완료. 각 선택 모델의 시험 분할을 한 번씩 평가합니다.")
        for trial in trials:
            model = module.ThresholdingModel.load_checkpoint(Path(trial["checkpoint"]), args.device)
            trial["test"] = evaluate(model, cache, pairs["test"], args, plans["test"])
            directory = Path(trial["checkpoint"]).parent
            common.write_json(directory / "report.json", trial)
            common.write_json(directory / "status.json", {"state": "completed"})
            log.message(f"시험 완료: {trial['arm']} seed={trial['seed']} | {trial['selected_origin']} epoch={trial['best_epoch']}")
        baselines["test"] = evaluate(reference, cache, pairs["test"], args, plans["test"])
        data.assert_unchanged()
        if any(common.sha256(path) != digest for path, digest in source_hashes.items()):
            raise RuntimeError("Source experiment changed during this run")
        rows, aggregate, deltas = comparison_tables(trials, baselines)
        for name, table in (("comparison.csv", rows), ("aggregate.csv", aggregate), ("paired_deltas.csv", deltas)):
            common.write_csv(args.output_dir / name, list(table[0]), table)
        report = {"trials": trials, "tau_zero": baselines, "aggregate": aggregate, "paired_deltas": deltas,
                  "seconds": time.perf_counter() - started, "input_and_source_unchanged": True,
                  "shared_schedule_sha256": schedules,
                  "limitations": ["Adapted distance/occupancy/mining losses; no paper reproduction or FE key-recovery claim.",
                                  "Actual fingerprint participants: 7. Validation/test each contain one person; impostors there are different fingers.",
                                  "Synthetic cross-modal correspondence. Initialization seeds are not independent participant folds.",
                                  "Previously inspected test split is exploratory, not a new untouched benchmark.",
                                  "All-modality-specific impostor evaluation differs from the previous single-permutation metrics.",
                                  "Common validation guards and hyperparameters are experimental starting choices, not guarantees.",
                                  "A/B and C/D intentionally differ in negative sampling and identity weighting; D also changes tau scaling.",
                                  "Reliability heads are frozen at random initialization and must not be used as calibrated probabilities."]}
        common.write_json(args.output_dir / "report.json", report)
        common.write_json(args.output_dir / "status.json", {"state": "completed", "seconds": report["seconds"]})
        log.message(f"완료 | 총 {common.duration(report['seconds'])} | {args.output_dir / 'comparison.csv'}")
        for item in aggregate:
            if item["split"] == "test":
                log.message(f"{item['arm']} / {item['modality']}: 동일 {item['genuine_hamming_bits_mean']:.2f} ± "
                            f"{item['genuine_hamming_bits_std']:.2f}, 상이 P05 {item['impostor_hamming_p05_bits_mean']:.2f} bit, "
                            f"AUC {item['genuine_vs_impostor_auroc_mean']:.5f} | 학습 모델 선택 {item['trained_model_count']}/{item['seed_count']}")
    except BaseException as error:
        common.write_json(args.output_dir / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        (args.output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
