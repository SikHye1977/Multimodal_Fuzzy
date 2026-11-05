#!/usr/bin/env python3
"""A: tau=0 + distance calibration; B: tau=0 + variability reliability;
C: independent bounded scalar tau + the same variability reliability as B.

Reuses the original MUCT/UareU person splits and frozen 512-D archives. No
feature extraction, image preprocessing, Fusion training, or FE key evaluation.
--check-only validates inputs without training or creating an output directory.
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
from scripts import train_thresholding as common
from scripts import train_thresholding_abcd as contrastive
from scripts import train_thresholding_architectures as architectures
from scripts.train_reliability_distance import prepare_source, roc_auc
from thresholding.variability_model import ModelConfig, VariabilityThresholdModel, MODALITIES

ARMS = ("A", "B", "C")
DESCRIPTIONS = {"A": "fixed tau=0 + two-parameter distance calibration per modality",
                "B": "fixed tau=0 + variability-aware per-bit reliability",
                "C": "independent bounded scalar tau + B reliability, trained sequentially"}
MANIFESTS = ("splits.json", "virtual_users.csv", "samples.csv", "train_pairs.csv",
             "validation_pairs.csv", "test_pairs.csv")


def make_model(args, arm, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = VariabilityThresholdModel(ModelConfig(
            arm, args.temperature, args.tau_base_bound, args.tau_delta_bound))
    return model.to(args.device)


def make_schedule(data, users, args, seed, epoch, stage, schedules):
    rows = common.make_pairs(data, users, "train", args.data_seed, epoch=epoch)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1 if stage == "threshold" else 2]))
    batches, padding = common.training_batches(rows, users, args.batch_size, rng)
    digest = hashlib.sha256()
    for batch, _ in batches:
        digest.update(np.asarray(batch, dtype="<i8").tobytes())
    key = f"{stage}:seed{seed}:epoch{epoch}"
    value = digest.hexdigest()
    if key in schedules and schedules[key] != value:
        raise RuntimeError("A/B/C training schedules differ for the same seed/epoch")
    schedules[key] = value
    return batches, padding


@torch.no_grad()
def fit_statistics(model, cache, data, users, rows, args):
    """Only TRAIN captures. Variance: captures -> identity -> person -> population.

Fingerprint identities are (person,finger); variance is averaged over a person's
fingers before averaging people. Thus repeated virtual mappings add no variance
observations. Distance/prior use the common epoch-1 pair sampling distribution.
"""
    if any(users[int(i)].split != "train" for i in rows[:, common.USER]):
        raise ValueError("Statistics may only use training participants")
    result = {}
    model.eval()
    for m in MODALITIES:
        identities = sorted({u.face_person if m == "face" else u.finger for u in users if u.split == "train"})
        by_person = defaultdict(list)
        capture_count = 0
        for identity in identities:
            indices = data.groups[m][identity]
            if len(indices) < 2:
                raise ValueError("Repeated real captures are required to fit variability")
            x = cache[m][torch.as_tensor(indices, device=args.device)]
            margin = model.quantize(m, x)["margin"].double()
            person = identity if m == "face" else identity[0]
            by_person[person].append(margin.var(dim=0, unbiased=True))
            capture_count += len(indices)
        variance = torch.stack([torch.stack(v).mean(0) for v in by_person.values()]).mean(0)
        shrunk = (1 - args.variance_shrinkage) * variance + args.variance_shrinkage * variance.mean()
        scale = shrunk.clamp_min(args.variance_floor ** 2).sqrt()
        distance_sum, agreements, count = 0., 0, 0
        ec, qc = (common.EF, common.QF) if m == "face" else (common.EP, common.QP)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            e, q = [model.quantize(m, cache[m][torch.as_tensor(batch[:, c], device=args.device)]) for c in (ec, qc)]
            distance_sum += float(e["distance"].double().sum() + q["distance"].double().sum())
            agreements += 2 * int((e["bits"] == q["bits"]).sum())
            count += 2 * e["bits"].numel()
        empirical = agreements / count
        result[m] = {"mean_distance": max(distance_sum / count, 1e-8),
                     "agreement_rate": float(np.clip(empirical, 1e-6, 1 - 1e-6)),
                     "empirical_agreement_rate": empirical,
                     "variability_scale": scale.cpu().tolist(),
                     "variance_identity_count": len(identities), "variance_person_count": len(by_person),
                     "unique_capture_count": capture_count, "source_split": "train",
                     "shrinkage": args.variance_shrinkage, "scale_floor": args.variance_floor,
                     "definition": "sqrt(shrunk mean within-identity margin variance); equal weight per real person"}
    return result


def tau_losses(model, e, q, rows, users, args):
    labels = contrastive.identity_codes(rows, users, args.device)
    values = architectures.compute_losses(e, q, labels, contrastive.loss_settings(args))
    bound = model.config.tau_bound
    stability = torch.stack([((e[f"{m}_tau"] - q[f"{m}_tau"]) / bound).square().mean()
                             for m in MODALITIES]).mean()
    residual = torch.stack([
        ((o[f"{m}_tau"] - model.config.tau_base_bound * model.thresholds[m].base_logit.tanh())
         / model.config.tau_delta_bound).square().mean()
        for m in MODALITIES for o in (e, q)]).mean()
    return {"total": values["total"] + args.tau_stability_weight * stability + args.tau_residual_weight * residual,
            **{k: values[k] for k in ("positive", "negative", "balance", "row")},
            "tau_stability": stability, "tau_residual": residual}


def reliability_loss(e, q):
    terms = []
    for m in MODALITIES:
        target = (e[f"{m}_bits"] == q[f"{m}_bits"]).float().detach()
        terms.extend(F.binary_cross_entropy_with_logits(o[f"{m}_reliability_logits"], target) for o in (e, q))
    # No class weighting, forced 50:50 flip labels, or impostor reliability targets.
    return torch.stack(terms).mean()


@torch.no_grad()
def collect(model, cache, rows, batch_size, reliability=False, on_batch=None):
    model.eval()
    outputs = ({}, {})
    keys = [f"{m}_{k}" for m in MODALITIES for k in
            (("tau", "bits", "distance", "reliability_logits") if reliability else ("tau", "bits"))]
    for step, start in enumerate(range(0, len(rows), batch_size), 1):
        batch = rows[start:start + batch_size]
        if reliability:
            pair = common.forward_pair(model, cache, batch)
        else:
            inputs = common.model_inputs(cache, batch)
            pair = []
            for face, finger in ((inputs[0], inputs[1]), (inputs[2], inputs[3])):
                out = {}
                for m, x in zip(MODALITIES, (face, finger)):
                    out.update({f"{m}_{k}": v for k, v in model.quantize(m, x).items()})
                pair.append(out)
        for dest, output in zip(outputs, pair):
            for key in keys:
                dest.setdefault(key, []).append(output[key].cpu())
        if on_batch:
            on_batch(step)
    return tuple({key: torch.cat(items) for key, items in out.items()} for out in outputs)


def bit_metrics(e, q, plan, args):
    result, tau, radius = {}, {}, []
    for m in MODALITIES:
        eh, qh = e[f"{m}_bits"].float(), q[f"{m}_bits"].float()
        genuine = (eh != qh).sum(1).float()
        anchors = eh[plan["anchors"].cpu()]
        matrix = anchors.sum(1)[:, None] + qh.sum(1)[None] - 2 * (anchors @ qh.T)
        impostor = matrix[plan["masks"][m].cpu()]
        hard = torch.cat((eh, qh))
        occupancy, coordinate = hard.mean(1), hard.mean(0)
        within = genuine <= args.bch_t
        radius.append(within)
        result[m] = {
            "genuine_hamming_bits": float(genuine.mean()), "genuine_hamming_fraction": float(genuine.mean() / 512),
            "genuine_hamming_p95_bits": float(torch.quantile(genuine, .95)),
            "genuine_zero_error_fraction": float((genuine == 0).float().mean()),
            "within_radius_fraction": float(within.float().mean()),
            "impostor_hamming_bits": float(impostor.mean()),
            "impostor_hamming_p05_bits": float(torch.quantile(impostor, .05)),
            "impostor_pair_count": len(impostor),
            "genuine_vs_impostor_auroc": roc_auc(np.r_[np.ones(len(genuine), bool), np.zeros(len(impostor), bool)],
                                                -np.r_[genuine.numpy(), impostor.numpy()]),
            "ones_fraction": float(hard.mean()),
            "outside_row_band_fraction": float(((occupancy < args.row_lower) | (occupancy > 1-args.row_lower)).float().mean()),
            "constant_coordinate_fraction": float(((coordinate == 0) | (coordinate == 1)).float().mean()),
            "coordinate_balance_error": float((coordinate - .5).square().mean())}
        tau[m] = {role: {"mean": float(out[f"{m}_tau"].mean()),
                          "std": float(out[f"{m}_tau"].std(unbiased=False))}
                  for role, out in (("enrollment", e), ("query", q))}
        tau[m]["genuine_mean_absolute_difference"] = float((e[f"{m}_tau"] - q[f"{m}_tau"]).abs().mean())
    return {"pair_count": len(e["face_bits"]), "bits": result, "tau": tau,
            "both_within_radius_fraction": float((radius[0] & radius[1]).float().mean()),
            "mean_within_radius_fraction": sum(result[m]["within_radius_fraction"] for m in MODALITIES)/2,
            "mean_genuine_hamming_bits": sum(result[m]["genuine_hamming_bits"] for m in MODALITIES)/2}


def threshold_better(candidate, best, args):
    for key in ("both_within_radius_fraction", "mean_within_radius_fraction"):
        if candidate[key] > best[key] + 1e-9:
            return True
        if candidate[key] < best[key] - 1e-9:
            return False
    return candidate["mean_genuine_hamming_bits"] < best["mean_genuine_hamming_bits"] - args.min_bit_gain


def average_precision(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores)
    total = int(labels.sum())
    if total == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]) + 1, len(s)]
    tp = np.cumsum(y)[ends - 1]
    return float(np.sum(np.diff(np.r_[0, tp]) / total * tp / ends))


def low_confidence_precision(labels, scores, fraction):
    """Expected error precision in the lowest-r fraction; fractional ties."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores)
    k = max(1, int(math.ceil(fraction * len(scores))))
    boundary = np.partition(scores, k - 1)[k - 1]
    below, tied = scores < boundary, scores == boundary
    return float((labels[below].sum() + (k - below.sum()) * labels[tied].mean()) / k)


def reliability_metrics(labels, logits, constant, distances=None, detailed=True):
    labels = np.asarray(labels, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64).ravel()
    probability = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    if not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite reliability predictions")
    values = {"bce": float((np.logaddexp(0, logits) - labels * logits).mean()),
              "brier": float(np.square(probability - labels).mean()),
              "agreement_rate": float(labels.mean()), "probability_mean": float(probability.mean()),
              "constant_bce": float(-(labels * math.log(constant) + (1-labels)*math.log1p(-constant)).mean())}
    if detailed:
        bins, ece = [], 0.
        bin_index = np.minimum((probability * 10).astype(int), 9)
        for i in range(10):
            selected = bin_index == i
            n = int(selected.sum())
            p, y = (float(probability[selected].mean()), float(labels[selected].mean())) if n else (None, None)
            bins.append({"lower": i/10, "upper": (i+1)/10, "count": n,
                         "predicted_agreement": p, "observed_agreement": y})
            if n:
                ece += n / len(labels) * abs(p-y)
        errors = labels == 0
        values.update({"agreement_auroc": roc_auc(labels.astype(bool), logits),
                       "error_average_precision": average_precision(errors, -logits),
                       "error_prevalence": float(errors.mean()), "ece_10": ece,
                       "probability_min": float(probability.min()), "probability_max": float(probability.max()),
                       "below_half_fraction": float((probability < .5).mean()),
                       "lowest10_error_precision": low_confidence_precision(errors, probability, .1),
                       "lowest20_error_precision": low_confidence_precision(errors, probability, .2),
                       "calibration_bins": bins})
        if distances is not None:
            values["raw_distance_auroc"] = roc_auc(labels.astype(bool), np.asarray(distances).ravel())
    return values


def evaluate_reliability(model, e, q, detailed=True):
    result = {}
    for m in MODALITIES:
        y = (e[f"{m}_bits"] == q[f"{m}_bits"]).numpy().ravel()
        constant = float(model.reliability[m].constant_probability)
        result[m] = {}
        for role, output in (("enrollment", e), ("query", q)):
            result[m][role] = reliability_metrics(y, output[f"{m}_reliability_logits"].numpy(), constant,
                                                   output[f"{m}_distance"].numpy(), detailed)
        result[m]["pooled"] = {k: (result[m]["enrollment"][k] + result[m]["query"][k])/2
                               for k in ("bce", "brier", "agreement_rate", "probability_mean", "constant_bce")}
    return {"modalities": result, "selection_bce": sum(result[m]["pooled"]["bce"] for m in MODALITIES)/2}


def scalar_oracle(x, target, bound=None):
    """Exact best scalar for a FIXED enrollment target, including tied coordinates.

This uses target bits and is never an inference model. Candidate thresholds use
the actual float32 >= convention, not midpoints that could round to a tie.
"""
    x = np.asarray(x, np.float32)
    target = np.asarray(target, bool)
    if x.ndim != 1 or target.shape != x.shape or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Invalid oracle inputs")
    order = np.argsort(x, kind="stable")
    sorted_x, y = x[order], target[order]
    changes = np.where(y, 1, -1)
    errors_after = int((~target).sum()) + np.cumsum(changes)
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    candidates = sorted_x[starts]
    counts = np.r_[int((~target).sum()), errors_after[starts[1:] - 1]]
    beyond = np.nextafter(sorted_x[-1], np.float32(np.inf))
    candidates = np.r_[candidates, beyond]
    counts = np.r_[counts, errors_after[-1]]
    if bound is not None:
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("Oracle bound must be positive and finite")
        inside = (candidates >= -bound) & (candidates <= bound)
        candidates, counts = candidates[inside], counts[inside]
        extra = np.asarray([-bound, bound], np.float32)
        candidates = np.r_[candidates, extra]
        counts = np.r_[counts, ((x[None] >= extra[:, None]) != target).sum(1)]
    best = int(np.lexsort((np.abs(candidates), counts))[0])
    return int(counts[best]), float(candidates[best])


def oracle_report(model, cache, rows, args, output_file):
    e, q = collect(model, cache, rows, args.batch_size)
    records, summary = [], {}
    values = {m: defaultdict(list) for m in MODALITIES}
    for i, row in enumerate(rows):
        item = {"pair_index": i, "virtual_user_index": int(row[common.USER])}
        for m, qc in (("face", common.QF), ("fingerprint", common.QP)):
            x = cache[m][int(row[qc])].detach().cpu().numpy()
            target = e[f"{m}_bits"][i].numpy()
            current = int((e[f"{m}_bits"][i] != q[f"{m}_bits"][i]).sum())
            bounded, bounded_tau = scalar_oracle(x, target, model.config.tau_bound)
            unlimited, unlimited_tau = scalar_oracle(x, target)
            if unlimited > bounded or bounded > current:
                raise RuntimeError("Oracle must not be worse than the feasible current scalar")
            for key, value in (("current_errors", current), ("bounded_oracle_errors", bounded),
                               ("unbounded_oracle_errors", unlimited), ("bounded_oracle_tau", bounded_tau),
                               ("unbounded_oracle_tau", unlimited_tau)):
                item[f"{m}_{key}"] = value
                values[m][key].append(value)
        records.append(item)
    for m in MODALITIES:
        summary[m] = {key: {"mean_bits": float(np.mean(values[m][key])),
                            "p95_bits": float(np.quantile(values[m][key], .95)),
                            "within_radius_fraction": float(np.mean(np.asarray(values[m][key]) <= args.bch_t))}
                      for key in ("current_errors", "bounded_oracle_errors", "unbounded_oracle_errors")}
    common.write_csv(output_file, list(records[0]), records)
    return {"diagnostic_only": True, "uses_enrollment_ground_truth": True,
            "not_used_for_training_or_checkpoint_selection": True, "query_thresholds_independent_per_modality": True,
            "enrollment_bits_held_fixed": True, "bounded_tau_range": [-model.config.tau_bound, model.config.tau_bound],
            "modalities": summary, "pairs_csv": str(output_file)}


def assert_fixed(model, state):
    current = model.state_dict()
    if any(not torch.equal(current[k].detach().cpu(), v) for k, v in state.items()):
        raise RuntimeError("Reliability training changed a threshold or training-only statistic")
    if any(p.grad is not None for p in model.thresholds.parameters()):
        raise RuntimeError("Reliability loss reached the frozen thresholds")


def train_threshold(model, data, users, cache, rows, plan, baseline, args, seed, directory, log, schedules, future):
    model.set_stage("threshold")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.threshold_lr)
    best, best_epoch, stale = baseline, 0, 0
    metadata = {"arm": "C", "seed": seed, "stage": "threshold", "epoch": 0, "origin": "fixed_zero_fallback"}
    model.save_checkpoint(directory / "best_threshold.pt", metadata)
    progress = common.Progress(f"C seed={seed} tau", args.threshold_epochs, future, args.log_interval)
    epochs_run = 0
    for epoch in range(1, args.threshold_epochs + 1):
        epochs_run = epoch
        model.train()
        batches, padding = make_schedule(data, users, args, seed, epoch, "threshold", schedules)
        total, n = 0., 0
        steps = len(batches) + math.ceil(len(rows) / args.batch_size)
        components = defaultdict(float)
        for step, (batch, _) in enumerate(batches, 1):
            e, q = common.forward_pair(model, cache, batch)
            loss = tau_losses(model, e, q, batch, users, args)
            optimizer.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip,
                                          error_if_nonfinite=True)
            optimizer.step()
            total += float(loss["total"].detach()) * len(batch)
            n += len(batch)
            for key, value in loss.items():
                components[key] += float(value.detach()) * len(batch)
            progress.update(epoch, step, steps, "train", total/n)
        e, q = collect(model, cache, rows, args.batch_size, on_batch=lambda k:
                       progress.update(epoch, len(batches)+k, steps, "validation", total/n))
        metrics = bit_metrics(e, q, plan, args)
        eligible, reasons = contrastive.selection_guard(metrics, baseline, args)
        improved = eligible and threshold_better(metrics, best, args)
        if improved:
            best, best_epoch, stale = metrics, epoch, 0
            model.save_checkpoint(directory / "best_threshold.pt", {**metadata, "epoch": epoch, "origin": "trained_network"})
        else:
            stale += 1
        model.save_checkpoint(directory / "last_threshold.pt", {**metadata, "epoch": epoch, "origin": "last_epoch"})
        progress.finish_line()
        log.record({"stage": "threshold", "epoch": epoch, "train_loss": {k: v/n for k, v in components.items()},
                    "padding": padding, "validation": metrics, "eligible": eligible, "reasons": reasons, "selected": improved})
        log.message(f"C seed={seed} tau epoch {epoch} | val 평균 오류 {metrics['mean_genuine_hamming_bits']:.3f} bit | "
                    f"양쪽 t 이내 {metrics['both_within_radius_fraction']:.3%} | best {best_epoch}")
        if stale >= args.patience:
            break
    selected, meta = VariabilityThresholdModel.load_checkpoint(directory / "best_threshold.pt", args.device)
    return selected, {"epochs_run": epochs_run, "best_epoch": best_epoch, "origin": meta["origin"], "validation": best}


def train_reliability(model, data, users, cache, rows, args, seed, directory, log, schedules, future):
    model.set_stage("reliability")
    fixed = model.fixed_reliability_state()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.reliability_lr,
                                 weight_decay=args.reliability_weight_decay)
    e, q = collect(model, cache, rows, args.batch_size, reliability=True)
    initial = evaluate_reliability(model, e, q, detailed=False)
    best, best_epoch, stale = initial, 0, 0
    metadata = {"arm": model.config.arm, "seed": seed, "stage": "reliability", "epoch": 0, "origin": "initial_prior"}
    model.save_checkpoint(directory / "best_model.pt", metadata)
    log.record({"stage": "reliability", "epoch": 0, "validation": initial, "selected": True})
    progress = common.Progress(f"{model.config.arm} seed={seed} reliability", args.reliability_epochs, future, args.log_interval)
    epochs_run = 0
    for epoch in range(1, args.reliability_epochs + 1):
        epochs_run = epoch
        model.train()
        batches, padding = make_schedule(data, users, args, seed, epoch, "reliability", schedules)
        total, n = 0., 0
        steps = len(batches) + math.ceil(len(rows) / args.batch_size)
        for step, (batch, _) in enumerate(batches, 1):
            e, q = common.forward_pair(model, cache, batch)
            loss = reliability_loss(e, q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip,
                                          error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            n += len(batch)
            progress.update(epoch, step, steps, "train", total/n)
        assert_fixed(model, fixed)
        e, q = collect(model, cache, rows, args.batch_size, reliability=True, on_batch=lambda k:
                       progress.update(epoch, len(batches)+k, steps, "validation", total/n))
        metrics = evaluate_reliability(model, e, q, detailed=False)
        improved = metrics["selection_bce"] < best["selection_bce"] - args.min_delta
        if improved:
            best, best_epoch, stale = metrics, epoch, 0
            model.save_checkpoint(directory / "best_model.pt", {**metadata, "epoch": epoch, "origin": "trained_network"})
        else:
            stale += 1
        model.save_checkpoint(directory / "last_model.pt", {**metadata, "epoch": epoch, "origin": "last_epoch"})
        progress.finish_line()
        log.record({"stage": "reliability", "epoch": epoch, "train_bce": total/n, "padding": padding,
                    "validation": metrics, "selected": improved, "threshold_and_statistics_unchanged": True})
        log.message(f"{model.config.arm} seed={seed} reliability epoch {epoch} | train BCE {total/n:.6f} | "
                    f"val BCE {metrics['selection_bce']:.6f} | best {best_epoch}")
        if stale >= args.patience:
            break
    selected, meta = VariabilityThresholdModel.load_checkpoint(directory / "best_model.pt", args.device)
    assert_fixed(selected, fixed)
    return selected, {"epochs_run": epochs_run, "best_epoch": best_epoch, "origin": meta["origin"],
                      "initial_validation": initial, "validation": best}


def evaluate_trial(model, cache, rows, plan, args, directory, split, data):
    e, q = collect(model, cache, rows, args.batch_size, reliability=True)
    result = {"bits": bit_metrics(e, q, plan, args), "reliability": evaluate_reliability(model, e, q)}
    records = []
    for i, row in enumerate(rows):
        record = {"pair_index": i, "virtual_user_index": int(row[common.USER])}
        for m, ec, qc in (("face", common.EF, common.QF), ("fingerprint", common.EP, common.QP)):
            record[f"{m}_enrollment_file"] = data.files[m][int(row[ec])]
            record[f"{m}_query_file"] = data.files[m][int(row[qc])]
            record[f"{m}_hamming_bits"] = int((e[f"{m}_bits"][i] != q[f"{m}_bits"][i]).sum())
            for role, out in (("enrollment", e), ("query", q)):
                record[f"{m}_{role}_tau"] = float(out[f"{m}_tau"][i, 0])
                record[f"{m}_{role}_reliability_mean"] = float(out[f"{m}_reliability_logits"][i].sigmoid().mean())
        records.append(record)
    common.write_csv(directory / f"{split}_pairs_metrics.csv", list(records[0]), records)
    np.savez_compressed(directory / f"{split}_outputs.npz", **{
        f"{role}_{key}": value.numpy() for role, out in (("enrollment", e), ("query", q)) for key, value in out.items()})
    return result


def tables(trials):
    rows = []
    for trial in trials:
        for split in ("validation", "test"):
            for m in MODALITIES:
                entry = trial[split]
                r = entry["reliability"]["modalities"][m]["query"]
                row = {"arm": trial["arm"], "seed": trial["seed"], "split": split, "modality": m,
                       "threshold_best_epoch": trial["threshold"]["best_epoch"],
                       "reliability_best_epoch": trial["reliability_training"]["best_epoch"],
                       **entry["bits"]["bits"][m]}
                row.update({f"query_{k}": v for k, v in r.items() if k != "calibration_bins"})
                row["tau_pair_mean_absolute_difference"] = entry["bits"]["tau"][m]["genuine_mean_absolute_difference"]
                rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"], row["split"], row["modality"]].append(row)
    metadata = {"arm", "seed", "split", "modality", "threshold_best_epoch", "reliability_best_epoch"}
    aggregates = []
    for (arm, split, m), items in groups.items():
        item = {"arm": arm, "split": split, "modality": m, "seed_count": len(items)}
        for key in rows[0]:
            if key in metadata:
                continue
            values = [r[key] for r in items if r[key] is not None]
            item[f"{key}_mean"] = float(np.mean(values)) if values else None
            item[f"{key}_std"] = float(np.std(values)) if values else None
        aggregates.append(item)
    indexed = {(r["arm"], r["seed"], r["split"], r["modality"]): r for r in rows}
    deltas = []
    for arm, reference in (("B", "A"), ("C", "B")):
        for row in rows:
            if row["arm"] != arm:
                continue
            base = indexed.get((reference, row["seed"], row["split"], row["modality"]))
            if base is None:
                continue
            item = {k: row[k] for k in ("seed", "split", "modality")}
            item["comparison"] = f"{arm}-{reference}"
            for key in ("genuine_hamming_bits", "within_radius_fraction", "query_bce", "query_brier",
                        "query_ece_10", "query_agreement_auroc", "query_error_average_precision"):
                item[f"{key}_delta"] = row[key] - base[key] if row[key] is not None and base[key] is not None else None
            deltas.append(item)
    return rows, aggregates, deltas


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-run", type=Path, default=ROOT / "thresholding/results/train_20261029_162319_486882")
    p.add_argument("--embedding-dir", type=Path, help="Relocated identical archives only; new datasets require a new data protocol")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    p.add_argument("--threshold-epochs", type=int, default=100)
    p.add_argument("--reliability-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold-lr", type=float, default=1e-4)
    p.add_argument("--reliability-lr", type=float, default=1e-3)
    p.add_argument("--reliability-weight-decay", type=float, default=0.)
    p.add_argument("--tau-base-bound", type=float, default=.01)
    p.add_argument("--tau-delta-bound", type=float, default=.01)
    p.add_argument("--tau-stability-weight", type=float, default=.01)
    p.add_argument("--tau-residual-weight", type=float, default=.001)
    p.add_argument("--variance-shrinkage", type=float, default=.25)
    p.add_argument("--variance-floor", type=float, default=1e-4)
    p.add_argument("--min-delta", type=float, default=1e-5, help="Validation BCE improvement for reliability selection")
    p.add_argument("--min-bit-gain", type=float, default=.1, help="Mean hard-bit improvement if radius rates tie")
    p.add_argument("--bch-t", type=int, default=8, help="Radius statistic only; no BCH or key recovery is run")
    for key, value in asdict(contrastive.LossSettings()).items():
        p.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    for key, value in {"max_impostor_drop_bits": 5., "max_impostor_p05_drop_bits": 5., "max_auroc_drop": .001,
                       "max_genuine_increase_bits": .5, "max_genuine_p95_increase_bits": 5.,
                       "max_row_outside_increase": .01}.items():
        p.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    p.add_argument("--grad-clip", type=float, default=5.)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    for key in ("threshold_epochs", "reliability_epochs", "patience", "threads", "log_interval"):
        if getattr(args, key) < 1:
            p.error(f"{key} must be >= 1")
    if args.batch_size < 2 or not 1 <= args.bch_t <= 255:
        p.error("batch-size >= 2 and 1 <= bch-t <= 255 are required")
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds) or not all(0 <= s < 2**32 for s in args.seeds):
        p.error("Unique arms and seeds in [0,2**32) are required")
    for key in ("threshold_lr", "reliability_lr", "tau_base_bound", "tau_delta_bound", "variance_floor", "grad_clip"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} must be positive and finite")
    for key in ("tau_stability_weight", "tau_residual_weight", "min_delta", "min_bit_gain", "reliability_weight_decay"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(f"{key} must be nonnegative and finite")
    if not math.isfinite(args.variance_shrinkage) or not 0 <= args.variance_shrinkage <= 1:
        p.error("variance-shrinkage must be in [0,1]")
    for key, value in vars(args).items():
        if key.startswith("max_") and (not math.isfinite(value) or value < 0):
            p.error(f"{key} must be nonnegative and finite")
    try:
        contrastive.loss_settings(args)
    except ValueError as error:
        p.error(str(error))
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT / "thresholding/results" / datetime.now().strftime("reliability_abc_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT / k for k in ("scripts", "muct-master", "UareU", "BCH", "feature_extractors")]
    protected.append(args.source_run)
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    if args.output_dir.exists() or any(args.output_dir.is_relative_to(x.resolve()) for x in protected):
        p.error("Output must be a new directory outside source data, source code and existing experiments")
    return args


def run_experiments(args, data, users, pairs, source, source_hashes, *, synthetic=False):
    started = time.perf_counter()
    args.temperature, args.data_seed = float(source["temperature"]), int(source["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = common.Log(args.output_dir)
    trials, schedules = [], {}
    try:
        common.write_json(args.output_dir / "status.json", {"state": "running", "synthetic_test_only": synthetic})
        if not synthetic:
            for name in MANIFESTS:
                shutil.copy2(args.source_run / name, args.output_dir / name)
        code_paths = [Path(__file__), ROOT / "thresholding/variability_model.py", Path(common.__file__),
                      Path(contrastive.__file__), Path(architectures.__file__),
                      ROOT / "scripts/train_reliability_distance.py", ROOT / "scripts/train_thresholding_ablation.py",
                      ROOT / "Thresholding Model.py"]
        hashes = {str(path): common.sha256(path) for path in code_paths}
        common.write_json(args.output_dir / "config.json", {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "descriptions": DESCRIPTIONS, "synthetic_test_only": synthetic,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "input_sha256": data.hashes, "source_sha256": source_hashes, "code_sha256": hashes,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_used_for_selection": False, "checkpoint_resuming_supported": False,
            "statistic_definition": fit_statistics.__doc__,
            "selection": "C tau: guarded hard radius rates then mean Hamming vs tau=0; r: pooled genuine BCE on validation",
            "loss_note": "Existing D contrastive/balance/row loss + normalized tau consistency and residual penalties; unweighted reliability BCE",
            "FE_key_evaluation": False})
        cache = {m: torch.tensor(v, device=args.device) for m, v in data.arrays.items()}
        plans = {s: contrastive.evaluation_plan(pairs[s], users, "cpu") for s in ("validation", "test")}
        reference = make_model(args, "A", args.seeds[0])
        e, q = collect(reference, cache, pairs["validation"], args.batch_size)
        baseline = bit_metrics(e, q, plans["validation"], args)
        common.write_json(args.output_dir / "tau_zero_validation.json", baseline)
        base_stats = fit_statistics(reference, cache, data, users, pairs["train"], args)
        common.write_json(args.output_dir / "tau_zero_train_statistics.json", base_stats)
        oracles = {"fixed_zero_validation": oracle_report(reference, cache, pairs["validation"], args,
                                                          args.output_dir / "oracle_zero_validation.csv")}
        common.write_json(args.output_dir / "oracle_diagnostics.json", oracles)
        jobs = [(seed, arm) for seed in args.seeds for arm in args.arms]
        log.message(f"결과: {args.output_dir}\n{len(args.arms)}조건 × {len(args.seeds)} seed = {len(jobs)}회 순차 학습")
        log.message("A/B: 동일 tau=0, C: tau 학습 후 고정. 확률 보정과 hard 비트 오류를 별도 평가합니다.")
        log.message("ETA는 현재 단계 속도와 남은 최대 epoch 기준 추정이며, 조기 종료·모델별 속도 차이·최종 평가는 별도입니다.")
        log.message(f"쌍 수: train={len(pairs['train'])}, validation={len(pairs['validation'])}, test={len(pairs['test'])}")
        for j, (seed, arm) in enumerate(jobs):
            directory = args.output_dir / f"{arm}_seed{seed}"
            directory.mkdir()
            trial_log = common.Log(directory)
            common.write_json(directory / "status.json", {"state": "running"})
            future = sum(args.reliability_epochs + (args.threshold_epochs if a == "C" else 0) for _, a in jobs[j+1:])
            torch.manual_seed(seed)
            if args.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = make_model(args, arm, seed)
            threshold = {"best_epoch": 0, "epochs_run": 0, "origin": "fixed_zero", "validation": baseline}
            if arm == "C":
                model, threshold = train_threshold(model, data, users, cache, pairs["validation"], plans["validation"], baseline,
                                                   args, seed, directory, trial_log, schedules, future + args.reliability_epochs)
                stats = fit_statistics(model, cache, data, users, pairs["train"], args)
            else:
                stats = base_stats
            for m in MODALITIES:
                model.reliability[m].set_statistics(stats[m])
            common.write_json(directory / "train_statistics.json", stats)
            common.write_json(directory / "threshold_summary.json", threshold)
            model, reliability = train_reliability(model, data, users, cache, pairs["validation"], args, seed,
                                                    directory, trial_log, schedules, future)
            validation = evaluate_trial(model, cache, pairs["validation"], plans["validation"], args, directory, "validation", data)
            if arm in ("A", "B") and validation["bits"] != baseline:
                raise RuntimeError("A/B must preserve exactly the fixed-zero bit baseline")
            trial = {"arm": arm, "seed": seed, "checkpoint": str(directory / "best_model.pt"),
                     "threshold": threshold, "reliability_training": reliability, "validation": validation,
                     "model_parameters": sum(p.numel() for p in model.parameters())}
            if arm == "C":
                oracles[f"C_seed{seed}_validation"] = oracle_report(model, cache, pairs["validation"], args,
                                                                    directory / "oracle_validation.csv")
            trials.append(trial)
            common.write_json(directory / "summary.json", trial)
            common.write_json(directory / "status.json", {"state": "trained_awaiting_test"})
            common.write_json(args.output_dir / "training_summary.json", trials)
            log.message(f"학습 완료 {j+1}/{len(jobs)}: {arm} seed={seed} | tau best={threshold['best_epoch']}, "
                        f"reliability best={reliability['best_epoch']}")
        log.message("모든 모델 선택 완료. 이제 고정한 체크포인트로 시험 분할을 평가합니다.")
        for trial in trials:
            model, _ = VariabilityThresholdModel.load_checkpoint(trial["checkpoint"], args.device)
            directory = Path(trial["checkpoint"]).parent
            trial["test"] = evaluate_trial(model, cache, pairs["test"], plans["test"], args, directory, "test", data)
            if trial["arm"] == "C":
                oracles[f"C_seed{trial['seed']}_test"] = oracle_report(model, cache, pairs["test"], args, directory / "oracle_test.csv")
            common.write_json(directory / "report.json", trial)
            common.write_json(directory / "status.json", {"state": "completed"})
            log.message(f"시험 완료: {trial['arm']} seed={trial['seed']}")
        oracles["fixed_zero_test"] = oracle_report(reference, cache, pairs["test"], args, args.output_dir / "oracle_zero_test.csv")
        common.write_json(args.output_dir / "oracle_diagnostics.json", oracles)
        rows, aggregate, deltas = tables(trials)
        for name, values in (("comparison.csv", rows), ("aggregate.csv", aggregate), ("paired_deltas.csv", deltas)):
            if values:
                common.write_csv(args.output_dir / name, list(values[0]), values)
        data.assert_unchanged()
        for mapping in (source_hashes, hashes):
            if any(common.sha256(path) != digest for path, digest in mapping.items()):
                raise RuntimeError("A source/code file changed during the run")
        report = {"trials": trials, "aggregate": aggregate, "paired_deltas": deltas, "oracles": oracles,
                  "seconds": time.perf_counter()-started, "input_and_source_unchanged": True,
                  "shared_schedule_sha256": schedules, "synthetic_test_only": synthetic,
                  "limitations": ["7 actual fingerprint participants, 5 train/1 validation/1 test; seeds do not create independent people.",
                                  "Synthetic face/finger associations; some fingers occur in several virtual users.",
                                  "An already inspected test split; exploratory comparison, not an untouched confirmation set.",
                                  "B/C marginal agreement reliability is not a helper-conditioned flip posterior or identity probability.",
                                  "Within-radius rates are diagnostics, not actual BCH/FE/key recovery rates.",
                                  "B-A isolates reliability architecture; C-B also changes binary targets, so compare bit error and calibration together.",
                                  "Each arm fixes its own quantizer before enrollment evaluation; not preservation of earlier enrolled keys.",
                                  "All trained parameters and statistics are shared model data; no per-user helper payload is created.",
                                  "Oracle thresholds know target enrollment bits and may collapse codes; they are not deployable thresholds."]}
        common.write_json(args.output_dir / "report.json", report)
        common.write_json(args.output_dir / "status.json", {"state": "completed", "seconds": report["seconds"], "synthetic_test_only": synthetic})
        log.message(f"완료 | 총 {common.duration(report['seconds'])} | {args.output_dir / 'comparison.csv'}")
        return report
    except BaseException as error:
        common.write_json(args.output_dir / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        (args.output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available() and not args.check_only:
        raise RuntimeError("CUDA unavailable; use the existing launcher or --device cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print("전체 저장 임베딩·기존 사람 단위 분할·가상 대응·평가 쌍을 확인합니다.", flush=True)
    data, users, pairs, source, source_hashes = prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output cannot be inside the input embedding directory")
    if args.check_only:
        data.assert_unchanged()
        print(json.dumps({"check_only": True, "training_started": False, "outputs_created": False,
                          "embedding_dir": str(data.directory), "pairs": {s: len(v) for s, v in pairs.items()},
                          "real_people": {s: {m: len({u.face_person if m == 'face' else u.fingerprint_person
                                                       for u in users if u.split == s}) for m in MODALITIES}
                                          for s in common.SPLITS}}, ensure_ascii=False, indent=2))
        return
    run_experiments(args, data, users, pairs, source, source_hashes)


if __name__ == "__main__":
    main()
