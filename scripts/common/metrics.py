"""Pair/context metrics and checkpoint selection rules used by S5-S9."""
from __future__ import annotations

import math

import numpy as np
import torch

from scripts import common
from . import losses as loss_ops
from .losses import identity_codes, xor_distance, hamming
from .data import MODALITIES
PRIMARY = "mean_identity_macro_within_radius_fraction"


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


def bit_statistics(enrollment, query, negatives):
    result = {}
    for modality in MODALITIES:
        e, q = enrollment[f"{modality}_bits"], query[f"{modality}_bits"]
        genuine = (e != q).float().mean(1)
        impostor = (e != q[negatives]).float().mean(1)
        combined = torch.cat((e, q)).float()
        result[modality] = {
            "genuine_hamming_fraction": float(genuine.mean()),
            "genuine_hamming_bits": float(genuine.mean() * 512),
            "genuine_hamming_p95_bits": float(torch.quantile(genuine, 0.95) * 512),
            "impostor_hamming_fraction": float(impostor.mean()),
            "impostor_hamming_bits": float(impostor.mean() * 512),
            "ones_fraction": float(combined.mean()),
            "coordinate_balance_error": float((combined.mean(0) - 0.5).square().mean()),
            "all_zero_or_one_code_fraction": float(((combined.sum(1) == 0) | (combined.sum(1) == 512)).float().mean()),
        }
    # Smaller is better. A constant code scores 0 rather than being rewarded
    # solely for zero genuine error. This is a selection metric, not a loss.
    result["selection_hamming_gap"] = sum(
        result[m]["genuine_hamming_fraction"] - result[m]["impostor_hamming_fraction"]
        for m in MODALITIES) / 2
    return result


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


def baseline_selection_guard(metrics, baseline, args):
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


def coordinate_bit_metrics(e, q, plan, args):
    metrics = bit_metrics(e, q, plan, args)
    for m in MODALITIES:
        for role, out in (("enrollment", e), ("query", q)):
            tau = out[f"{m}_tau"]
            if tau.shape != out[f"{m}_bits"].shape or tau.shape[1] != 512:
                raise RuntimeError("Expected genuine 512-D thresholds, not scalar broadcasting")
            metrics["tau"][m][role].update({
                "mean_absolute": float(tau.abs().mean()), "min": float(tau.min()), "max": float(tau.max()),
                "within_vector_std_mean": float(tau.std(dim=1, unbiased=False).mean()),
                "across_observations_std_mean": float(tau.std(dim=0, unbiased=False).mean()),
                "bound_saturation_fraction": float((tau.abs() >= .98*(args.tau_base_bound+args.tau_delta_bound)).float().mean())})
    return metrics


def coordinate_selection_guard(metrics, baseline, args):
    eligible, reasons = baseline_selection_guard(metrics, baseline, args)
    # Row balance alone permits an identical balanced code for all people.
    for m in MODALITIES:
        for key, allowance in (("constant_coordinate_fraction", args.max_constant_increase),
                               ("coordinate_balance_error", args.max_coordinate_balance_increase)):
            if metrics["bits"][m][key] > baseline["bits"][m][key] + allowance + 1e-12:
                reasons.append(f"{m}.{key} increased by more than {allowance}")
    return eligible and not reasons, reasons


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


def bch_selection_guard(metrics, baseline, args):
    _, reasons = coordinate_selection_guard(metrics, baseline, args)
    for m in MODALITIES:
        for scope, values in metrics["unique_pairs"][m]["negative_scopes"].items():
            original = baseline["unique_pairs"][m]["negative_scopes"][scope]["FAR_h"]
            if values["FAR_h"] is not None and values["FAR_h"] > original + args.max_far_h_increase + 1e-12:
                reasons.append(f"{m}.{scope}.FAR_h increased by more than {args.max_far_h_increase}")
    return not reasons, reasons


def joint_evaluation_plan(rows, users):
    # Preserve different companion contexts even when one image pair repeats.
    rows = rows[unique_indices(rows[:, :4])]
    labels = {m: x.numpy() for m, x in loss_ops.identity_codes(rows, users, "cpu").items()}
    people = {"face": np.array([users[int(i)].face_person for i in rows[:, common.USER]]),
              "fingerprint": np.array([users[int(i)].fingerprint_person for i in rows[:, common.USER]])}
    anchors = unique_indices(rows[:, [common.EF, common.EP]])
    queries = unique_indices(rows[:, [common.QF, common.QP]])
    weights = {m: loss_ops.balanced_weights(torch.from_numpy(labels[m])).numpy() for m in MODALITIES}
    swaps = {}
    for m in MODALITIES:
        choices = []
        for i, label in enumerate(labels[m]):
            candidates = (np.arange(1, len(rows)+1)+i) % len(rows)
            valid = candidates[labels[m][candidates] != label]
            if not len(valid):
                raise ValueError(f"No different-{m}-identity probe for swap diagnostics")
            choices.append(valid[0])
        swaps[m] = np.array(choices, dtype=np.int64)
    return {"rows": rows, "labels": labels, "people": people, "anchors": anchors,
            "queries": queries, "weights": weights, "swaps": swaps}


def distance_stats(values, t):
    return {"pair_count": int(len(values)), "mean_hamming_bits": float(np.mean(values)) if len(values) else None,
            "p05_hamming_bits": float(np.quantile(values, .05)) if len(values) else None,
            "within_radius_count": int((values <= t).sum()),
            "FAR_h": float((values <= t).mean()) if len(values) else None}


def summarize_joint(e, q, plan, args, tau_bound):
    rows, metrics, within = plan["rows"], {}, []
    ei, qi = plan["anchors"], plan["queries"]
    observations = np.concatenate((rows[:, [common.EF, common.EP]], rows[:, [common.QF, common.QP]]))
    oi = unique_indices(observations)
    for m, cols in (("face", [common.EF, common.QF]), ("fingerprint", [common.EP, common.QP])):
        eb, qb = e[m+"_bits"].float(), q[m+"_bits"].float()
        d = (eb != qb).sum(1).numpy()
        ok = d <= args.bch_t
        within.append(ok)
        distances = hamming(eb[ei], qb[qi], matrix=True).numpy()
        distinct = plan["labels"][m][ei, None] != plan["labels"][m][None, qi]
        same_person = plan["people"][m][ei, None] == plan["people"][m][None, qi]
        masks = {"all_different_identities": distinct, "different_people": ~same_person}
        if m == "fingerprint":
            masks["same_person_different_fingers"] = same_person & distinct
        scopes = {name: distance_stats(distances[mask], args.bch_t) for name, mask in masks.items()}
        negatives = distances[distinct]
        bits = torch.cat((eb, qb))[oi]
        obs_labels = np.concatenate((plan["labels"][m], plan["labels"][m]))[oi]
        obs_weights = loss_ops.balanced_weights(torch.from_numpy(obs_labels))
        coordinate = (bits*obs_weights[:, None]).sum(0)
        row = bits.mean(1)
        tau = torch.cat((e[m+"_tau"], q[m+"_tau"]))[oi]
        # Equal image-pair weights, retaining ALL companion contexts per pair.
        _, inverse, counts = np.unique(rows[:, cols], axis=0, return_inverse=True, return_counts=True)
        image_weights = 1/(len(counts)*counts[inverse])
        weights = plan["weights"][m].astype(np.float64)
        weights /= weights.sum()
        metrics[m] = {"context_pair_count": len(d), "genuine_hamming_bits": float(d.mean()),
            "genuine_hamming_p95_bits": float(np.quantile(d, .95)), "within_radius_count": int(ok.sum()),
            "within_radius_fraction": float(ok.mean()), "within_target_fraction": float((d <= args.bch_t-args.safety_margin).mean()),
            "zero_error_fraction": float((d == 0).mean()),
            "identity_macro_hamming_bits": float(d@weights), "identity_macro_within_radius_fraction": float(ok@weights),
            "image_pair_count": len(counts), "image_pair_context_mean_hamming_bits": float(d@image_weights),
            "image_pair_context_within_radius_fraction": float(ok@image_weights),
            "genuine_vs_impostor_auroc": roc_auc(np.r_[np.ones(len(d), bool), np.zeros(len(negatives), bool)], -np.r_[d, negatives]),
            "ones_fraction": float((row*obs_weights).sum()),
            "coordinate_balance_error": float((coordinate-.5).square().mean()),
            "constant_coordinate_fraction": float(((bits.min(0).values == 1) | (bits.max(0).values == 0)).float().mean()),
            "outside_row_band_fraction": float((((row < args.row_lower) | (row > 1-args.row_lower)).float()*obs_weights).sum()),
            "tau_mean_absolute": float(tau.abs().mean()), "tau_across_contexts_std_mean": float(tau.std(0, unbiased=False).mean()),
            "tau_genuine_mean_absolute_difference": float((e[m+"_tau"]-q[m+"_tau"]).abs().mean()),
            "tau_bound_saturation_fraction": float((tau.abs() >= .98*tau_bound).float().mean()),
            "negative_scopes": scopes}
    return {"modalities": metrics, "context_pair_count": len(rows),
            "both_within_radius_fraction": float((within[0] & within[1]).mean()),
            PRIMARY: sum(metrics[m]["identity_macro_within_radius_fraction"] for m in MODALITIES)/2,
            "mean_identity_macro_hamming_bits": sum(metrics[m]["identity_macro_hamming_bits"] for m in MODALITIES)/2}


def joint_selection_guard(candidate, baseline, args):
    reasons = []
    for m in MODALITIES:
        c, b = candidate["modalities"][m], baseline["modalities"][m]
        for key, tolerance in (("genuine_vs_impostor_auroc", args.max_auroc_drop),
                               ("identity_macro_within_radius_fraction", args.max_radius_drop)):
            if c[key] < b[key]-tolerance-1e-9:
                reasons.append(f"{m}.{key} decreased by more than {tolerance}")
        for key, tolerance in (("coordinate_balance_error", args.max_balance_increase),
                               ("constant_coordinate_fraction", args.max_constant_increase),
                               ("outside_row_band_fraction", args.max_row_outside_increase)):
            if c[key] > b[key]+tolerance+1e-9:
                reasons.append(f"{m}.{key} increased by more than {tolerance}")
        for scope in c["negative_scopes"]:
            cv, bv = (v["negative_scopes"][scope]["FAR_h"] for v in (c, b))
            if cv is not None and cv > bv+args.max_far_h_increase+1e-12:
                reasons.append(f"{m}.{scope}.FAR_h increased")
        cv, bv = (v["swaps"]["wrong_"+m]["changed_identity_macro_FAR_h"] for v in (candidate, baseline))
        if cv > bv+args.max_far_h_increase+1e-12:
            reasons.append(f"wrong_{m}.FAR_h increased")
    return not reasons, reasons


def joint_better(candidate, best, args):
    if abs(candidate[PRIMARY]-best[PRIMARY]) > 1e-12:
        return candidate[PRIMARY] > best[PRIMARY]
    return candidate["mean_identity_macro_hamming_bits"] < best["mean_identity_macro_hamming_bits"]-args.min_bit_gain
