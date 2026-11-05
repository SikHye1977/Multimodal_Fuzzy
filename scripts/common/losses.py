"""Shared threshold and reliability objectives; no training-script imports."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from scripts import common
from .data import MODALITIES


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


def soft_threshold_loss(enrollment, query, labels, settings):
    """Exactly the prior D objective, additionally retaining per-modality terms."""
    per_modality = {}
    for m in MODALITIES:
        e, q = enrollment[f"{m}_soft_bits"], query[f"{m}_soft_bits"]
        weights = balanced_weights(labels[m])
        negative, ordinary, hard = mined_negative(e, q, labels[m], getattr(settings, f"{m}_margin"),
                                                       settings.hard_k, settings.hard_mix)
        values = {"positive": (e - q).abs().mean(), "negative": negative,
                  "negative_all": ordinary, "negative_hard": hard,
                  "balance": sum(((weights[:, None] * x).sum(0) - 0.5).square().mean() for x in (e, q)) / 2,
                  "row": sum(row_penalty(x, weights, settings.row_lower) for x in (e, q)) / 2}
        for term in ("negative", "balance", "row"):
            values[f"weighted_{term}"] = getattr(settings, f"{term}_weight") * values[term]
        values["total"] = values["positive"] + sum(values[f"weighted_{k}"] for k in ("negative", "balance", "row"))
        per_modality[m] = values
    result = {key: sum(per_modality[m][key] for m in MODALITIES) / 2 for key in per_modality["face"]}
    # Preserve the original D arithmetic order for the optimization objective.
    result["total"] = result["positive"] + result["weighted_negative"] + result["weighted_balance"] + result["weighted_row"]
    result.update({f"{m}_{key}": value for m, values in per_modality.items() for key, value in values.items()})
    return result


def coordinate_threshold_loss(model, e, q, rows, users, args):
    labels = identity_codes(rows, users, args.device)
    loss = soft_threshold_loss(e, q, labels, loss_settings(args))
    stability = torch.stack([((e[f"{m}_tau"] - q[f"{m}_tau"]) / model.config.tau_bound).square().mean()
                             for m in MODALITIES]).mean()
    residual = torch.stack([(out[f"{m}_delta_tau"] / model.config.tau_delta_bound).square().mean()
                            for m in MODALITIES for out in (e, q)]).mean()
    return {"total": loss["total"] + args.tau_stability_weight*stability + args.tau_residual_weight*residual,
            **{key: value for key, value in loss.items() if key != "total"},
            "tau_stability": stability, "tau_residual": residual}


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


def mined_hamming_negative(distances, labels, margin, hard_k, hard_mix):
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
    weights = balanced_weights(labels)
    ordinary, hard = (ordinary_rows * weights).sum(), (hard_rows * weights).sum()
    return (1 - hard_mix) * ordinary + hard_mix * hard, ordinary, hard


def bch_threshold_loss(model, e, q, rows, users, args, arm):
    if arm == "A":
        # Exact original implementation, including regularizers and mining.
        return coordinate_threshold_loss(model, e, q, rows, users, args)
    if arm not in ("B", "C"):
        raise ValueError(f"Unknown arm {arm}")
    settings = loss_settings(args)
    labels = identity_codes(rows, users, args.device)
    terms = defaultdict(list)
    for m in MODALITIES:
        eb = ste_bits(e[f"{m}_soft_bits"], e[f"{m}_bits"])
        qb = ste_bits(q[f"{m}_soft_bits"], q[f"{m}_bits"])
        d = xor_distance(eb, qb)
        positive = (d / 512).mean() if arm == "B" else bch_positive(d, args.bch_t, args.safety_margin)
        neg, ordinary, hard = mined_hamming_negative(xor_distance(eb, qb, matrix=True) / 512,
            labels[m], getattr(settings, m + "_margin"), settings.hard_k, settings.hard_mix)
        weights = balanced_weights(labels[m])
        balance = sum(((weights[:, None] * b).sum(0) - .5).square().mean() for b in (eb, qb)) / 2
        row = sum(row_penalty(b, weights, settings.row_lower) for b in (eb, qb)) / 2
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


def ste(soft, hard):
    return hard.to(soft.dtype).detach() + (soft - soft.detach())


def hamming(a, b, matrix=False):
    if matrix:
        return a.sum(1)[:, None] + b.sum(1)[None, :] - 2*(a @ b.T)
    return (a + b - 2*a*b).sum(1)


def negative_loss(distance, labels, margin, hard_k, hard_mix):
    identities, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    if len(identities) < 2:
        raise ValueError("Each batch needs at least two identities per modality")
    penalty = (F.relu(margin-distance)/margin).square()
    membership = F.one_hot(inverse, len(identities)).to(distance.dtype)
    grouped = penalty @ (membership/counts.to(distance.dtype))
    valid = labels[:, None] != identities[None, :]
    ordinary = (grouped*valid).sum(1)/valid.sum(1)
    closest = torch.full_like(grouped, float("inf")).scatter_reduce(
        1, inverse[None, :].expand(len(labels), -1), distance, reduce="amin", include_self=True)
    hard = closest.masked_fill(~valid, float("inf")).topk(min(hard_k, len(identities)-1), largest=False).values
    weights = balanced_weights(labels)
    return ((1-hard_mix)*ordinary + hard_mix*(F.relu(margin-hard)/margin).square().mean(1)).dot(weights)


def joint_threshold_loss(model, e, q, rows, users, args):
    labels = identity_codes(rows, users, e["face_soft_bits"].device)
    terms, losses = defaultdict(list), {}
    for m in MODALITIES:
        eb, qb = (ste(out[m+"_soft_bits"], out[m+"_bits"]) for out in (e, q))
        weights = balanced_weights(labels[m])
        d = hamming(eb, qb)
        positive = ((F.relu(d-(args.bch_t-args.safety_margin))/args.bch_t).square()*weights).sum()
        negative = negative_loss(hamming(eb, qb, matrix=True), labels[m],
            getattr(args, m+"_margin_bits"), args.hard_k, args.hard_mix)
        balance = sum(((weights[:, None]*b).sum(0)-.5).square().mean() for b in (eb, qb))/2
        row = sum(row_penalty(b, weights, args.row_lower) for b in (eb, qb))/2
        residual = sum(((out[m+"_delta_tau"]/model.config.tau_delta_bound).square().mean(1)*weights).sum()
                       for out in (e, q))/2
        # Different tau can compensate noise; do not force tau agreement.
        for key, val in {"positive": positive, "negative": negative, "balance": balance,
                         "row": row, "residual": residual}.items():
            terms[key].append(val)
            losses[m+"_"+key] = val
        losses[m+"_hamming_bits"] = (d.detach()*weights).sum()
    losses.update({key: torch.stack(values).mean() for key, values in terms.items()})
    losses["total"] = (losses["positive"] + args.negative_weight*losses["negative"]
                       + args.balance_weight*losses["balance"] + args.row_weight*losses["row"]
                       + args.residual_weight*losses["residual"])
    return losses


def reliability_loss(e, q):
    terms = []
    for m in MODALITIES:
        target = (e[f"{m}_bits"] == q[f"{m}_bits"]).float().detach()
        terms.extend(F.binary_cross_entropy_with_logits(o[f"{m}_reliability_logits"], target) for o in (e, q))
    # No class weighting, forced 50:50 flip labels, or impostor reliability targets.
    return torch.stack(terms).mean()
