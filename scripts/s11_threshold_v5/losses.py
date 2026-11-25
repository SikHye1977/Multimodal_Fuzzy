"""Controlled A/B/C/D comparisons; all anchors/tau are enrollment-only."""
import torch
from torch.nn import functional as F

from .protocol import MODEL_NAMES

ARMS = ("A", "B", "C", "D")
SETTINGS = {
    "A": {"architecture": "legacy_mlp", "positive": "enrollment", "negatives": False},
    "B": {"architecture": "legacy_mlp", "positive": "query", "negatives": False},
    "C": {"architecture": "legacy_mlp", "positive": "query", "negatives": True},
    "D": {"architecture": "coordinate_mlp", "positive": "query", "negatives": True},
}
DESCRIPTIONS = {"A": "V4-style MLP: enrollment positive loss only",
                "B": "same MLP: disjoint genuine query loss",
                "C": "B + mixed random/nearest impostor query loss",
                "D": "C with coordinate-shared MLP and local enrollment statistics"}


def ste_bits(embedding, tau, temperature):
    margin = embedding - tau
    soft = (margin/temperature).sigmoid()
    return (margin >= 0).to(soft.dtype).detach() + (soft - soft.detach())


def claimed_distances(target, embeddings, tau, temperature):
    """embeddings [B,K,512]; every donor uses its claimant's tau [B,512]."""
    bits = ste_bits(embeddings, tau[:, None, :], temperature)
    return (target[:, None, :]*(1-bits) + (1-target[:, None, :])*bits).sum(-1)


@torch.no_grad()
def select_negatives(target, tau, identities, bank, arm, args, generator):
    n_groups, batch = len(bank["members"]), len(target)
    k = min(args.negative_identities, n_groups-1)
    hard_k = max(1, round(k*args.hard_mix)) if args.hard_mix > 0 else 0
    hard_k = min(k, hard_k)
    forbidden = torch.arange(n_groups, device=tau.device)[None, :] == identities[:, None]
    parts = []
    if hard_k:
        distances = []
        for start in range(0, len(bank["pool"]), args.distance_chunk):
            bits = bank["pool"][None, start:start+args.distance_chunk] >= tau[:, None, :]
            distances.append((bits != target[:, None, :]).sum(-1))
        distances = torch.cat(distances, dim=1)
        labels = bank["pool_labels"][None, :].expand(batch, -1)
        nearest = torch.full((batch, n_groups), 513, device=tau.device, dtype=torch.long)
        nearest.scatter_reduce_(1, labels, distances, reduce="amin", include_self=True)
        hard_ids = nearest.masked_fill(forbidden, 514).topk(hard_k, largest=False).indices
        positions = torch.arange(len(bank["pool"]), device=tau.device)[None, :].expand(batch, -1)
        candidates = positions.masked_fill(distances != nearest.gather(1, labels), len(bank["pool"]))
        closest_index = torch.full_like(nearest, len(bank["pool"]))
        closest_index.scatter_reduce_(1, labels, candidates, reduce="amin", include_self=True)
        parts.append(closest_index.gather(1, hard_ids))
        forbidden.scatter_(1, hard_ids, True)
    if k > hard_k:
        scores = torch.rand((batch, n_groups), device=tau.device, generator=generator)
        random_ids = scores.masked_fill(forbidden, 2).topk(k-hard_k, largest=False).indices
        offsets = (torch.rand(random_ids.shape, device=tau.device, generator=generator)
                   * bank["member_counts"][random_ids]).long()
        parts.append(bank["members"][random_ids, offsets])
    selected = torch.cat(parts, dim=1)
    if bool((bank["pool_labels"][selected] == identities[:, None]).any()):
        raise RuntimeError("A genuine identity was selected as a negative")
    return selected


def masked_mean(values, mask):
    """One mean per registration; large query sets do not dominate identities."""
    counts = mask.sum(1)
    if bool((counts == 0).any()):
        raise ValueError("No positive captures in a registration")
    return values.masked_fill(~mask, 0).sum(1)/counts


def compute_loss(outputs, bank, positions, arm, args, generator):
    result = {}
    settings = SETTINGS[arm]
    rows = bank["rows"][positions]
    for column, (m, name) in enumerate(MODEL_NAMES.items()):
        tau = outputs[name+"_tau"]
        target = outputs[name+"_target_bits"]
        reference = bank[m]["target"][rows[:, column]]
        if not torch.equal(target, reference):
            raise RuntimeError("The fixed tau=0 enrollment target changed")
        weights = bank[m]["weights"][positions]
        if settings["positive"] == "enrollment":
            logits = outputs[name+"_bit_logits"]
            mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
        else:
            # Train through enrollment-predicted tau, not through a query encoder.
            queries = bank[m]["queries"][rows[:, column]]
            mask = bank[m]["query_mask"][rows[:, column]]
            logits = (queries-tau[:, None])/args.temperature
        bce = masked_mean(F.binary_cross_entropy_with_logits(
            logits, target[:, None].expand_as(logits), reduction="none").mean(-1), mask)
        soft = logits.sigmoid()
        bits = (logits >= 0).to(soft.dtype).detach() + (soft-soft.detach())
        distance = (target[:, None]*(1-bits) + (1-target[:, None])*bits).sum(-1)
        radius = masked_mean((F.relu(distance-(args.bch_t-args.safety_margin))/args.bch_t).square(), mask)
        regularizer = (tau/args.tau_bound).square().mean(1)
        negative, active = torch.zeros_like(bce), torch.zeros_like(bce)
        if settings["negatives"]:
            ids = bank[m]["identity"][rows[:, column]]
            selected = select_negatives(target, tau, ids, bank[m], arm, args, generator)
            dneg = claimed_distances(target, bank[m]["pool"][selected], tau, args.temperature)
            margin = getattr(args, m+"_negative_margin")
            negative = (F.relu(margin-dneg)/margin).square().mean(1)
            active = (dneg.detach() < margin).float().mean(1)
        for key, values in {"bce": bce, "radius": radius, "negative": negative,
                            "tau_penalty": regularizer, "negative_active_fraction": active,
                            "positive_hamming": masked_mean(distance.detach(), mask)}.items():
            result[m+"_"+key] = (values*weights).mean()
    for key in ("bce", "radius", "negative", "tau_penalty"):
        result[key] = torch.stack([result[m+"_"+key] for m in MODEL_NAMES]).mean()
    result["total"] = (args.bce_weight*result["bce"] + args.radius_weight*result["radius"]
                       + args.negative_weight*result["negative"] + args.tau_weight*result["tau_penalty"])
    return result
