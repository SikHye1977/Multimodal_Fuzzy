"""A/B retain V5-D loss; C/D replace squared radius after common warm-up."""
import torch
from torch.nn import functional as F

from scripts.s11_threshold_v5.losses import compute_loss as v5_loss, masked_mean, claimed_distances
from scripts.s11_threshold_v5.protocol import MODEL_NAMES

ARMS = {
    "A": {"architecture": "coordinate", "objective": "radius"},
    "B": {"architecture": "set", "objective": "radius"},
    "C": {"architecture": "coordinate", "objective": "boundary_keep"},
    "D": {"architecture": "set", "objective": "boundary_keep"},
}


def recovery_terms(distance, reference_distance, mask, t, width):
    """Forward distances are exact Hamming; backward uses the caller's STE.

    Keep is an expectation over ALL genuine queries (indicator included), not
    a reweighted mean only over successes. Empty protected sets give zero.
    """
    if reference_distance.requires_grad:
        raise ValueError("The warm-up reference must be frozen")
    protected = (reference_distance <= t) & mask
    boundary = masked_mean(torch.sigmoid((distance-(t+.5))/width), mask)
    keep = masked_mean((F.relu(distance-t)/t).square()*protected, mask)
    return boundary, keep, protected


def compute_loss(outputs, bank, positions, args, generator, *, improved=False, reference=None):
    result = v5_loss(outputs, bank, positions, "D", args, generator)
    if improved and reference is None:
        raise ValueError("C/D require a frozen warm-up reference")
    rows = bank["rows"][positions]
    for column, (modality, name) in enumerate(MODEL_NAMES.items()):
        ids = rows[:, column]
        datum = bank[modality]
        mask, target = datum["query_mask"][ids], datum["target"][ids]
        distance = claimed_distances(target, datum["queries"][ids], outputs[name+"_tau"], args.temperature)
        with torch.no_grad():
            ref_tau = torch.zeros_like(outputs[name+"_tau"]) if reference is None else reference[name+"_tau"]
            d0 = claimed_distances(target, datum["queries"][ids], ref_tau, args.temperature)
        boundary, keep, protected = recovery_terms(distance, d0, mask, args.bch_t, args.boundary_width)
        success = distance.detach() <= args.bch_t
        weights = datum["weights"][positions]
        terms = {"boundary": boundary, "keep": keep,
                 "protected_fraction": masked_mean(protected.float(), mask),
                 "gained_fraction": masked_mean((success & ~protected).float(), mask),
                 "lost_fraction": masked_mean((~success & protected).float(), mask)}
        for key, value in terms.items():
            result[modality+"_"+key] = (value*weights).mean()
    for key in ("boundary", "keep"):
        result[key] = torch.stack([result[m+"_"+key] for m in MODEL_NAMES]).mean()
    if improved:
        result["total"] = (args.bce_weight*result["bce"] + args.boundary_weight*result["boundary"]
                           + args.keep_weight*result["keep"] + args.negative_weight*result["negative"]
                           + args.tau_weight*result["tau_penalty"])
    return result
