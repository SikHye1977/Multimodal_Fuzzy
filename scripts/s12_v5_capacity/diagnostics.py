"""Fixed, nonlearned tau-only probes. They do not certify confidentiality."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.common.runtime import write_json
from scripts.s11_threshold_v5.evaluation import collect_tau, distance_summary, evaluate
from scripts.s11_threshold_v5.protocol import MODEL_NAMES


def tau_candidates(tau):
    """Predict using tau ONLY: no enrollment, anchor, query, ID or fitted probe.

    Exact zero ties are explicit. zero_query equals binarizing an all-zero
    vector; it is a missing-input control, not a real normalized embedding.
    Neither sign orientation is selected using validation/test answers.
    """
    tau = np.asarray(tau)
    if tau.ndim != 2 or tau.shape[1] != 512 or not np.isfinite(tau).all():
        raise ValueError("Expected finite tau [registrations,512]")
    return {"negative_sign": (tau < 0).astype(np.uint8),
            "nonnegative_sign": (tau >= 0).astype(np.uint8),
            "zero_query": (tau <= 0).astype(np.uint8)}


def score_tau_only(tau, target, identities, t, bch=None):
    target = np.asarray(target)
    if target.shape != np.asarray(tau).shape or not np.isin(target, (0, 1)).all():
        raise ValueError("Expected binary target matching tau")
    if len(identities) != len(target) or not len(target):
        raise ValueError("One identity per nonempty registration is required")
    candidates = tau_candidates(tau)
    result = {"zero_coordinate_fraction": float((np.asarray(tau) == 0).mean()), "rules": {}}
    for name, bits in candidates.items():
        errors = (bits != target).sum(1)
        summary = distance_summary(errors, identities, identities, t)
        summary.update(within_t_count=int((errors <= t).sum()),
                       exact_count=int((errors == 0).sum()),
                       max_error=int(errors.max()),
                       bit_accuracy=float((bits == target).mean()))
        if bch is not None:
            recovered = bch.recover(target.astype(np.uint8), bits)
            summary["bch"] = {"exact_target_rate": float((recovered == 1).mean()),
                              "no_candidate_rate": float((recovered == 0).mean()),
                              "wrong_candidate_rate": float((recovered == 2).mean())}
        result["rules"][name] = summary
    return result, candidates


def evaluate_diagnostics(model, data, plan, bank, args, *, directory=None, bch=None, progress=None):
    metrics = evaluate(model, data, plan, bank, args, directory=directory, bch=bch, progress=progress)
    taus = collect_tau(model, bank, args.batch_size)
    saved = {}
    for m in MODEL_NAMES:
        target = bank[m]["target"].cpu().numpy().astype(np.uint8)
        identities = [r["identity_id"] for r in plan["modalities"][m]]
        score, guesses = score_tau_only(taus[m], target, identities, args.bch_t, bch)
        metrics[m]["tau_only"] = score
        saved[m+"_identity"] = np.asarray(identities)
        for rule, bits in guesses.items():
            saved[f"{m}_{rule}_bits"] = bits
            saved[f"{m}_{rule}_errors"] = (bits != target).sum(1)
    if directory is not None:
        write_json(Path(directory)/"metrics.json", metrics)
        np.savez_compressed(Path(directory)/"tau_only_bits.npz", **saved)
    return metrics


class AnchorShiftControl(nn.Module):
    """Enrollment-only positive control: deliberately exposes every anchor bit.

    This is NOT a secure construction or a candidate for model selection.
    It distinguishes learning from simply moving each threshold toward c.
    """
    def __init__(self, bound):
        super().__init__()
        self.bound = bound

    def forward(self, face, finger):
        return {name+"_tau": -self.bound*(2*(x.mean(1) >= 0).to(x.dtype)-1)
                for name, x in (("face", face), ("finger", finger))}
