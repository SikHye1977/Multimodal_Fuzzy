"""Matched schedules, output collection and coordinate-model initialization."""
from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np
import torch

from scripts import common
from .data import MODALITIES
from Thresholding_Model_512 import Thresholding512Config, ThresholdingModel512
from thresholding.variability_model import ModelConfig, ReliabilityHead


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


def assert_fixed(model, state):
    current = model.state_dict()
    if any(not torch.equal(current[k].detach().cpu(), v) for k, v in state.items()):
        raise RuntimeError("Reliability training changed a threshold or training-only statistic")
    if any(p.grad is not None for p in model.thresholds.parameters()):
        raise RuntimeError("Reliability loss reached the frozen thresholds")


def make_coordinate_model(args, mode, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ThresholdingModel512(Thresholding512Config(
            mode, args.temperature, args.tau_base_bound, args.tau_delta_bound, args.hidden_dim))
        # Reliability initialization must not depend on threshold parameter count.
        torch.manual_seed((seed + 100003) % 2**32)
        for m in MODALITIES:
            model.reliability[m].load_state_dict(ReliabilityHead(ModelConfig(arm="B")).state_dict())
    return model.to(args.device).set_stage("inference")


def quantize_pair(model, cache, rows):
    ef, ep, qf, qp = common.model_inputs(cache, rows)
    return tuple({f"{m}_{key}": value for m, x in zip(MODALITIES, inputs)
                  for key, value in model.quantize(m, x).items()} for inputs in ((ef, ep), (qf, qp)))
