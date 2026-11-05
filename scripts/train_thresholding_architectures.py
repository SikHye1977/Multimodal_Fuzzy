#!/usr/bin/env python3
"""Compare fixed zero, shared scalar, shared dual, and independent thresholds.

All learned architectures use the preceding D loss, a +/-0.02 scalar bound,
the same frozen 512-D embeddings, participant splits and batch schedules.
Only main() runs training. Imports and --help never start an experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_thresholding_abcd as abcd

common, previous = abcd.common, abcd.previous
MODALITIES = common.MODALITIES
ARMS = ("shared_scalar", "shared_dual", "independent")
DESCRIPTIONS = {
    "shared_scalar": "얼굴+지문 입력, 공통 scalar tau 하나",
    "shared_dual": "얼굴+지문 입력, 공유 계층 뒤 모달리티별 scalar tau",
    "independent": "각 모달리티 입력만 사용하는 독립 scalar tau 예측기",
}


@dataclass(frozen=True)
class ArchitectureConfig:
    architecture: str = "shared_scalar"
    temperature: float = 0.02
    tau_bound: float = 0.02

    def __post_init__(self):
        if self.architecture not in (*ARMS, "fixed_zero"):
            raise ValueError(f"Unknown architecture: {self.architecture}")
        for name in ("temperature", "tau_bound"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


class ArchitectureThresholdModel(nn.Module):
    """Predict thresholds only; original coordinates are never transformed.

    Reliability prediction and FE are deliberately absent from this experiment.
    The new checkpoint format prevents an old single-tau loader misreading it.
    """
    checkpoint_format = "multimodal-threshold-architecture-v1"

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ArchitectureConfig()
        self.register_buffer("dtype_anchor", torch.empty(0), persistent=False)
        if self.config.architecture == "fixed_zero":
            return
        self.face_mlp = nn.Sequential(nn.Linear(512, 128), nn.GELU())
        self.fingerprint_mlp = nn.Sequential(nn.Linear(512, 128), nn.GELU())
        if self.config.architecture == "independent":
            self.contexts = nn.ModuleDict({m: nn.Sequential(nn.Linear(128, 64), nn.GELU())
                                           for m in MODALITIES})
        else:
            self.shared_mlp = nn.Sequential(nn.Linear(256, 64), nn.GELU())
        names = ("common",) if self.config.architecture == "shared_scalar" else MODALITIES
        self.heads = nn.ModuleDict({name: nn.Linear(64, 1) for name in names})
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, face_embedding, fingerprint_embedding):
        for x in (face_embedding, fingerprint_embedding):
            if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 512 or len(x) < 1:
                raise ValueError("Expected nonempty (B,512) tensors")
            if x.dtype != self.dtype_anchor.dtype or x.device != self.dtype_anchor.device:
                raise ValueError("Input dtype/device must match the model")
            if not bool(torch.isfinite(x).all()):
                raise ValueError("Nonfinite embedding")
        if face_embedding.shape != fingerprint_embedding.shape:
            raise ValueError("Face and fingerprint shapes differ")
        embeddings = dict(zip(MODALITIES, (face_embedding, fingerprint_embedding)))
        arch = self.config.architecture
        if arch == "fixed_zero":
            taus = {m: x.new_zeros((len(x), 1)) for m, x in embeddings.items()}
        else:
            features = {m: getattr(self, f"{m}_mlp")(x) for m, x in embeddings.items()}
            if arch == "independent":
                contexts = {m: self.contexts[m](features[m]) for m in MODALITIES}
            else:
                context = self.shared_mlp(torch.cat([features[m] for m in MODALITIES], dim=1))
                contexts = {m: context for m in MODALITIES}
            if arch == "shared_scalar":
                tau = self.config.tau_bound * torch.tanh(self.heads["common"](context))
                taus = {m: tau for m in MODALITIES}
            else:
                taus = {m: self.config.tau_bound * torch.tanh(self.heads[m](contexts[m])) for m in MODALITIES}
        output = {}
        for m, x in embeddings.items():
            tau = taus[m]
            output.update({f"{m}_tau": tau, f"{m}_soft_bits": torch.sigmoid((x - tau) / self.config.temperature),
                           f"{m}_bits": (x >= tau).to(torch.uint8), f"{m}_distance": (x - tau).abs()})
        return output

    def shared_parameter_names(self):
        if self.config.architecture == "independent":
            return set()
        return {name for name, _ in self.named_parameters()
                if self.config.architecture == "shared_scalar" or not name.startswith("heads.")}

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != cls.checkpoint_format:
            raise ValueError("Use the matching architecture checkpoint loader")
        model = cls(ArchitectureConfig(**saved["config"]))
        model.load_state_dict(saved["state_dict"], strict=True)
        return model.to(device).eval()


def initial_states(module, seed, temperature, bound):
    """Shared scalar exactly reuses the original D initialization for this seed.

    Shared dual copies both heads from that scalar head, so its initial outputs
    match exactly. Independent branches share the first-layer/head tensors with
    those models, but their 128->64 contexts are separately initialized. Their
    outputs and full parameter counts are not claimed to be identical.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        original = module.ThresholdingModel(module.ThresholdingConfig(temperature, bound))
        result = {}
        for arm in ARMS:
            torch.manual_seed((seed + 100003) % 2**32)
            model = ArchitectureThresholdModel(ArchitectureConfig(arm, temperature, bound))
            for m in MODALITIES:
                getattr(model, f"{m}_mlp").load_state_dict(getattr(original, f"{m}_mlp").state_dict())
            if arm != "independent":
                model.shared_mlp.load_state_dict(original.shared_mlp.state_dict())
            for head in model.heads.values():
                head.load_state_dict(original.threshold_head.state_dict())
            result[arm] = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return result


def make_model(arm, args, state=None):
    model = ArchitectureThresholdModel(ArchitectureConfig(arm, args.temperature, args.tau_bound))
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.to(args.device)


def zero_tau_model(model):
    with torch.no_grad():
        for head in model.heads.values():
            head.weight.zero_()
            head.bias.zero_()
    return model.eval()


def compute_losses(enrollment, query, labels, settings):
    """Exactly the prior D objective, additionally retaining per-modality terms."""
    per_modality = {}
    for m in MODALITIES:
        e, q = enrollment[f"{m}_soft_bits"], query[f"{m}_soft_bits"]
        weights = abcd.balanced_weights(labels[m])
        negative, ordinary, hard = abcd.mined_negative(e, q, labels[m], getattr(settings, f"{m}_margin"),
                                                       settings.hard_k, settings.hard_mix)
        values = {"positive": (e - q).abs().mean(), "negative": negative,
                  "negative_all": ordinary, "negative_hard": hard,
                  "balance": sum(((weights[:, None] * x).sum(0) - 0.5).square().mean() for x in (e, q)) / 2,
                  "row": sum(abcd.row_penalty(x, weights, settings.row_lower) for x in (e, q)) / 2}
        for term in ("negative", "balance", "row"):
            values[f"weighted_{term}"] = getattr(settings, f"{term}_weight") * values[term]
        values["total"] = values["positive"] + sum(values[f"weighted_{k}"] for k in ("negative", "balance", "row"))
        per_modality[m] = values
    result = {key: sum(per_modality[m][key] for m in MODALITIES) / 2 for key in per_modality["face"]}
    # Preserve the original D arithmetic order for the optimization objective.
    result["total"] = result["positive"] + result["weighted_negative"] + result["weighted_balance"] + result["weighted_row"]
    result.update({f"{m}_{key}": value for m, values in per_modality.items() for key, value in values.items()})
    return result


def gradient_diagnostics(model, losses):
    """First training batch of each epoch only; never a selection criterion."""
    named = list(model.named_parameters())
    gradients = {m: torch.autograd.grad(losses[f"{m}_total"], [p for _, p in named],
                                      retain_graph=True, allow_unused=True) for m in MODALITIES}
    shared = model.shared_parameter_names()
    values = {}
    shared_norm = {}
    for m, gs in gradients.items():
        norm = sum(float(g.detach().square().sum()) for g in gs if g is not None)
        values[f"{m}_gradient_norm"] = math.sqrt(norm)
        shared_norm[m] = sum(float(g.detach().square().sum()) for (name, _), g in zip(named, gs)
                             if name in shared and g is not None)
    dot = sum(float((a.detach() * b.detach()).sum()) for (name, _), a, b in
              zip(named, gradients["face"], gradients["fingerprint"])
              if name in shared and a is not None and b is not None)
    denominator = math.sqrt(shared_norm["face"] * shared_norm["fingerprint"])
    values["shared_gradient_cosine"] = max(-1., min(1., dot / denominator)) if denominator else None
    values["shared_parameter_count"] = sum(p.numel() for name, p in named if name in shared)
    values["scope"] = "first_training_batch_of_epoch; unweighted_per_modality_total; before_clipping"
    return values


def save_checkpoint(model, path, seed, epoch, origin):
    payload = {"format": model.checkpoint_format, "config": asdict(model.config),
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "purpose": "threshold_architecture_research_only; no_reliability_or_FE",
               "experiment": {"seed": seed, "epoch": epoch, "origin": origin}}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


# Share the existing identity masks and checkpoint-selection rules unchanged.
evaluation_plan = abcd.evaluation_plan
selection_guard = abcd.selection_guard
TABLE_METRICS = abcd.TABLE_METRICS


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
            for key in ("face_tau", "fingerprint_tau", "face_bits", "fingerprint_bits"):
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
            "genuine_vs_impostor_auroc": abcd.roc_auc(np.r_[np.ones(len(g), dtype=bool), np.zeros(len(imp), dtype=bool)], -np.r_[g, imp]),
            "legacy_fixed_negative_mean_bits": float(legacy.mean()),
            "ones_fraction": float(hard.mean()),
            "outside_row_band_fraction": float(((row_mean < args.row_lower) | (row_mean > 1 - args.row_lower)).float().mean()),
            "extreme_code_fraction": float(((row_mean < 0.05) | (row_mean > 0.95)).float().mean()),
            "per_code_ones_min": float(row_mean.min()), "per_code_ones_max": float(row_mean.max()),
            "constant_coordinate_fraction": float(((column_mean == 0) | (column_mean == 1)).float().mean()),
            "coordinate_balance_error": float((column_mean - 0.5).square().mean()),
        }
    tau = {}
    for m in MODALITIES:
        tau[m] = {name: {"mean": float(out[f"{m}_tau"].mean()),
                         "std": float(out[f"{m}_tau"].std(unbiased=False)),
                         "min": float(out[f"{m}_tau"].min()), "max": float(out[f"{m}_tau"].max())}
                  for name, out in (("enrollment", e), ("query", q))}
        tau[m]["genuine_mean_absolute_difference"] = float((e[f"{m}_tau"] - q[f"{m}_tau"]).abs().mean())
    tau["cross_modality_mean_absolute_difference"] = {
        name: float((out["face_tau"] - out["fingerprint_tau"]).abs().mean())
        for name, out in (("enrollment", e), ("query", q))}
    return {"pair_count": len(rows), "enrollment_contexts": len(plan["anchors"]), "bits": metrics, "tau": tau,
            "selection_score": sum(metrics[m]["genuine_hamming_fraction"] for m in MODALITIES) / 2}



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
                             "tau_enrollment_mean": metrics["tau"][m]["enrollment"]["mean"],
                             "tau_enrollment_std": metrics["tau"][m]["enrollment"]["std"],
                             "tau_query_mean": metrics["tau"][m]["query"]["mean"],
                             "tau_genuine_absolute_difference": metrics["tau"][m]["genuine_mean_absolute_difference"]})
        keys = TABLE_METRICS[:7]
        for arm in ARMS:
            for m in MODALITIES:
                selected = [r for r in rows if r["split"] == split and r["arm"] == arm and r["modality"] == m]
                aggregate.append({"split": split, "arm": arm, "modality": m, "seed_count": len(selected),
                                  "trained_model_count": sum(r["selected_origin"] == "trained_network" for r in selected),
                                  **{f"{key}_{stat}": float(getattr(np, stat)([r[key] for r in selected]))
                                     for key in keys for stat in ("mean", "std")}})
        for before, after in ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[2]), (ARMS[0], ARMS[2])):
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
    parser.add_argument("--hard-k", type=int, default=4, help="Nearest DISTINCT negative identities per anchor")
    parser.add_argument("--tau-bound", type=float, default=0.02, help="All learned architectures use tau=bound*tanh(logit)")
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
    for name in ("lr", "grad_clip", "tau_bound"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be positive and finite")
    limits = {"min_delta": 1, "max_impostor_drop_bits": 512, "max_impostor_p05_drop_bits": 512,
              "max_auroc_drop": 1, "max_genuine_increase_bits": 512,
              "max_genuine_p95_increase_bits": 512, "max_row_outside_increase": 1}
    for name, upper in limits.items():
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= upper:
            parser.error(f"{name} must be finite and in [0,{upper}]")
    try:
        abcd.loss_settings(args)
    except ValueError as error:
        parser.error(str(error))
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT / "thresholding/results" / datetime.now().strftime("tau_architectures_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT / name for name in ("muct-master", "UareU", "feature_extractors", "BCH", "scripts")]
    protected += [args.source_run] + ([args.embedding_dir] if args.embedding_dir else [])
    if any(args.output_dir.is_relative_to(p.resolve()) for p in protected):
        parser.error("Output must be separate from input datasets, archives, source code, and the source run")
    if args.output_dir.exists():
        parser.error("Output directory already exists")
    return args


def train_arm(state, cache, data, users, validation, plan, baseline, args, settings,
              arm, seed, directory, remaining_trials, schedules):
    directory.mkdir()
    log = common.Log(directory)
    common.write_json(directory / "status.json", {"state": "running"})
    model = make_model(arm, args, state)
    initial = evaluate(model, cache, validation, args, plan)
    eligible, initial_reasons = selection_guard(initial, baseline, args)
    best_metrics, best_epoch, origin = baseline, None, "tau_zero_fallback"
    best = baseline["selection_score"]
    best_path = directory / "best_threshold.pt"
    fallback = zero_tau_model(deepcopy(model))
    save_checkpoint(fallback, best_path, seed, None, origin)
    del fallback
    if eligible and initial["selection_score"] < best - args.min_delta:
        best_metrics, best_epoch, origin, best = initial, 0, "initial_network", initial["selection_score"]
        save_checkpoint(model, best_path, seed, 0, origin)
    log.record({"epoch": 0, "validation": initial, "eligible": eligible, "ineligible_reasons": initial_reasons,
                "selected_origin": origin, "best_epoch": best_epoch})
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    progress = common.Progress(f"{arm} seed={seed}", args.epochs, remaining_trials * args.epochs, args.log_interval)
    log.message(f"\n[{arm}, seed={seed}] {DESCRIPTIONS[arm]} | tau bound ±{args.tau_bound:g}")
    stale, epoch = 0, 0
    first_row_active = {m: None for m in MODALITIES}
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            rows = common.make_pairs(data, users, "train", args.data_seed, epoch)
            rng = np.random.default_rng(np.random.SeedSequence([args.data_seed, epoch, 9001]))
            batches, padded = common.training_batches(rows, users, args.batch_size, rng)
            digest = previous.schedule_digest(batches)
            if schedules.setdefault(epoch, digest) != digest:
                raise RuntimeError("Shared training rows/order changed across seeds or architectures")
            steps = len(batches) + math.ceil(len(validation) / args.batch_size)
            sums, processed, row_active_batches, diagnostics = {}, 0, {m: 0 for m in MODALITIES}, {}
            for step, (batch, _) in enumerate(batches, 1):
                optimizer.zero_grad(set_to_none=True)
                e, q = common.forward_pair(model, cache, batch)
                losses = compute_losses(e, q, abcd.identity_codes(batch, users, args.device), settings)
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Nonfinite loss, arm={arm}, seed={seed}, epoch={epoch}")
                if step == 1:
                    diagnostics = gradient_diagnostics(model, losses)
                losses["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
                if step == 1:
                    diagnostics["combined_gradient_norm_before_clipping"] = float(grad_norm)
                    diagnostics["clip_activated"] = float(grad_norm) > args.grad_clip
                    if any(p.grad is None for p in params):
                        raise RuntimeError("Missing gradient in threshold model")
                optimizer.step()
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach()) * len(batch)
                for m in MODALITIES:
                    if float(losses[f"{m}_weighted_row"].detach()) > 0:
                        row_active_batches[m] += 1
                        if first_row_active[m] is None:
                            first_row_active[m] = epoch
                processed += len(batch)
                progress.update(epoch, step, steps, "train", sums["total"] / processed,
                                force=step == 1 or step == len(batches))
            metrics = evaluate(model, cache, validation, args, plan,
                               on_batch=lambda n: progress.update(epoch, len(batches) + n, steps,
                                                                  "validation", sums["total"] / processed))
            eligible, reasons = selection_guard(metrics, baseline, args)
            improved = eligible and metrics["selection_score"] < best - args.min_delta
            if improved:
                best_metrics, best_epoch, best, origin, stale = metrics, epoch, metrics["selection_score"], "trained_network", 0
                save_checkpoint(model, best_path, seed, epoch, origin)
            else:
                stale += 1
            save_checkpoint(model, directory / "last_threshold.pt", seed, epoch, "last_epoch")
            log.record({"epoch": epoch, "train_loss": {key: value / processed for key, value in sums.items()},
                        "validation": metrics, "eligible": eligible, "ineligible_reasons": reasons,
                        "best": improved, "best_epoch": best_epoch, "selected_origin": origin,
                        "gradient_diagnostics": diagnostics, "row_active_batches": row_active_batches,
                        "schedule_sha256": digest, "original_pairs": len(rows), "processed_pairs": processed,
                        "supplementary_pairs": padded, "seconds": time.perf_counter() - epoch_started})
            progress.finish_line()
            log.message(f"[{arm} seed={seed} {epoch}/{args.epochs}] loss={sums['total']/processed:.5f} | "
                        f"적격={eligible} | best={origin}:{best_epoch} | patience={stale}/{args.patience}")
            for m in MODALITIES:
                b, tau = metrics["bits"][m], metrics["tau"][m]
                log.message(f"  {m}: 동일 {b['genuine_hamming_bits']:.2f}, 상이 P05 "
                            f"{b['impostor_hamming_p05_bits']:.2f} bit | AUC {b['genuine_vs_impostor_auroc']:.5f} | "
                            f"쏠림 {100*b['outside_row_band_fraction']:.2f}% | "
                            f"tau E/Q {tau['enrollment']['mean']:.6f}/{tau['query']['mean']:.6f}")
            cosine = diagnostics["shared_gradient_cosine"]
            log.message(f"  첫 배치 공유 gradient cosine: {cosine if cosine is not None else 'N/A'} | "
                        f"코드 쏠림 loss 활성 배치: {row_active_batches}")
            if reasons:
                log.message("  선택 제외: " + "; ".join(reasons))
            if stale >= args.patience:
                log.message("검증에서 적격 개선이 없어 이 조건을 조기 종료합니다.")
                break
    except BaseException as error:
        progress.finish_line()
        try:
            save_checkpoint(model, directory / "interrupted.pt", seed, epoch, "interrupted")
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
              "initial_state_sha256": previous.state_digest(state), "tau_bound": args.tau_bound,
              "trainable_parameters": sum(p.numel() for p in params),
              "first_row_loss_active_epoch": first_row_active,
              "seconds": time.perf_counter() - progress.start, "checkpoint": str(best_path),
              "reliability_trained": False}
    common.write_json(directory / "summary.json", result)
    common.write_json(directory / "status.json", {"state": "trained_awaiting_test"})
    if origin != "trained_network":
        log.message(f"학습 후 적격 개선 없음: {origin} 유지")
    return result


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
    data, users, pairs, source, source_hashes = abcd.prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output cannot be inside the embedding directory")
    args.temperature, args.data_seed = float(source["temperature"]), int(source["seed"])
    settings = abcd.loss_settings(args)
    module = common.load_model_module()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = common.Log(args.output_dir)
    try:
        common.write_json(args.output_dir / "status.json", {"state": "running"})
        for name in previous.MANIFESTS:
            shutil.copy2(args.source_run / name, args.output_dir / name)
        code_paths = (Path(__file__), Path(abcd.__file__), Path(previous.__file__), Path(common.__file__),
                      ROOT / "Thresholding Model.py", ROOT / "scripts/train_reliability_distance.py")
        code_hashes = {str(path): common.sha256(path) for path in code_paths}
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"started_at_utc": datetime.now(timezone.utc).isoformat(), "arms": DESCRIPTIONS,
                       "baseline": "fixed_zero; evaluated once per split, no optimization",
                       "loss_settings": asdict(settings), "input_sha256": data.hashes,
                       "source_sha256": source_hashes, "code_sha256": code_hashes,
                       "python": platform.python_version(), "torch": str(torch.__version__),
                       "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
                       "variance_loss": False, "reliability_training": False, "FE_evaluation": False,
                       "test_used_for_selection": False, "checkpoint_resuming_supported": False,
                       "initialization": initial_states.__doc__,
                       "parameter_counts": {arm: sum(p.numel() for p in make_model(arm, args).parameters()) for arm in ARMS},
                       "evaluation": "Same hard metrics and all valid modality-specific negative pairs as ABCD.",
                       "selection": "Same fixed tau=0 validation guards and joint checkpoint for ALL architectures. No per-modality epoch mixing.",
                       "gradient_diagnostics": "First training batch of each epoch only; shared-parameter cosine is null for independent branches."})
        common.write_json(args.output_dir / "config.json", config)
        cache = {m: torch.tensor(value, device=args.device) for m, value in data.arrays.items()}
        plans = {split: evaluation_plan(pairs[split], users, args.device) for split in ("validation", "test")}
        reference = make_model("fixed_zero", args)
        baselines = {"validation": evaluate(reference, cache, pairs["validation"], args, plans["validation"])}
        save_checkpoint(reference, args.output_dir / "tau_zero.pt", None, None, "fixed_zero")
        common.write_json(args.output_dir / "tau_zero_validation.json", baselines["validation"])
        count = len(ARMS) * len(args.seeds)
        log.message(f"결과: {args.output_dir}\n고정 tau=0 기준선 + 3구조 × {len(args.seeds)} seed = {count}회 독립 학습")
        log.message(f"모든 구조: tau bound ±{args.tau_bound:g}, T={args.temperature:g}, 동일 D loss·배치·선택 기준")
        log.message("ETA는 남은 최대 epoch 기준이며 조기 종료 시 줄어듭니다. 신뢰도·FE 학습/평가는 포함하지 않습니다.")
        partitions = json.loads((args.source_run / "splits.json").read_text())
        for split in common.SPLITS:
            log.message(f"{split}: 얼굴 {len(partitions['face_people'][split])}명 / 지문 참여자 "
                        f"{partitions['fingerprint_people'][split]} / 등록·질의 {len(pairs[split])}쌍")
        trials, schedules = [], {}
        for seed in args.seeds:
            states = initial_states(module, seed, args.temperature, args.tau_bound)
            for arm in ARMS:
                trials.append(train_arm(states[arm], cache, data, users, pairs["validation"], plans["validation"],
                                        baselines["validation"], args, settings, arm, seed,
                                        args.output_dir / f"{arm}_seed{seed}", count-len(trials)-1, schedules))
                common.write_json(args.output_dir / "training_summary.json", trials)
        log.message("\n모든 학습·검증 선택 완료. 선택된 각 모델의 시험 분할을 한 번씩 평가합니다.")
        for trial in trials:
            model = ArchitectureThresholdModel.load_checkpoint(Path(trial["checkpoint"]), args.device)
            trial["test"] = evaluate(model, cache, pairs["test"], args, plans["test"])
            directory = Path(trial["checkpoint"]).parent
            common.write_json(directory / "report.json", trial)
            common.write_json(directory / "status.json", {"state": "completed"})
            log.message(f"시험 완료: {trial['arm']} seed={trial['seed']} | {trial['selected_origin']} epoch={trial['best_epoch']}")
        baselines["test"] = evaluate(reference, cache, pairs["test"], args, plans["test"])
        data.assert_unchanged()
        for hashes in (source_hashes, code_hashes):
            if any(common.sha256(path) != digest for path, digest in hashes.items()):
                raise RuntimeError("Source experiment or code changed during this run")
        rows, aggregate, deltas = comparison_tables(trials, baselines)
        for name, table in (("comparison.csv", rows), ("aggregate.csv", aggregate), ("paired_deltas.csv", deltas)):
            common.write_csv(args.output_dir / name, list(table[0]), table)
        report = {"trials": trials, "tau_zero": baselines, "aggregate": aggregate, "paired_deltas": deltas,
                  "seconds": time.perf_counter() - started, "input_and_source_unchanged": True,
                  "shared_schedule_sha256": schedules,
                  "limitations": ["Architecture comparison using the previous D loss; no guarantee of improvement or FE key recovery.",
                                  "Actual fingerprint participants: 7. Validation/test each contain one person; their negatives are different fingers.",
                                  "Synthetic cross-modal correspondence. Seeds repeat initialization on the same participant split.",
                                  "Previously inspected test split is exploratory, not a new untouched benchmark.",
                                  "Parameter counts differ slightly; independent context layers also differ in initialization and connectivity.",
                                  "Soft occupancy penalties, hard-bit selection guards and shared optimizer/global clipping are unchanged.",
                                  "First-batch gradient cosines diagnose local training behavior, not proof of the cause of test errors.",
                                  "No reliability heads or FE operations. Load checkpoints only with ArchitectureThresholdModel."]}
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
