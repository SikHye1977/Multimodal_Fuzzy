#!/usr/bin/env python3
"""Joint Thresholding V3: four output designs on frozen cached embeddings.

No feature extraction, BCH/FE decoding, reliability fitting or Fusion training.
--check-only reads and validates inputs without creating outputs or training.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
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
from scripts import train_thresholding_reliability_abc as previous
from scripts import train_thresholding_abcd as contrastive
from scripts.train_reliability_distance import roc_auc

common = previous.common
MODALITIES = common.MODALITIES
MODEL_PATH = ROOT / "Thresholding Model V3.py"
spec = importlib.util.spec_from_file_location("thresholding_model_v3", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model_module
spec.loader.exec_module(model_module)
ThresholdingModelV3 = model_module.ThresholdingModelV3
MODES = (*model_module.MODES, "independent_512")
PRIMARY = "mean_identity_macro_within_radius_fraction"


def make_model(args, mode, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if mode == "independent_512":
            from Thresholding_Model_512 import ThresholdingModel512, Thresholding512Config
            model = ThresholdingModel512(Thresholding512Config(mode="dynamic", temperature=args.temperature,
                tau_base_bound=args.tau_base_bound, tau_delta_bound=args.tau_delta_bound,
                hidden_dim=args.independent_hidden_dim))
        else:
            model = ThresholdingModelV3(model_module.ThresholdingV3Config(mode=mode,
                hidden_dim=args.hidden_dim, context_dim=args.context_dim, temperature=args.temperature,
                tau_base_bound=args.tau_base_bound, tau_delta_bound=args.tau_delta_bound))
    return model.to(args.device)


def forward_model(model, face, fingerprint):
    if isinstance(model, ThresholdingModelV3):
        return model(face, fingerprint)
    return {f"{m}_{k}": v for m, x in zip(MODALITIES, (face, fingerprint))
            for k, v in model.quantize(m, x).items()}


def forward_pair(model, cache, rows):
    ef, ep, qf, qp = common.model_inputs(cache, rows)
    return forward_model(model, ef, ep), forward_model(model, qf, qp)


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
    weights = contrastive.balanced_weights(labels)
    return ((1-hard_mix)*ordinary + hard_mix*(F.relu(margin-hard)/margin).square().mean(1)).dot(weights)


def loss_function(model, e, q, rows, users, args):
    labels = contrastive.identity_codes(rows, users, e["face_soft_bits"].device)
    terms, losses = defaultdict(list), {}
    for m in MODALITIES:
        eb, qb = (ste(out[m+"_soft_bits"], out[m+"_bits"]) for out in (e, q))
        weights = contrastive.balanced_weights(labels[m])
        d = hamming(eb, qb)
        positive = ((F.relu(d-(args.bch_t-args.safety_margin))/args.bch_t).square()*weights).sum()
        negative = negative_loss(hamming(eb, qb, matrix=True), labels[m],
            getattr(args, m+"_margin_bits"), args.hard_k, args.hard_mix)
        balance = sum(((weights[:, None]*b).sum(0)-.5).square().mean() for b in (eb, qb))/2
        row = sum(contrastive.row_penalty(b, weights, args.row_lower) for b in (eb, qb))/2
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


def unique_indices(values):
    return np.sort(np.unique(values, axis=0, return_index=True)[1])


def evaluation_plan(rows, users):
    # Preserve different companion contexts even when one image pair repeats.
    rows = rows[unique_indices(rows[:, :4])]
    labels = {m: x.numpy() for m, x in contrastive.identity_codes(rows, users, "cpu").items()}
    people = {"face": np.array([users[int(i)].face_person for i in rows[:, common.USER]]),
              "fingerprint": np.array([users[int(i)].fingerprint_person for i in rows[:, common.USER]])}
    anchors = unique_indices(rows[:, [common.EF, common.EP]])
    queries = unique_indices(rows[:, [common.QF, common.QP]])
    weights = {m: contrastive.balanced_weights(torch.from_numpy(labels[m])).numpy() for m in MODALITIES}
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


@torch.no_grad()
def collect(model, cache, rows, args, on_batch=None):
    model.eval()
    collected = ({}, {})
    for step, start in enumerate(range(0, len(rows), args.batch_size), 1):
        e, q = forward_pair(model, cache, rows[start:start+args.batch_size])
        for dest, out in zip(collected, (e, q)):
            for m in MODALITIES:
                for key in ("tau", "bits"):
                    name = m+"_"+key
                    dest.setdefault(name, []).append(out[name].cpu())
        if on_batch:
            on_batch(step)
    return tuple({key: torch.cat(parts) for key, parts in dest.items()} for dest in collected)


def distance_stats(values, t):
    return {"pair_count": int(len(values)), "mean_hamming_bits": float(np.mean(values)) if len(values) else None,
            "p05_hamming_bits": float(np.quantile(values, .05)) if len(values) else None,
            "within_radius_count": int((values <= t).sum()),
            "FAR_h": float((values <= t).mean()) if len(values) else None}


def summarize(e, q, plan, args, tau_bound):
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
        obs_weights = contrastive.balanced_weights(torch.from_numpy(obs_labels))
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


@torch.no_grad()
def swap_diagnostics(model, cache, plan, e, q, args, on_batch=None):
    rows, result, step = plan["rows"], {}, 0
    for changed in MODALITIES:
        other = "fingerprint" if changed == "face" else "face"
        swapped, distances = [], {m: [] for m in MODALITIES}
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            donor = rows[plan["swaps"][changed][start:start+len(batch)]]
            fi = donor[:, common.QF] if changed == "face" else batch[:, common.QF]
            pi = donor[:, common.QP] if changed == "fingerprint" else batch[:, common.QP]
            inputs = [cache[m][torch.as_tensor(idx, device=args.device)] for m, idx in zip(MODALITIES, (fi, pi))]
            out = forward_model(model, *inputs)
            for m in MODALITIES:
                b = out[m+"_bits"].cpu()
                distances[m].append((b != e[m+"_bits"][start:start+len(batch)]).sum(1).numpy())
            swapped.append((out[other+"_bits"].cpu() != q[other+"_bits"][start:start+len(batch)]).sum(1).numpy())
            step += 1
            if on_batch:
                on_batch(step)
        distances = {m: np.concatenate(parts) for m, parts in distances.items()}
        wrong, kept = distances[changed], distances[other]
        weights = plan["weights"][changed].astype(np.float64)
        weights /= weights.sum()
        result["wrong_"+changed] = {
            "context_pair_count": len(wrong), "changed_modality": changed,
            "changed_mean_hamming_bits": float(wrong.mean()),
            "changed_FAR_h": float((wrong <= args.bch_t).mean()),
            "changed_identity_macro_FAR_h": float((wrong <= args.bch_t)@weights),
            "kept_modality": other, "kept_within_radius_fraction": float((kept <= args.bch_t).mean()),
            "kept_bits_changed_vs_original_query": float(np.concatenate(swapped).mean()),
            "both_within_radius_fraction": float(((wrong <= args.bch_t) & (kept <= args.bch_t)).mean())}
    return result


@torch.no_grad()
def evaluate(model, cache, plan, args, on_batch=None):
    e, q = collect(model, cache, plan["rows"], args, on_batch)
    result = summarize(e, q, plan, args, model.config.tau_bound)
    batches = math.ceil(len(plan["rows"])/args.batch_size)
    result["swaps"] = swap_diagnostics(model, cache, plan, e, q, args,
        (lambda k: on_batch(batches+k)) if on_batch else None)
    return result, e, q


def selection_guard(candidate, baseline, args):
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


def better(candidate, best, args):
    if abs(candidate[PRIMARY]-best[PRIMARY]) > 1e-12:
        return candidate[PRIMARY] > best[PRIMARY]
    return candidate["mean_identity_macro_hamming_bits"] < best["mean_identity_macro_hamming_bits"]-args.min_bit_gain


@torch.no_grad()
def initial_digest(model, cache, rows):
    digest = hashlib.sha256()
    ef, ep, qf, qp = common.model_inputs(cache, rows)
    for out, inputs in zip(forward_pair(model, cache, rows), ((ef, ep), (qf, qp))):
        for m, x in zip(MODALITIES, inputs):
            tau, bits = out[m+"_tau"], out[m+"_bits"]
            if bool((tau != 0).any()) or not torch.equal(bits, (x >= 0).to(torch.uint8)):
                raise RuntimeError("All configurations must start at tau=0")
            digest.update(tau.expand(-1, 512).cpu().numpy().tobytes())
            digest.update(bits.cpu().numpy().tobytes())
    return digest.hexdigest()


def train_trial(model, mode, seed, data, users, cache, plan, baseline, args, folder, schedules, future):
    log = common.Log(folder)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    best, best_epoch, stale = baseline, 0, 0
    metadata = {"mode": mode, "seed": seed, "selection_metric": PRIMARY, "bch_t": args.bch_t,
                "requires_new_enrollment": True, "BCH_decoding": False, "reliability_fitted": False}
    model.save_checkpoint(folder/"best_threshold.pt", {**metadata, "epoch": 0, "origin": "zero_fallback"})
    log.record({"epoch": 0, "validation": baseline, "selected": True})
    progress = common.Progress(f"V3/{mode} seed={seed}", args.epochs, future, args.log_interval)
    for epoch in range(1, args.epochs+1):
        model.train()
        batches, padding = previous.make_schedule(data, users, args, seed, epoch, "threshold", schedules)
        steps = len(batches) + 3*math.ceil(len(plan["rows"])/args.batch_size)
        sums, count = defaultdict(float), 0
        for step, (rows, _) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            e, q = forward_pair(model, cache, rows)
            losses = loss_function(model, e, q, rows, users, args)
            if not all(bool(torch.isfinite(v)) for v in losses.values()):
                raise RuntimeError("Nonfinite loss")
            losses["total"].backward()
            grad = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            for key, val in losses.items():
                sums[key] += float(val.detach())*len(rows)
            sums["gradient_norm"] += float(grad)*len(rows)
            count += len(rows)
            progress.update(epoch, step, steps, "train", sums["total"]/count)
        metrics, _, _ = evaluate(model, cache, plan, args,
            lambda k: progress.update(epoch, len(batches)+k, steps, "validation+swap", sums["total"]/count))
        eligible, reasons = selection_guard(metrics, baseline, args)
        selected = eligible and better(metrics, best, args)
        if selected:
            best, best_epoch, stale = metrics, epoch, 0
            model.save_checkpoint(folder/"best_threshold.pt", {**metadata, "epoch": epoch, "origin": "trained"})
        else:
            stale += 1
        model.save_checkpoint(folder/"last_threshold.pt", {**metadata, "epoch": epoch, "origin": "last_epoch"})
        progress.finish_line()
        log.record({"epoch": epoch, "train_loss": {k: v/count for k, v in sums.items()}, "padding": padding,
                    "validation": metrics, "eligible": eligible, "guard_reasons": reasons, "selected": selected})
        message = [f"{mode} seed={seed} epoch={epoch}"]
        for m in MODALITIES:
            v = metrics["modalities"][m]
            message.append(f"{m}: {v['identity_macro_hamming_bits']:.2f} bit / <= {args.bch_t}: {100*v['identity_macro_within_radius_fraction']:.2f}%")
        log.message(" | ".join(message)+f" | 선택 epoch={best_epoch} | guard={'통과' if eligible else ', '.join(reasons)}")
        if stale >= args.patience:
            break
    return {"best_epoch": best_epoch, "epochs_run": epoch, "validation": best,
            "origin": "trained" if best_epoch else "zero_fallback"}


def export_details(folder, split, plan, data, e, q, args):
    rows = plan["rows"]
    arrays = {f"{role}_{k}": v.numpy() for role, out in (("enrollment", e), ("query", q)) for k, v in out.items()}
    arrays["pair_rows"] = rows
    records = []
    for i, row in enumerate(rows):
        record = {"context_index": i, "virtual_user_index": int(row[common.USER])}
        for name, m, col in (("enrollment_face", "face", common.EF), ("query_face", "face", common.QF),
                             ("enrollment_fingerprint", "fingerprint", common.EP), ("query_fingerprint", "fingerprint", common.QP)):
            record[name] = data.files[m][int(row[col])]
        for m in MODALITIES:
            record[m+"_hamming_bits"] = int((e[m+"_bits"][i] != q[m+"_bits"][i]).sum())
            record[m+"_within_radius"] = int(record[m+"_hamming_bits"] <= args.bch_t)
            record[m+"_tau_mean_absolute_difference"] = float((e[m+"_tau"][i]-q[m+"_tau"][i]).abs().mean())
        records.append(record)
    for m in MODALITIES:
        arrays[m+"_hamming_bits"] = np.count_nonzero(arrays["enrollment_"+m+"_bits"] != arrays["query_"+m+"_bits"], axis=1)
        arrays[m+"_within_radius"] = arrays[m+"_hamming_bits"] <= args.bch_t
        arrays["wrong_"+m+"_donor_row"] = plan["swaps"][m]
    np.savez_compressed(folder/f"{split}_bits_tau.npz", **arrays)
    common.write_csv(folder/f"{split}_context_pairs.csv", list(records[0]), records)


def write_tables(folder, trials, baselines):
    rows = []
    for trial in trials:
        for split in ("validation", "test"):
            for m in MODALITIES:
                values, zero = trial[split]["modalities"][m], baselines[split]["modalities"][m]
                row = {"mode": trial["mode"], "seed": trial["seed"], "split": split, "modality": m,
                       "best_epoch": trial["training"]["best_epoch"], "origin": trial["training"]["origin"]}
                row.update({k: v for k, v in values.items() if k != "negative_scopes"})
                for scope in ("all_different_identities", "different_people", "same_person_different_fingers"):
                    for key in ("pair_count", "mean_hamming_bits", "p05_hamming_bits", "FAR_h"):
                        row[scope+"_"+key] = values["negative_scopes"].get(scope, {}).get(key)
                row["within_radius_percent"] = 100*values["identity_macro_within_radius_fraction"]
                row["within_radius_delta_vs_zero_pp"] = 100*(values["identity_macro_within_radius_fraction"]-zero["identity_macro_within_radius_fraction"])
                row["hamming_delta_vs_zero_bits"] = values["identity_macro_hamming_bits"]-zero["identity_macro_hamming_bits"]
                row["wrong_modality_FAR_h"] = trial[split]["swaps"]["wrong_"+m]["changed_identity_macro_FAR_h"]
                rows.append(row)
    common.write_csv(folder/"comparison.csv", list(rows[0]), rows)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("mode", "split", "modality"))].append(row)
    aggregates = []
    keys = [k for k in rows[0] if k not in ("mode", "split", "modality", "seed", "origin")]
    for group, entries in groups.items():
        item = dict(zip(("mode", "split", "modality"), group))
        item["seed_count"] = len(entries)
        for key in keys:
            numbers = [entry[key] for entry in entries if entry[key] is not None]
            item[key+"_mean"] = float(np.mean(numbers)) if numbers else None
            item[key+"_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        aggregates.append(item)
    common.write_csv(folder/"aggregate.csv", list(aggregates[0]), aggregates)
    return aggregates


def load_checkpoint(path, mode, device):
    if mode == "independent_512":
        from Thresholding_Model_512 import ThresholdingModel512
        return ThresholdingModel512.load_checkpoint(path, device)
    return ThresholdingModelV3.load_checkpoint(path, device)


def run(args, data, users, pairs, source, source_hashes, *, synthetic=False):
    started = time.perf_counter()
    args.data_seed = int(source["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log, trials, schedules, initial_hashes, current = common.Log(args.output_dir), [], {}, {}, None
    try:
        common.write_json(args.output_dir/"status.json", {"state": "running", "synthetic_test_only": synthetic})
        if not synthetic:
            for name in previous.MANIFESTS:
                shutil.copy2(args.source_run/name, args.output_dir/name)
        paths = [Path(__file__), MODEL_PATH, Path(previous.__file__), Path(common.__file__),
                 Path(contrastive.__file__), ROOT/"scripts/train_reliability_distance.py"]
        if "independent_512" in args.modes:
            paths += [ROOT/"Thresholding_Model_512.py", ROOT/"thresholding/variability_model.py"]
        code_hashes = {str(p): common.sha256(p) for p in paths}
        config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "torch": str(torch.__version__),
            "synthetic_test_only": synthetic, "input_sha256": data.hashes, "source_sha256": source_hashes,
            "code_sha256": code_hashes, "BCH_decoding": False, "FE_key_evaluation": False,
            "test_used_for_selection": False, "reliability_trained": False, "requires_new_enrollment": True,
            "primary_metric": PRIMARY, "success_definition": "pre-decoder Hamming <= bch_t",
            "evaluation_unit": "unique joint enrollment/query context; primary metric equal identity weight per modality",
            "loss": "identity-balanced C + mined negatives + coordinate balance + row occupancy + bounded residual penalty; no tau agreement loss",
            "selection": "validation radius rate, then Hamming; FAR/swap/ROC/collapse guards; no mean-impostor-distance guard",
            "resume_supported": False}
        common.write_json(args.output_dir/"config.json", config)
        cache = {m: torch.tensor(x, device=args.device) for m, x in data.arrays.items()}
        plans = {s: evaluation_plan(pairs[s], users) for s in ("validation", "test")}
        zero = make_model(args, "shared_scalar", args.seeds[0])
        baseline, e, q = evaluate(zero, cache, plans["validation"], args)
        baselines = {"validation": baseline}
        common.write_json(args.output_dir/"zero_validation.json", baseline)
        export_details(args.output_dir, "zero_validation", plans["validation"], data, e, q, args)
        jobs = [(seed, mode) for seed in args.seeds for mode in args.modes]
        log.message(f"결과: {args.output_dir}\nV3 {len(jobs)}회 순차 실행 | <= {args.bch_t} bits | 학습 목표 {args.bch_t-args.safety_margin} bits")
        log.message(f"공통 초기 tau=0, 동일 시드 배치 일치 | device={args.device} | 학습 쌍 {len(pairs['train'])}")
        log.message("ETA는 최대 epoch 기준 추정이며 조기 종료·최종 평가에 따라 달라집니다.")
        for j, (seed, mode) in enumerate(jobs):
            current = args.output_dir/f"{mode}_seed{seed}"
            current.mkdir()
            common.write_json(current/"status.json", {"state": "training"})
            model = make_model(args, mode, seed)
            digest = initial_digest(model, cache, pairs["train"][:args.batch_size])
            if str(seed) in initial_hashes and initial_hashes[str(seed)] != digest:
                raise RuntimeError("Initial outputs differ across configurations")
            initial_hashes[str(seed)] = digest
            training = train_trial(model, mode, seed, data, users, cache, plans["validation"], baseline,
                args, current, schedules, (len(jobs)-j-1)*args.epochs)
            trial = {"mode": mode, "seed": seed, "training": training, "checkpoint": str(current/"best_threshold.pt"),
                "initial_outputs_sha256": digest, "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "model_config": asdict(model.config)}
            trials.append(trial)
            common.write_json(current/"training_summary.json", trial)
            common.write_json(current/"status.json", {"state": "trained_awaiting_test"})
            common.write_json(args.output_dir/"training_summary.json", trials)
            common.write_json(args.output_dir/"shared_schedule_sha256.json", schedules)
            log.message(f"학습 완료 {j+1}/{len(jobs)} | {mode} | 선택 epoch={training['best_epoch']}")
        log.message("모든 체크포인트 선택 완료. 시험 분할과 단일 모달리티 교체 검사를 수행합니다.")
        baselines["test"], e, q = evaluate(zero, cache, plans["test"], args)
        common.write_json(args.output_dir/"zero_test.json", baselines["test"])
        export_details(args.output_dir, "zero_test", plans["test"], data, e, q, args)
        for j, trial in enumerate(trials):
            current = Path(trial["checkpoint"]).parent
            model, _ = load_checkpoint(trial["checkpoint"], trial["mode"], args.device)
            progress = common.Progress(f"최종평가 {trial['mode']} seed={trial['seed']}", 2, 2*(len(trials)-j-1), args.log_interval)
            for epoch, split in enumerate(("validation", "test"), 1):
                steps = 3*math.ceil(len(plans[split]["rows"])/args.batch_size)
                trial[split], e, q = evaluate(model, cache, plans[split], args,
                    lambda k: progress.update(epoch, k, steps, split, None))
                export_details(current, split, plans[split], data, e, q, args)
            progress.finish_line()
            if trial["validation"] != trial["training"]["validation"]:
                raise RuntimeError("Reloaded checkpoint does not reproduce validation metrics")
            common.write_json(current/"report.json", trial)
            common.write_json(current/"status.json", {"state": "completed"})
        aggregates = write_tables(args.output_dir, trials, baselines)
        data.assert_unchanged()
        for hashes in (source_hashes, code_hashes):
            if any(common.sha256(Path(p)) != digest for p, digest in hashes.items()):
                raise RuntimeError("Source or code changed during the run")
        report = {"trials": trials, "baselines": baselines, "aggregate": aggregates,
            "seconds": time.perf_counter()-started, "synthetic_test_only": synthetic,
            "input_and_source_unchanged": True, "shared_schedule_sha256": schedules,
            "initial_outputs_sha256": initial_hashes,
            "limitations": ["Within-radius and FAR_h are distance proxies, not BCH/FE/key decoding results.",
                "Virtual cross-dataset correspondence, no natural same-person face/fingerprint relationship.",
                "Fingerprint participants: 7 total, original split 5/1/1; different-person FAR unavailable on one-person splits.",
                "Context pairs are dependent. Seed repetitions do not increase the participant count.",
                "Primary identity-macro metrics differ from earlier image-pair-deduplicated averages.",
                "Test data was inspected in earlier experiments; this is exploratory evidence.",
                "Mean Hamming <=32 does not imply every pair is within radius. New model requires new enrollment."]}
        common.write_json(args.output_dir/"report.json", report)
        common.write_json(args.output_dir/"status.json", {"state": "completed", "seconds": report["seconds"], "synthetic_test_only": synthetic})
        log.message(f"완료 | {common.duration(report['seconds'])} | {args.output_dir/'comparison.csv'}")
        return report
    except BaseException as error:
        status = {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)}
        common.write_json(args.output_dir/"status.json", status)
        if current is not None:
            common.write_json(current/"status.json", status)
        (args.output_dir/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-run", type=Path,
        default=ROOT/"thresholding/results/train_20261029_162319_486882")
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory only; defaults to a timestamped V3 directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(model_module.MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--check-only", action="store_true", help="Validate existing inputs without training or creating outputs")
    integers = {"epochs": 100, "patience": 20, "batch_size": 64, "hidden_dim": 128,
                "context_dim": 64, "independent_hidden_dim": 32, "threads": 2,
                "log_interval": 10, "bch_t": 32, "safety_margin": 4, "hard_k": 4}
    floats = {"learning_rate": 1e-4, "temperature": .02, "tau_base_bound": .01,
              "tau_delta_bound": .01, "grad_clip": 5., "face_margin_bits": 224.,
              "fingerprint_margin_bits": 160., "negative_weight": 1., "balance_weight": 1.,
              "row_weight": 1., "residual_weight": .001, "row_lower": .3, "hard_mix": .5,
              "min_bit_gain": .1, "max_far_h_increase": 0., "max_radius_drop": 0.,
              "max_auroc_drop": .001, "max_balance_increase": .01,
              "max_constant_increase": .02, "max_row_outside_increase": .01}
    for name, default in integers.items():
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=default)
    for name, default in floats.items():
        parser.add_argument("--"+name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args(argv)
    for name in integers:
        if getattr(args, name) < (0 if name == "safety_margin" else 1):
            parser.error(f"Invalid {name}")
    if args.batch_size < 2 or not 1 <= args.bch_t <= 255 or args.safety_margin >= args.bch_t:
        parser.error("Need batch_size>=2, 1<=bch_t<=255, and 0<=safety_margin<bch_t")
    for name in floats:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    for name in ("learning_rate", "temperature", "tau_base_bound", "tau_delta_bound", "grad_clip"):
        if getattr(args, name) == 0:
            parser.error(f"{name} must be positive")
    if not 0 < args.row_lower < .5 or not 0 <= args.hard_mix <= 1:
        parser.error("Need 0<row_lower<0.5 and 0<=hard_mix<=1")
    if not all(args.bch_t < getattr(args, m+"_margin_bits") <= 512 for m in MODALITIES):
        parser.error("Negative margins must lie in (bch_t,512]")
    if any(getattr(args, k) > 1 for k in floats if k.startswith("max_")):
        parser.error("Fraction tolerances cannot exceed 1")
    if len(args.modes) != len(set(args.modes)) or len(args.seeds) != len(set(args.seeds)):
        parser.error("Modes and seeds must not repeat")
    if not all(0 <= s < 2**32 for s in args.seeds):
        parser.error("Seeds must be in [0,2**32)")
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT/"thresholding/results"/datetime.now().strftime("v3_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT/name for name in ("muct-master", "UareU", "feature_extractors", "BCH", "scripts")]
    protected.append(args.source_run)
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    if any(args.output_dir.is_relative_to(p.resolve()) for p in protected):
        parser.error("Output must be separate from source data, code, embeddings and source experiment")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory")
    if any((p/"config.json").exists() and (p/"status.json").exists() for p in args.output_dir.parents):
        parser.error("Do not place a new experiment inside an existing experiment")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data, users, pairs, source, hashes = previous.prepare_source(args)
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output must not be inside the embedding directory")
    if args.check_only:
        print(json.dumps({"input_validation": "passed", "training_started": False,
            "outputs_created": False, "source_run": str(args.source_run), "modes": args.modes,
            "seeds": args.seeds, "trial_count": len(args.modes)*len(args.seeds),
            "pair_counts": {s: len(p) for s, p in pairs.items()},
            "participants": {s: {"face": len({u.face_person for u in users if u.split == s}),
                "fingerprint": len({u.fingerprint_person for u in users if u.split == s})} for s in common.SPLITS},
            "cuda_available": torch.cuda.is_available()}, ensure_ascii=False, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use feature_extractors/run.sh or explicitly pass --device cpu")
    run(args, data, users, pairs, source, hashes)


if __name__ == "__main__":
    main()

