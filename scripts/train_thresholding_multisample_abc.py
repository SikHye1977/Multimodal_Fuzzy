#!/usr/bin/env python3
"""Matched A/B/C comparison: single capture, pooled captures, learned set tau.

Defaults: 5 enrollment captures and 1 query capture per modality. Only C trains.
Uses frozen caches, original person splits and virtual mappings. No extraction,
BCH/FE decoding, or Fusion training. --check-only creates no output or optimizer.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Thresholding_Model_Set import SetThresholdConfig, SetThresholdingModel, set_statistics
from scripts import train_thresholding_v3 as v3

common, previous = v3.common, v3.previous
MODALITIES, PRIMARY = v3.MODALITIES, v3.PRIMARY
DESCRIPTIONS = {
    "A": "One designated capture from each matched enrollment/query set, tau=0",
    "B": "L2-normalized mean of all captures in each set, tau=0",
    "C": "Same representative as B; jointly predicted separate 512-D taus from set statistics",
}


def validate_population(data, users, args):
    for m in MODALITIES:
        partitions = defaultdict(set)
        for u in users:
            identity = u.face_person if m == "face" else u.finger
            person = u.face_person if m == "face" else u.fingerprint_person
            partitions[person].add(u.split)
            indices = data.groups[m][identity]
            if len(set(indices)) != len(indices):
                raise ValueError(f"Duplicate capture indices for {m}/{identity}")
            if len(indices) < args.enroll_captures+args.query_captures:
                raise ValueError(f"{m}/{identity}: {len(indices)} captures cannot provide disjoint "
                                 f"{args.enroll_captures}+{args.query_captures} sets")
        if any(len(splits) != 1 for splits in partitions.values()):
            raise ValueError(f"Person leakage across splits: {m}")
        for split in common.SPLITS:
            identities = {u.face_person if m == "face" else u.finger for u in users if u.split == split}
            if len(identities) < 2:
                raise ValueError(f"Need two distinct {m} identities in {split}")


def make_protocol(data, users, args, split, epoch=0):
    """Independent enrollment episodes; no query enters its enrollment statistics.

Each round visits every capture as a query at least once per virtual user.
Across episodes roles can rotate; within an episode the original capture IDs
are strictly disjoint. A's designated capture is the first member of each set.
"""
    rng = np.random.default_rng(np.random.SeedSequence([args.data_seed, common.SPLITS.index(split), epoch]))
    members, lookups = {m: [] for m in MODALITIES}, {m: {} for m in MODALITIES}

    def index(m, values):
        # Preserve A's designated member; canonicalize the other members.
        key = (int(values[0]), *sorted(int(x) for x in values[1:]))
        if key not in lookups[m]:
            lookups[m][key] = len(members[m])
            members[m].append(key)
        return lookups[m][key]

    rows = []
    rounds = args.train_rounds if split == "train" else args.evaluation_rounds
    for ui, user in enumerate(users):
        if user.split != split:
            continue
        groups = {"face": data.groups["face"][user.face_person],
                  "fingerprint": data.groups["fingerprint"][user.finger]}
        for _ in range(rounds):
            orders = {m: rng.permutation(groups[m]) for m in MODALITIES}
            for j in range(max(map(len, orders.values()))):
                row = []
                for m in MODALITIES:
                    order = orders[m]
                    query = [int(order[(j+k) % len(order)]) for k in range(args.query_captures)]
                    candidates = [int(i) for i in order if i not in query]
                    enrollment = rng.choice(candidates, args.enroll_captures, replace=False).tolist()
                    if set(enrollment) & set(query):
                        raise RuntimeError("Enrollment/query capture overlap")
                    row.extend((index(m, enrollment), index(m, query)))
                rows.append((*row, ui))
    rows = np.asarray(rows, dtype=np.int64)
    plan = v3.evaluation_plan(rows, users)
    plan.update(split=split, members=members, epoch=epoch)
    coverage = {}
    for m, columns in (("face", (common.EF, common.QF)), ("fingerprint", (common.EP, common.QP))):
        identities = {u.face_person if m == "face" else u.finger for u in users if u.split == split}
        required = {i for identity in identities for i in data.groups[m][identity]}
        observed = {i for sid in plan["rows"][:, columns].ravel() for i in members[m][sid]}
        if required != observed:
            raise RuntimeError(f"Not all {split}/{m} captures are covered")
        coverage[m] = len(observed)
    plan["capture_coverage"] = coverage
    digest = hashlib.sha256(plan["rows"].astype("<i8").tobytes())
    digest.update(json.dumps(members, sort_keys=True).encode())
    plan["sha256"] = digest.hexdigest()
    return plan


@torch.no_grad()
def build_bank(data, plan, device):
    """Cache deterministic statistics, not trainable features or identity labels."""
    bank = {}
    for m in MODALITIES:
        members = plan["members"][m]
        source = torch.from_numpy(data.arrays[m])
        statistics = torch.empty(len(members), 1026, dtype=source.dtype)
        by_size = defaultdict(list)
        for i, values in enumerate(members):
            by_size[len(values)].append(i)
        for positions in by_size.values():
            for start in range(0, len(positions), 256):
                idx = positions[start:start+256]
                captures = torch.tensor([members[i] for i in idx])
                statistics[idx] = set_statistics(source[captures])
        bank[m] = {"statistics": statistics.to(device),
                   "single": source[[values[0] for values in members]].to(device)}
    return bank


def forward_ids(model, arm, bank, face_ids, fingerprint_ids):
    stats, singles = [], []
    for m, ids in zip(MODALITIES, (face_ids, fingerprint_ids)):
        idx = torch.as_tensor(ids, dtype=torch.long, device=bank[m]["statistics"].device)
        stats.append(bank[m]["statistics"][idx])
        singles.append(bank[m]["single"][idx])
    if arm == "C":
        return model.forward_statistics(*stats)
    if arm not in ("A", "B"):
        raise ValueError(f"Unknown arm: {arm}")
    result = {}
    for m, st, single in zip(MODALITIES, stats, singles):
        x = single if arm == "A" else st[:, :512]
        result[m+"_bits"] = (x >= 0).to(torch.uint8)
        result[m+"_tau"] = torch.zeros_like(x)
    return result


def forward_pair(model, arm, bank, rows):
    return (forward_ids(model, arm, bank, rows[:, common.EF], rows[:, common.EP]),
            forward_ids(model, arm, bank, rows[:, common.QF], rows[:, common.QP]))


@torch.no_grad()
def evaluate(model, arm, bank, plan, args, on_batch=None):
    if model is not None:
        model.eval()
    rows, collected, step = plan["rows"], ({}, {}), 0
    for start in range(0, len(rows), args.batch_size):
        outputs = forward_pair(model, arm, bank, rows[start:start+args.batch_size])
        for dest, out in zip(collected, outputs):
            for m in MODALITIES:
                for key in ("tau", "bits"):
                    name = m+"_"+key
                    dest.setdefault(name, []).append(out[name].cpu())
        step += 1
        if on_batch:
            on_batch(step)
    e, q = ({k: torch.cat(v) for k, v in out.items()} for out in collected)
    result = v3.summarize(e, q, plan, args, args.tau_base_bound+args.tau_delta_bound)
    # IDs here refer to sample sets, not individual images as in the V3 trainer.
    for metrics in result["modalities"].values():
        for key in list(metrics):
            if key.startswith("image_pair_"):
                metrics["set_pair_"+key[len("image_pair_"):]] = metrics.pop(key)
    result["swaps"] = {}
    for changed in MODALITIES:
        other = "fingerprint" if changed == "face" else "face"
        changes, distances = [], {m: [] for m in MODALITIES}
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            donor = rows[plan["swaps"][changed][start:start+len(batch)]]
            fi = donor[:, common.QF] if changed == "face" else batch[:, common.QF]
            pi = donor[:, common.QP] if changed == "fingerprint" else batch[:, common.QP]
            out = forward_ids(model, arm, bank, fi, pi)
            for m in MODALITIES:
                distances[m].append((out[m+"_bits"].cpu() != e[m+"_bits"][start:start+len(batch)]).sum(1).numpy())
            changes.append((out[other+"_bits"].cpu() != q[other+"_bits"][start:start+len(batch)]).sum(1).numpy())
            step += 1
            if on_batch:
                on_batch(step)
        d = {m: np.concatenate(v) for m, v in distances.items()}
        weights = plan["weights"][changed].astype(np.float64)
        weights /= weights.sum()
        wrong, kept = d[changed], d[other]
        result["swaps"]["wrong_"+changed] = {
            "context_pair_count": len(rows), "changed_modality": changed,
            "changed_mean_hamming_bits": float(wrong.mean()),
            "changed_FAR_h": float((wrong <= args.bch_t).mean()),
            "changed_identity_macro_FAR_h": float((wrong <= args.bch_t)@weights),
            "kept_modality": other, "kept_within_radius_fraction": float((kept <= args.bch_t).mean()),
            "kept_bits_changed_vs_original_query": float(np.concatenate(changes).mean()),
            "both_within_radius_fraction": float(((wrong <= args.bch_t) & (kept <= args.bch_t)).mean())}
    return result, e, q


def make_model(args, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = SetThresholdConfig(**{k: getattr(args, k) for k in SetThresholdConfig.__dataclass_fields__})
        model = SetThresholdingModel(config)
    return model.to(args.device)


def save_protocol(folder, split, plan, data):
    common.write_json(folder/f"{split}_sets.json", {"sha256": plan["sha256"],
        "members": plan["members"], "rows": plan["rows"].tolist(),
        "row_columns": ["enrollment_face_set", "query_face_set", "enrollment_fingerprint_set",
                        "query_fingerprint_set", "virtual_user_index"],
        "files": data.files, "capture_coverage": plan["capture_coverage"],
        "first_member": "A uses member 0; B/C aggregate all members", "independent_enrollment_episodes": True})


def export_details(folder, split, arm, plan, data, e, q, args):
    arrays = {f"{role}_{k}": v.numpy() for role, out in (("enrollment", e), ("query", q)) for k, v in out.items()}
    arrays["set_pair_rows"] = plan["rows"]
    records = []
    for i, row in enumerate(plan["rows"]):
        record = {"episode_index": i, "virtual_user_index": int(row[common.USER])}
        for name, m, col in (("enrollment_face", "face", common.EF), ("query_face", "face", common.QF),
                            ("enrollment_fingerprint", "fingerprint", common.EP), ("query_fingerprint", "fingerprint", common.QP)):
            indices = plan["members"][m][row[col]]
            used = indices[:1] if arm == "A" else indices
            record[name+"_files"] = json.dumps([data.files[m][j] for j in used], ensure_ascii=False)
        for m in MODALITIES:
            record[m+"_hamming_bits"] = int((e[m+"_bits"][i] != q[m+"_bits"][i]).sum())
            record[m+"_within_radius"] = int(record[m+"_hamming_bits"] <= args.bch_t)
        records.append(record)
    for m in MODALITIES:
        arrays[m+"_hamming_bits"] = np.count_nonzero(arrays["enrollment_"+m+"_bits"] != arrays["query_"+m+"_bits"], axis=1)
        arrays[m+"_within_radius"] = arrays[m+"_hamming_bits"] <= args.bch_t
        arrays["wrong_"+m+"_donor_row"] = plan["swaps"][m]
    np.savez_compressed(folder/f"{split}_bits_tau.npz", **arrays)
    common.write_csv(folder/f"{split}_episodes.csv", list(records[0]), records)


def train_trial(model, seed, data, users, validation_plan, validation_bank, baseline, args, folder, future):
    log = common.Log(folder)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best, best_epoch, stale = baseline, 0, 0
    metadata = {"arm": "C", "seed": seed, "selection_reference": "B",
                "enroll_captures": args.enroll_captures, "query_captures": args.query_captures,
                "requires_new_enrollment": True, "BCH_decoding": False}
    model.save_checkpoint(folder/"best_threshold.pt", {**metadata, "epoch": 0, "origin": "B_zero_fallback"})
    log.record({"epoch": 0, "validation": baseline, "selected": True})
    progress = common.Progress(f"C/set-tau seed={seed}", args.epochs, future, args.log_interval)
    schedules = {}
    for epoch in range(1, args.epochs+1):
        plan = make_protocol(data, users, args, "train", epoch)
        bank = build_bank(data, plan, args.device)
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 928]))
        batches, padding = common.training_batches(plan["rows"], users, args.batch_size, rng)
        schedules[str(epoch)] = {"sets_sha256": plan["sha256"], "batch_sha256":
            hashlib.sha256(b"".join(batch.astype("<i8").tobytes() for batch, _ in batches)).hexdigest()}
        model.train()
        total_steps = len(batches)+3*math.ceil(len(validation_plan["rows"])/args.batch_size)
        sums, count = defaultdict(float), 0
        for step, (rows, _) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            e, q = forward_pair(model, "C", bank, rows)
            losses = v3.loss_function(model, e, q, rows, users, args)
            if not all(bool(torch.isfinite(v)) for v in losses.values()):
                raise RuntimeError("Nonfinite loss")
            losses["total"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            count += len(rows)
            for key, value in losses.items():
                sums[key] += float(value.detach())*len(rows)
            sums["gradient_norm"] += float(gradient)*len(rows)
            progress.update(epoch, step, total_steps, "train", sums["total"]/count)
        metrics, _, _ = evaluate(model, "C", validation_bank, validation_plan, args,
            lambda k: progress.update(epoch, len(batches)+k, total_steps, "validation+swap", sums["total"]/count))
        eligible, reasons = v3.selection_guard(metrics, baseline, args)
        selected = eligible and v3.better(metrics, best, args)
        if selected:
            best, best_epoch, stale = metrics, epoch, 0
            model.save_checkpoint(folder/"best_threshold.pt", {**metadata, "epoch": epoch, "origin": "trained"})
        else:
            stale += 1
        model.save_checkpoint(folder/"last_threshold.pt", {**metadata, "epoch": epoch, "origin": "last_epoch"})
        progress.finish_line()
        log.record({"epoch": epoch, "train_loss": {k: v/count for k, v in sums.items()},
                    "validation": metrics, "eligible": eligible, "guard_reasons": reasons,
                    "selected": selected, "padding": padding, "schedule": schedules[str(epoch)]})
        values = [f"C seed={seed} epoch={epoch}"]
        for m in MODALITIES:
            v = metrics["modalities"][m]
            values.append(f"{m}: {v['identity_macro_hamming_bits']:.2f} bit / <={args.bch_t}: "
                          f"{100*v['identity_macro_within_radius_fraction']:.2f}%")
        log.message(" | ".join(values)+f" | 선택 epoch={best_epoch} | guard={'통과' if eligible else ', '.join(reasons)}")
        if stale >= args.patience:
            break
    return {"best_epoch": best_epoch, "epochs_run": epoch, "validation": best,
            "origin": "trained" if best_epoch else "B_zero_fallback", "schedules": schedules}


def write_tables(folder, trials):
    rows = []
    baselines = {t["arm"]: t for t in trials if t["arm"] in ("A", "B")}
    for trial in trials:
        for split in ("validation", "test"):
            for m in MODALITIES:
                v = trial[split]["modalities"][m]
                row = {"arm": trial["arm"], "seed": trial["seed"], "split": split, "modality": m,
                       "best_epoch": trial["training"]["best_epoch"], "origin": trial["training"]["origin"]}
                row.update({k: value for k, value in v.items() if k != "negative_scopes"})
                for scope in ("all_different_identities", "different_people", "same_person_different_fingers"):
                    for key in ("pair_count", "mean_hamming_bits", "p05_hamming_bits", "FAR_h"):
                        row[scope+"_"+key] = v["negative_scopes"].get(scope, {}).get(key)
                for baseline_arm in ("A", "B"):
                    b = baselines[baseline_arm][split]["modalities"][m]
                    row[f"within_radius_delta_vs_{baseline_arm}_pp"] = 100*(v["identity_macro_within_radius_fraction"]-b["identity_macro_within_radius_fraction"])
                    row[f"hamming_delta_vs_{baseline_arm}_bits"] = v["identity_macro_hamming_bits"]-b["identity_macro_hamming_bits"]
                row["wrong_modality_FAR_h"] = trial[split]["swaps"]["wrong_"+m]["changed_identity_macro_FAR_h"]
                row["both_within_radius_fraction"] = trial[split]["both_within_radius_fraction"]
                rows.append(row)
    common.write_csv(folder/"comparison.csv", list(rows[0]), rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["split"], row["modality"])].append(row)
    keys = [k for k in rows[0] if k not in ("arm", "seed", "split", "modality", "origin")]
    aggregates = []
    for group, items in grouped.items():
        result = dict(zip(("arm", "split", "modality"), group))
        result["run_count"] = len(items)
        for key in keys:
            values = [item[key] for item in items if item[key] is not None]
            result[key+"_mean"] = float(np.mean(values)) if values else None
            result[key+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregates.append(result)
    common.write_csv(folder/"aggregate.csv", list(aggregates[0]), aggregates)
    deltas = []
    for row in rows:
        if row["arm"] == "A":
            continue
        ref = "A" if row["arm"] == "B" else "B"
        deltas.append({"comparison": row["arm"]+"-"+ref, "seed": row["seed"], "split": row["split"],
                       "modality": row["modality"],
                       "within_radius_delta_pp": row[f"within_radius_delta_vs_{ref}_pp"],
                       "hamming_delta_bits": row[f"hamming_delta_vs_{ref}_bits"]})
    common.write_csv(folder/"paired_deltas.csv", list(deltas[0]), deltas)
    return aggregates


def run(args, data, users, source_hashes, *, synthetic=False):
    started = time.perf_counter()
    validate_population(data, users, args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log, current = common.Log(args.output_dir), None
    try:
        common.write_json(args.output_dir/"status.json", {"state": "running", "synthetic_test_only": synthetic})
        paths = [Path(__file__), ROOT/"Thresholding_Model_Set.py", Path(v3.__file__), Path(common.__file__),
                 Path(previous.__file__), Path(v3.contrastive.__file__), ROOT/"Thresholding Model V3.py",
                 ROOT/"scripts/train_reliability_distance.py"]
        code_hashes = {str(p): common.sha256(p) for p in paths}
        if not synthetic:
            original = args.output_dir/"source_manifests"
            original.mkdir()
            for name in previous.MANIFESTS:
                shutil.copy2(args.source_run/name, original/name)
        config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "torch": str(torch.__version__),
                  "synthetic_test_only": synthetic, "input_sha256": data.hashes, "source_sha256": source_hashes,
                  "code_sha256": code_hashes, "arms": DESCRIPTIONS, "BCH_decoding": False,
                  "FE_key_evaluation": False, "requires_new_enrollment": True, "test_used_for_selection": False,
                  "primary_metric": PRIMARY, "model_config": asdict(make_model(args, args.seeds[0]).config),
                  "protocol": "Disjoint enrollment/query sets within each episode; independent simulated enrollments across episodes",
                  "A_sampling": "First, uniformly sampled enrollment member; first query member. No quality selection.",
                  "selection_reference": "B", "baseline_seeds": "A and B deterministic; evaluated once, not replicated per seed",
                  "train_sampling": "New episodes per epoch, shared data_seed; model seed only changes initialization and batch order",
                  "resume_supported": False}
        common.write_json(args.output_dir/"config.json", config)
        log.message(f"결과: {args.output_dir}\n등록 {args.enroll_captures}장 / 질의 {args.query_captures}장 | "
                    f"A/B 기준선 + C 학습 {len(args.seeds)}회 | <= {args.bch_t} bits")
        log.message("ETA는 최대 epoch 기준 추정이며 조기 종료와 최종 평가 시간에 따라 달라집니다.")
        validation = make_protocol(data, users, args, "validation")
        validation_bank = build_bank(data, validation, args.device)
        save_protocol(args.output_dir, "validation", validation, data)
        save_protocol(args.output_dir, "train_epoch1", make_protocol(data, users, args, "train", 1), data)
        trials = []
        for arm in ("A", "B"):
            folder = args.output_dir/arm
            folder.mkdir()
            progress = common.Progress(f"{arm} 기준선", 1, 0, args.log_interval)
            steps = 3*math.ceil(len(validation["rows"])/args.batch_size)
            metrics, e, q = evaluate(None, arm, validation_bank, validation, args,
                                    lambda k: progress.update(1, k, steps, "validation", None))
            progress.finish_line()
            export_details(folder, "validation", arm, validation, data, e, q, args)
            trials.append({"arm": arm, "seed": None, "training": {"best_epoch": 0, "epochs_run": 0,
                           "origin": "not_trainable"}, "validation": metrics, "folder": str(folder)})
            common.write_json(folder/"validation.json", metrics)
        baseline = trials[1]["validation"]
        for j, seed in enumerate(args.seeds):
            current = args.output_dir/f"C_seed{seed}"
            current.mkdir()
            common.write_json(current/"status.json", {"state": "training"})
            model = make_model(args, seed)
            # C starts exactly at the matched B zero-threshold baseline.
            initial, _, _ = evaluate(model, "C", validation_bank, validation, args)
            if initial != baseline:
                raise RuntimeError("Initial C metrics must equal B exactly")
            training = train_trial(model, seed, data, users, validation, validation_bank, baseline, args,
                                   current, (len(args.seeds)-j-1)*args.epochs)
            trial = {"arm": "C", "seed": seed, "training": training, "folder": str(current),
                     "checkpoint": str(current/"best_threshold.pt"),
                     "parameter_count": sum(p.numel() for p in model.parameters())}
            trials.append(trial)
            common.write_json(current/"training_summary.json", trial)
            common.write_json(current/"status.json", {"state": "trained_awaiting_test"})
            common.write_json(args.output_dir/"training_summary.json", trials)
            log.message(f"C 학습 {j+1}/{len(args.seeds)} 완료 | seed={seed} | 선택 epoch={training['best_epoch']}")
        log.message("모든 체크포인트 선택 완료. 시험 분할을 평가합니다.")
        test = make_protocol(data, users, args, "test")
        test_bank = build_bank(data, test, args.device)
        save_protocol(args.output_dir, "test", test, data)
        for j, trial in enumerate(trials):
            current = Path(trial["folder"])
            model = SetThresholdingModel.load_checkpoint(trial["checkpoint"], args.device)[0] if trial["arm"] == "C" else None
            splits = ("validation", "test") if model is not None else ("test",)
            progress = common.Progress(f"최종평가 {trial['arm']} seed={trial['seed']}", len(splits), 0, args.log_interval)
            for i, split in enumerate(splits, 1):
                plan, bank = (validation, validation_bank) if split == "validation" else (test, test_bank)
                steps = 3*math.ceil(len(plan["rows"])/args.batch_size)
                metrics, e, q = evaluate(model, trial["arm"], bank, plan, args,
                                        lambda k: progress.update(i, k, steps, split, None))
                trial[split] = metrics
                export_details(current, split, trial["arm"], plan, data, e, q, args)
            progress.finish_line()
            if model is not None and trial["validation"] != trial["training"]["validation"]:
                raise RuntimeError("Checkpoint reload changed validation metrics")
            common.write_json(current/"report.json", trial)
            common.write_json(current/"status.json", {"state": "completed"})
        aggregates = write_tables(args.output_dir, trials)
        data.assert_unchanged()
        for hashes in (source_hashes, code_hashes):
            if any(common.sha256(Path(p)) != digest for p, digest in hashes.items()):
                raise RuntimeError("Source or code changed during the experiment")
        report = {"trials": trials, "aggregate": aggregates, "seconds": time.perf_counter()-started,
            "synthetic_test_only": synthetic, "input_and_source_unchanged": True,
            "protocol_sha256": {"validation": validation["sha256"], "test": test["sha256"]},
            "limitations": ["Within-radius success and FAR_h are distance proxies, not BCH/FE decoding results.",
                "Face/fingerprint association is virtual, not real same-person multimodal data.",
                "Seven fingerprint participants, original 5/1/1 split; different-person FAR unavailable on held-out splits.",
                "Episodes reuse captures and are correlated; seeds and synthetic pairings do not add participants.",
                "A/B/C use a new set sampling protocol; compare within this run, not directly against old V3 pairs.",
                "Query singleton std is unobserved; its zero placeholder has an explicit missingness indicator.",
                "Test participants have been used in earlier experiments; exploratory rather than fresh confirmatory evidence.",
                "Changing quantization/aggregation requires new enrollment bits and FE helper data."]}
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-run", type=Path, default=ROOT/"thresholding/results/train_20261029_162319_486882")
    p.add_argument("--embedding-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--check-only", action="store_true")
    ints = {"enroll_captures": 5, "query_captures": 1, "train_rounds": 1, "evaluation_rounds": 2,
            "epochs": 100, "patience": 20, "batch_size": 64, "hidden_dim": 128, "context_dim": 64,
            "threads": 2, "log_interval": 10, "bch_t": 32, "safety_margin": 4, "hard_k": 4}
    floats = {"learning_rate": 1e-4, "temperature": .02, "tau_base_bound": .01,
              "tau_delta_bound": .01, "grad_clip": 5., "face_margin_bits": 224.,
              "fingerprint_margin_bits": 160., "negative_weight": 1., "balance_weight": 1.,
              "row_weight": 1., "residual_weight": .001, "row_lower": .3, "hard_mix": .5,
              "min_bit_gain": .1, "max_far_h_increase": 0., "max_radius_drop": 0.,
              "max_auroc_drop": .001, "max_balance_increase": .01,
              "max_constant_increase": .02, "max_row_outside_increase": .01}
    for key, value in ints.items():
        p.add_argument("--"+key.replace("_", "-"), type=int, default=value)
    for key, value in floats.items():
        p.add_argument("--"+key.replace("_", "-"), type=float, default=value)
    args = p.parse_args(argv)
    for key in ints:
        if getattr(args, key) < (0 if key == "safety_margin" else 1):
            p.error(f"Invalid {key}")
    if args.enroll_captures+args.query_captures > 8:
        p.error("UareU has 8 captures/finger; enrollment+query must be <=8 without overlap")
    if args.batch_size < 2 or not 1 <= args.bch_t <= 255 or args.safety_margin >= args.bch_t:
        p.error("Need batch_size>=2, 1<=bch_t<=255 and 0<=safety_margin<bch_t")
    for key in floats:
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            p.error(f"{key} must be finite and nonnegative")
    for key in ("learning_rate", "temperature", "tau_base_bound", "tau_delta_bound", "grad_clip"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if not 0 < args.row_lower < .5 or not 0 <= args.hard_mix <= 1:
        p.error("Need 0<row_lower<0.5 and 0<=hard_mix<=1")
    if not all(args.bch_t < getattr(args, m+"_margin_bits") <= 512 for m in MODALITIES):
        p.error("Negative margins must be in (bch_t,512]")
    if any(getattr(args, key) > 1 for key in floats if key.startswith("max_")):
        p.error("Fraction tolerances cannot exceed 1")
    if len(args.seeds) != len(set(args.seeds)) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        p.error("Seeds must be distinct integers in [0,2**32)")
    args.source_run = args.source_run.expanduser().resolve()
    if args.embedding_dir:
        args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                       ROOT/"thresholding/results"/datetime.now().strftime("multisample_abc_%Y%m%d_%H%M%S_%f"))
    protected = [ROOT/name for name in ("muct-master", "UareU", "feature_extractors", "BCH", "scripts")]
    protected.append(args.source_run)
    if args.embedding_dir:
        protected.append(args.embedding_dir)
    if any(args.output_dir.is_relative_to(path.resolve()) for path in protected):
        p.error("Output must be separate from input data, code and source experiment")
    if args.output_dir.exists():
        p.error("Output directory already exists")
    if any((parent/"config.json").exists() and (parent/"status.json").exists() for parent in args.output_dir.parents):
        p.error("Do not nest an experiment inside an existing experiment")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data, users, _, source, source_hashes = previous.prepare_source(args)
    args.data_seed = int(source["seed"])
    if args.output_dir.is_relative_to(data.directory.resolve()):
        raise ValueError("Output must not be inside the embedding directory")
    validate_population(data, users, args)
    if args.check_only:
        protocols = {s: make_protocol(data, users, args, s, epoch=1 if s == "train" else 0) for s in common.SPLITS}
        print(json.dumps({"input_validation": "passed", "training_started": False, "outputs_created": False,
            "enroll_captures": args.enroll_captures, "query_captures": args.query_captures,
            "trainable_trials": len(args.seeds), "baseline_arms": ["A", "B"],
            "episodes": {s: len(plan["rows"]) for s, plan in protocols.items()},
            "capture_coverage": {s: plan["capture_coverage"] for s, plan in protocols.items()},
            "participants": {s: {"face": len({u.face_person for u in users if u.split == s}),
                "fingerprint": len({u.fingerprint_person for u in users if u.split == s})} for s in common.SPLITS},
            "cuda_available": torch.cuda.is_available()}, ensure_ascii=False, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use feature_extractors/run.sh or explicitly select --device cpu")
    run(args, data, users, source_hashes)


if __name__ == "__main__":
    main()
