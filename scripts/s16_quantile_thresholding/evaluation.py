"""C1 hard-bit evaluation, preserved from the original S14-S16 experiment.

Independent of V5/V6 models and training entry points.
"""
from collections import defaultdict
import numpy as np
import torch
from scripts.common.metrics import roc_auc
from scripts.common.runtime import write_csv, write_json
from .protocol import MODEL_NAMES, batch_forward


def bound_error_lower_bound(query, target, bound):
    """Optimistic bound, NOT achievable recovery performance.

    Give each query its own arbitrary 512-D tau in the closed [-bound,bound]
    box, ignore impostors, and count bits even that oracle cannot correct.
    The real enrollment-only model has additional constraints. At +bound,
    a value is always one because binarization uses >=.
    """
    return (((target == 1) & (query < -bound)) |
            ((target == 0) & (query >= bound))).sum(-1)


def distance_summary(distances, anchors, donors, t):
    """Macro averaging: donor identity within claimant, then claimant identities."""
    d = np.asarray(distances, dtype=np.float64)
    empty = {"count": 0, "mean": None, "std": None, "p05": None, "p95": None,
             "within_t": None, "macro_mean": None, "macro_within_t": None}
    if not len(d):
        return empty
    pairs = defaultdict(list)
    for index, (a, b) in enumerate(zip(anchors, donors)):
        pairs[int(a), int(b)].append(index)
    people = defaultdict(list)
    for (a, _), indices in pairs.items():
        people[a].append((d[indices].mean(), (d[indices] <= t).mean()))
    macro = np.mean([np.mean(values, axis=0) for values in people.values()], axis=0)
    return {"count": len(d), "mean": float(d.mean()), "std": float(d.std()),
            "p05": float(np.quantile(d, .05)), "p95": float(np.quantile(d, .95)),
            "within_t": float((d <= t).mean()), "macro_mean": float(macro[0]),
            "macro_within_t": float(macro[1])}


@torch.no_grad()
def collect_tau(model, bank, batch_size):
    result = {m: np.zeros((len(bank[m]["enrollment"]), 512), np.float32) for m in MODEL_NAMES}
    if model is None:
        return result
    model.eval()
    seen = {m: set() for m in MODEL_NAMES}
    for start in range(0, len(bank["rows"]), batch_size):
        positions = torch.arange(start, min(start+batch_size, len(bank["rows"])),
                                 device=bank["rows"].device)
        outputs = batch_forward(model, bank, positions)
        rows = bank["rows"][positions].cpu().numpy()
        for column, (m, name) in enumerate(MODEL_NAMES.items()):
            values = outputs[name+"_tau"].cpu().numpy()
            for record, value in zip(rows[:, column], values):
                if record in seen[m] and not np.allclose(result[m][record], value, atol=1e-7):
                    raise RuntimeError("A shared finger acquired different tau values across virtual users")
                result[m][record] = value
                seen[m].add(int(record))
    for m in MODEL_NAMES:
        if len(seen[m]) != len(result[m]) or not np.isfinite(result[m]).all():
            raise RuntimeError(f"Missing or non-finite registration tau: {m}")
    return result


def evaluate(model, data, plan, bank, args, *, directory=None, bch=None, progress=None):
    taus = collect_tau(model, bank, args.batch_size)
    metrics, saved, comparison_rows = {}, {}, []
    completed = 0
    total = sum(len(v) for v in plan["modalities"].values())
    for m in MODEL_NAMES:
        records = plan["modalities"][m]
        target = bank[m]["target"].cpu().numpy().astype(np.uint8)
        enrollment = bank[m]["enrollment"].cpu().numpy()
        genuine, negatives, enrollment_distances, bound_errors = [], [], [], []
        query_bits, query_indices, template_indices = [], [], []
        # Queries belong to their donor's held-out set in this registration round.
        by_round = defaultdict(list)
        for record in records:
            by_round[record["round"]].extend(
                (q, record["identity_id"], record["person"]) for q in record["queries"])
        for template, record in enumerate(records):
            tau, anchor = taus[m][template], target[template]
            eid, person = record["identity_id"], record["person"]
            fit_distance = ((enrollment[template] >= tau) != anchor).sum(-1)
            enrollment_distances.extend((int(d), eid, eid) for d in fit_distance)
            ids = record["queries"]
            bits = (data.arrays[m][ids] >= tau).astype(np.uint8)
            distances = (bits != anchor).sum(-1)
            lower = bound_error_lower_bound(data.arrays[m][ids], anchor, args.tau_bound)
            bound_errors.extend((int(d), eid, eid) for d in lower)
            genuine.extend((int(d), eid, eid) for d in distances)
            query_bits.extend(bits)
            query_indices.extend(ids)
            template_indices.extend([template]*len(ids))
            for q, d in zip(ids, distances):
                comparison_rows.append({"modality": m, "kind": "genuine", "round": record["round"],
                    "template": template, "claimant_identity": eid, "donor_identity": eid,
                    "query_index": q, "query_file": str(data.files[m][q]), "hamming": int(d),
                    "within_t": int(d <= args.bch_t)})
            other = [(q, i, p) for q, i, p in by_round[record["round"]] if i != eid]
            for start in range(0, len(other), args.distance_chunk):
                chunk = other[start:start+args.distance_chunk]
                ids = [v[0] for v in chunk]
                dists = ((data.arrays[m][ids] >= tau) != anchor).sum(-1)
                for (q, donor, donor_person), distance in zip(chunk, dists):
                    category = "same_person_different_fingers" if person == donor_person else "different_people"
                    negatives.append((int(distance), eid, donor, category))
                    if directory is not None:
                        comparison_rows.append({"modality": m, "kind": category, "round": record["round"],
                            "template": template, "claimant_identity": eid, "donor_identity": donor,
                            "query_index": q, "query_file": str(data.files[m][q]), "hamming": int(distance),
                            "within_t": int(distance <= args.bch_t)})
            completed += 1
            if progress:
                progress(completed, total)
        def summarize(rows):
            return distance_summary([r[0] for r in rows], [r[1] for r in rows],
                                    [r[2] for r in rows], args.bch_t)
        neg = {"all_different_identities": summarize(negatives)}
        for category in ("different_people", "same_person_different_fingers"):
            if m == "fingerprint" or category == "different_people":
                neg[category] = summarize([r for r in negatives if r[3] == category])
        all_d = [r[0] for r in genuine] + [r[0] for r in negatives]
        labels = np.asarray([1]*len(genuine) + [0]*len(negatives))
        qbits = np.asarray(query_bits, np.uint8)
        ti = np.asarray(template_indices, np.int64)
        # Equal weight per actual identity when diagnosing coordinate occupancy.
        identity_occupancy = []
        for identity in sorted({r["identity_id"] for r in records}):
            positions = np.asarray([records[i]["identity_id"] == identity for i in ti])
            identity_occupancy.append(qbits[positions].mean(0))
        occupancy = np.mean(identity_occupancy, axis=0)
        metrics[m] = {"genuine": summarize(genuine), "enrollment": summarize(enrollment_distances),
            "impostor": neg, "auroc": roc_auc(labels, -np.asarray(all_d, np.float64)),
            "query_one_fraction": float(occupancy.mean()),
            "constant_coordinate_fraction": float(((occupancy == 0) | (occupancy == 1)).mean()),
            "tau_abs_mean": float(np.abs(taus[m]).mean()),
            "tau_abs_max": float(np.abs(taus[m]).max()),
            "tau_saturated_fraction": float((np.abs(taus[m]) >= .95*args.tau_bound).mean()),
            "tau_across_registrations_std_mean": float(taus[m].std(axis=0).mean())}
        lower_summary = summarize(bound_errors)
        metrics[m]["bound_diagnostic"] = {
            "optimistic_error_lower_bound": lower_summary,
            "macro_impossible_within_t": 1-lower_summary["macro_within_t"],
            "note": "Per-query arbitrary vector tau oracle, ignores impostors and enrollment-only prediction. "
                    "A lower bound <=t does not imply that any deployable model succeeds."}
        if bch is not None:
            recovered = bch.recover(target[ti], qbits)
            metrics[m]["bch_genuine"] = {"count": len(recovered),
                "exact_target_rate": float((recovered == 1).mean()),
                "no_candidate_rate": float((recovered == 0).mean()),
                "wrong_candidate_rate": float((recovered == 2).mean())}
            saved[m+"_bch_outcomes"] = recovered
        saved.update({m+"_tau": taus[m], m+"_target_bits": target,
            m+"_enrollment_indices": np.asarray([r["enrollment"] for r in records]),
            m+"_query_bits": qbits, m+"_query_indices": np.asarray(query_indices),
            m+"_query_template_indices": ti,
            m+"_bound_error_lower_bound": np.asarray([r[0] for r in bound_errors], np.int64)})
    metrics["mean_macro_within_t"] = float(np.mean([metrics[m]["genuine"]["macro_within_t"] for m in MODEL_NAMES]))
    metrics["mean_macro_hamming"] = float(np.mean([metrics[m]["genuine"]["macro_mean"] for m in MODEL_NAMES]))
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory/"metrics.json", metrics)
        np.savez_compressed(directory/"bits_tau.npz", **saved)
        write_csv(directory/"pairs.csv", list(comparison_rows[0]), comparison_rows)
    return metrics
