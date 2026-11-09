"""C1 result tables using the original suite's metric/averaging definitions."""
import numpy as np
from scripts.common.runtime import write_csv, write_json
from .protocol import MODEL_NAMES

FORMAT = "threshold-c1-standalone-v1"


def summary_rows(results):
    rows = []
    for r in results:
        for split, metrics in r["metrics"].items():
            for m in MODEL_NAMES:
                v = metrics[m]
                row = {k: r[k] for k in ("condition", "width", "seed", "checkpoint", "epoch", "parameter_count")}
                row.update(split=split, modality=m, genuine_count=v["genuine"]["count"],
                    genuine_macro_hamming=v["genuine"]["macro_mean"],
                    genuine_within_t=v["genuine"]["macro_within_t"],
                    radius_FRR=1-v["genuine"]["macro_within_t"],
                    impostor_hamming=v["impostor"]["all_different_identities"]["macro_mean"],
                    impostor_p05=v["impostor"]["all_different_identities"]["p05"],
                    radius_FAR=v["impostor"]["all_different_identities"]["macro_within_t"],
                    different_people_FAR=v["impostor"]["different_people"]["macro_within_t"],
                    auroc=v["auroc"], tau_abs_mean=v["tau_abs_mean"],
                    tau_saturated_fraction=v["tau_saturated_fraction"],
                    genuine_bch_exact=v.get("bch_genuine", {}).get("exact_target_rate"))
                for rule, d in v["tau_only"]["rules"].items():
                    row[rule+"_hamming"] = d["macro_mean"]
                    row[rule+"_within_t"] = d["macro_within_t"]
                    row[rule+"_count"] = d["count"]
                    row[rule+"_within_t_count"] = d["within_t_count"]
                    row[rule+"_bch_exact"] = d.get("bch", {}).get("exact_target_rate")
                rows.append(row)
    return rows


def record(condition, width, seed, checkpoint, epoch, parameters, directory, metrics):
    return dict(condition=condition, width=width, seed=seed, checkpoint=checkpoint, epoch=epoch,
                parameter_count=parameters, directory=directory, metrics=metrics)


def aggregate_table(path, rows, keys, fields):
    aggregates = []
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in keys) == group]
        item = dict(zip(keys, group))
        item["runs"] = len(selected)
        for field in fields:
            values = [r[field] for r in selected if r[field] is not None]
            item[field+"_mean"] = float(np.mean(values)) if values else None
            item[field+"_std"] = float(np.std(values)) if values else None
        aggregates.append(item)
    write_csv(path, list(aggregates[0]), aggregates)


def attack_rows(row, split, name, view, scores, attack_seed=None, attack_epoch=None):
    return [dict(condition=row["condition"], seed=row["seed"], checkpoint=row["checkpoint"],
                 threshold_epoch=row["epoch"], split=split, modality=m, attacker=name, attack_input=view,
                 attack_seed=attack_seed, attack_epoch=attack_epoch, hamming=scores[m]["macro_mean"],
                 within_t=scores[m]["macro_within_t"], exact_bits=scores[m]["exact_bits"],
                 exact_bch=scores[m].get("bch_exact_macro"), endpoint=row["id"]) for m in MODEL_NAMES]


def export(output, results, attack_results):
    rows = summary_rows(results)
    by_key = {(r["condition"], r["seed"], r["checkpoint"]): r for r in results}
    for item in rows:
        row = by_key[(item["condition"], item["seed"], item["checkpoint"])]
        item["public_tau_helper_bits"] = row["storage"][item["modality"]]["payload_bits"]
        selected = [a for a in attack_results if a["endpoint"] == row["id"] and a["split"] == item["split"] and a["modality"] == item["modality"]]
        item["strongest_evaluated_attack_within_t"] = max((a["within_t"] for a in selected), default=None)
    if rows:
        write_csv(output/"comparison.csv", list(rows[0]), rows)
        aggregate_table(output/"aggregate.csv", rows, ("condition", "checkpoint", "split", "modality"),
            ("genuine_macro_hamming", "genuine_within_t", "radius_FAR", "different_people_FAR", "impostor_p05",
             "strongest_evaluated_attack_within_t", "public_tau_helper_bits"))
    if attack_results:
        write_csv(output/"attacks.csv", list(attack_results[0]), attack_results)
        aggregate_table(output/"attack_aggregate.csv", attack_results,
            ("condition", "checkpoint", "split", "modality", "attacker", "attack_input"), ("hamming", "within_t", "exact_bch"))
    write_json(output/"report.json", {"format": FORMAT, "results": results, "attacks": attack_results,
        "interpretation": "C1 is fitted once on TRAIN; no neural threshold optimization or test selection. "
        "All model/attacker selection precedes test. Public-helper attacks include stored codes and BCH syndromes. "
        "Zero observed attack success is not a security proof. C1 uses estimated marginal Gaussian distributions, "
        "not certified zero leakage. Different payload budgets are explicitly reported. Current BCH restores anchor bits, not a complete FE key. "
        "Fingerprint test has one person; different-person FAR is null. The existing test set has been reused: exploratory results."})
