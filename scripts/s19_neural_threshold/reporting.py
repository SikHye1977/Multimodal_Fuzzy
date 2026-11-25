"""Neural ablation reports, keeping best/last and privacy diagnostics distinct."""
from scripts.common.runtime import write_csv, write_json
from scripts.s16_quantile_thresholding.reporting import summary_rows, aggregate_table, attack_rows

FORMAT = "threshold-neural-architecture-objective-v1"


def export(output, results, attack_results):
    rows = summary_rows(results)
    mapping = {(r["condition"], r["seed"], r["checkpoint"]): r for r in results}
    for row in rows:
        endpoint = mapping[row["condition"], row["seed"], row["checkpoint"]]
        row.update(architecture=endpoint.get("architecture", "control"),
                   objective=endpoint.get("objective", "control"),
                   selected_origin=endpoint.get("selected_origin", "control"),
                   public_tau_bits=endpoint["storage"][row["modality"]]["payload_bits"])
        scores = [a["within_t"] for a in attack_results if a["endpoint"] == endpoint["id"]
                  and a["split"] == row["split"] and a["modality"] == row["modality"]]
        row["strongest_evaluated_attack_within_t"] = max(scores, default=None)
    if rows:
        write_csv(output/"comparison.csv", list(rows[0]), rows)
        aggregate_table(output/"aggregate.csv", rows, ("condition", "checkpoint", "split", "modality"),
            ("genuine_macro_hamming", "genuine_within_t", "radius_FAR", "different_people_FAR",
             "genuine_bch_exact", "strongest_evaluated_attack_within_t"))
    if attack_results:
        write_csv(output/"attacks.csv", list(attack_results[0]), attack_results)
        aggregate_table(output/"attack_aggregate.csv", attack_results,
            ("condition", "checkpoint", "split", "modality", "attacker", "attack_input"),
            ("hamming", "within_t", "exact_bch"))
    write_json(output/"report.json", {"format": FORMAT, "results": results, "attacks": attack_results,
        "interpretation": "A/B use V5-D loss throughout; C/D switch after matched warm-up. "
        "Warm-up teachers are frozen per architecture/seed, never fitted per enrolled user. "
        "All thresholds and independent attackers are selected before test. Best selection includes fixed "
        "public-tau guards; last is an unselected diagnostic. Learned attacks are not security proofs, and "
        "feeding a syndrome to an MLP is not an algebraic attack. BCH restores anchor bits, not FE keys. "
        "Face and fingerprint are independent heads, virtual pairing is not real paired biometric data. "
        "The reused test split is exploratory; fingerprint different-person FAR may be unavailable."})
