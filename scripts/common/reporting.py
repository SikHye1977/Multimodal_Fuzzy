"""Shared aggregation of threshold/reliability experiment results."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import MODALITIES


def reliability_tables(trials):
    rows = []
    for trial in trials:
        for split in ("validation", "test"):
            for m in MODALITIES:
                entry = trial[split]
                r = entry["reliability"]["modalities"][m]["query"]
                row = {"arm": trial["arm"], "seed": trial["seed"], "split": split, "modality": m,
                       "threshold_best_epoch": trial["threshold"]["best_epoch"],
                       "reliability_best_epoch": trial["reliability_training"]["best_epoch"],
                       **entry["bits"]["bits"][m]}
                row.update({f"query_{k}": v for k, v in r.items() if k != "calibration_bins"})
                row["tau_pair_mean_absolute_difference"] = entry["bits"]["tau"][m]["genuine_mean_absolute_difference"]
                rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"], row["split"], row["modality"]].append(row)
    metadata = {"arm", "seed", "split", "modality", "threshold_best_epoch", "reliability_best_epoch"}
    aggregates = []
    for (arm, split, m), items in groups.items():
        item = {"arm": arm, "split": split, "modality": m, "seed_count": len(items)}
        for key in rows[0]:
            if key in metadata:
                continue
            values = [r[key] for r in items if r[key] is not None]
            item[f"{key}_mean"] = float(np.mean(values)) if values else None
            item[f"{key}_std"] = float(np.std(values)) if values else None
        aggregates.append(item)
    indexed = {(r["arm"], r["seed"], r["split"], r["modality"]): r for r in rows}
    deltas = []
    for arm, reference in (("B", "A"), ("C", "B")):
        for row in rows:
            if row["arm"] != arm:
                continue
            base = indexed.get((reference, row["seed"], row["split"], row["modality"]))
            if base is None:
                continue
            item = {k: row[k] for k in ("seed", "split", "modality")}
            item["comparison"] = f"{arm}-{reference}"
            for key in ("genuine_hamming_bits", "within_radius_fraction", "query_bce", "query_brier",
                        "query_ece_10", "query_agreement_auroc", "query_error_average_precision"):
                item[f"{key}_delta"] = row[key] - base[key] if row[key] is not None and base[key] is not None else None
            deltas.append(item)
    return rows, aggregates, deltas
