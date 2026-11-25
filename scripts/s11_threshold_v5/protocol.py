"""Five-capture templates plus disjoint queries, within person-disjoint splits."""
from collections import Counter, defaultdict
import hashlib
import json

import numpy as np
import torch

from scripts.common.data import MODALITIES, SPLITS

MODEL_NAMES = {"face": "face", "fingerprint": "finger"}


def validate_population(data, users):
    for m in MODALITIES:
        partitions = defaultdict(set)
        assigned = set()
        for u in users:
            identity = u.face_person if m == "face" else u.finger
            person = u.face_person if m == "face" else u.fingerprint_person
            assigned.add(identity)
            partitions[person].add(u.split)
            ids = data.groups[m][identity]
            if len(ids) < 6 or len(ids) != len(set(ids)):
                raise ValueError(f"{m}/{identity}: need >=6 distinct captures")
        if assigned != set(data.groups[m]):
            raise ValueError(f"Unassigned identities in {m}")
        if any(len(s) != 1 for s in partitions.values()):
            raise ValueError(f"Person leakage across splits in {m}")
        x = data.arrays[m]
        if (x.ndim != 2 or x.shape[1] != 512 or x.dtype != np.float32
                or not np.isfinite(x).all()
                or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4)):
            raise ValueError(f"Expected finite normalized float32 embeddings: {m}")


def make_plan(data, users, split, seed, rounds=1, epoch=0):
    """One fixed five-capture registration per identity per round.

    All remaining captures are queries; none enter their own enrollment target
    or tau. Different rounds represent re-enrollment, not independent people.
    Finger templates are shared by virtual mappings of the same actual finger.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLITS.index(split), epoch]))
    chosen = [u for u in users if u.split == split]
    plan = {"split": split, "epoch": epoch, "rounds": rounds, "modalities": {}}
    maps = {}
    for m in MODALITIES:
        identities = sorted({u.face_person if m == "face" else u.finger for u in chosen})
        if len(identities) < 2:
            raise ValueError(f"Need >=2 {m} identities in {split}")
        maps[m] = {identity: i for i, identity in enumerate(identities)}
        records = []
        for r in range(rounds):
            for i, identity in enumerate(identities):
                order = rng.permutation(data.groups[m][identity]).tolist()
                records.append({"round": r, "identity_id": i,
                    "identity": identity if isinstance(identity, str) else list(identity),
                    "person": identity if m == "face" else identity[0],
                    "enrollment": sorted(order[:5]), "queries": sorted(order[5:])})
        plan["modalities"][m] = records
    rows = []
    for r in range(rounds):
        for u in chosen:
            rows.append([r*len(maps["face"])+maps["face"][u.face_person],
                         r*len(maps["fingerprint"])+maps["fingerprint"][u.finger]])
    plan["rows"] = rows
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def prepare_bank(data, plan, device):
    bank = {}
    for column, m in enumerate(MODALITIES):
        records = plan["modalities"][m]
        indices = np.asarray([r["enrollment"] for r in records])
        enroll = torch.from_numpy(data.arrays[m][indices]).to(device)
        if bool((enroll.mean(1).norm(dim=1) < 1e-8).any()):
            raise ValueError(f"Near-zero enrollment center in {m}")
        identity = torch.tensor([r["identity_id"] for r in records], device=device)
        # Equal weight per actual identity over a complete joint training epoch.
        row_ids = [records[row[column]]["identity_id"] for row in plan["rows"]]
        counts = Counter(row_ids)
        weights = torch.tensor([len(row_ids)/(len(counts)*counts[i]) for i in row_ids],
                               device=device, dtype=torch.float32)
        # Queries are loss/evaluation targets, NEVER inputs to batch_forward.
        # Padding is excluded before reducing any positive loss.
        query_indices = np.full((len(records), max(len(r["queries"]) for r in records)), -1,
                                dtype=np.int64)
        for i, r in enumerate(records):
            if set(r["enrollment"]) & set(r["queries"]) or not r["queries"]:
                raise ValueError("Enrollment/query overlap or empty query set")
            query_indices[i, :len(r["queries"])] = r["queries"]
        query_mask = query_indices >= 0
        queries = np.zeros((*query_indices.shape, 512), dtype=np.float32)
        queries[query_mask] = data.arrays[m][query_indices[query_mask]]
        # Each bank belongs to exactly one split. Negative donors are queries
        # from other identities in that split, not their own predicted tau.
        pools = defaultdict(set)
        for r in records:
            pools[r["identity_id"]].update(r["queries"])
        pool_indices, labels, group_members = [], [], []
        for i in sorted(pools):
            ids = sorted(pools[i])
            group_members.append(list(range(len(pool_indices), len(pool_indices)+len(ids))))
            pool_indices.extend(ids)
            labels.extend([i]*len(ids))
        padded = np.zeros((len(group_members), max(map(len, group_members))), dtype=np.int64)
        for i, members in enumerate(group_members):
            padded[i, :len(members)] = members
        bank[m] = {"enrollment": enroll, "identity": identity,
            "target": (enroll.mean(1) >= 0).float(), "weights": weights,
            "queries": torch.from_numpy(queries).to(device),
            "query_mask": torch.from_numpy(query_mask).to(device),
            "query_indices": query_indices,
            "pool_indices": np.asarray(pool_indices),
            "pool": torch.from_numpy(data.arrays[m][pool_indices]).to(device),
            "pool_labels": torch.tensor(labels, device=device),
            "members": torch.from_numpy(padded).to(device),
            "member_counts": torch.tensor(list(map(len, group_members)), device=device)}
    bank["rows"] = torch.tensor(plan["rows"], dtype=torch.long, device=device)
    return bank


def batch_forward(model, bank, positions):
    rows = bank["rows"][positions]
    return model(bank["face"]["enrollment"][rows[:, 0]],
                 bank["fingerprint"]["enrollment"][rows[:, 1]])


def describe_plan(plan):
    result = {}
    for m, records in plan["modalities"].items():
        result[m] = {"people": len({r["person"] for r in records}),
            "identities": len({r["identity_id"] for r in records}),
            "registrations": len(records), "query_comparisons": sum(len(r["queries"]) for r in records),
            "unique_enrollment_captures": len({i for r in records for i in r["enrollment"]}),
            "unique_query_captures": len({i for r in records for i in r["queries"]})}
    return result
