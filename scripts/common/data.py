"""Frozen embeddings, participant splits and enrollment/query pairs."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import torch

from .runtime import sha256
SPLITS = ("train", "validation", "test")
MODALITIES = ("face", "fingerprint")
EF, QF, EP, QP, USER = range(5)
MANIFESTS = ("splits.json", "virtual_users.csv", "samples.csv", "train_pairs.csv", "validation_pairs.csv", "test_pairs.csv")


@dataclass(frozen=True)
class VirtualUser:
    split: str
    face_person: str
    fingerprint_person: str
    finger_id: str

    @property
    def finger(self):
        return self.fingerprint_person, self.finger_id


class Embeddings:
    """Read-only archives; validate completeness rather than silently dropping files."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.arrays, self.files, self.groups, self.cameras, self.hashes = {}, {}, {}, {}, {}
        for modality in MODALITIES:
            path = self.directory / f"{modality}_embeddings.npz"
            self.hashes[str(path.resolve())] = sha256(path)
            with np.load(path, allow_pickle=False) as archive:
                x = archive["embeddings"].copy()
                files = [str(name) for name in archive["files"].tolist()]
            if x.shape != (len(files), 512) or x.dtype != np.float32:
                raise ValueError(f"{modality}: expected float32 (N,512) embeddings")
            if len(set(files)) != len(files) or not np.isfinite(x).all():
                raise ValueError(f"{modality}: duplicate files or nonfinite embeddings")
            if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4):
                raise ValueError(f"{modality}: embeddings must already be L2 normalized")
            groups = defaultdict(list)
            for row, filename in enumerate(files):
                stem = Path(filename).stem
                if modality == "face":
                    # Keep the suffix opaque: unusual glasses annotations are valid.
                    match = re.fullmatch(r"i(\d+)([a-z])([a-e])-(.+)", stem)
                    if not match:
                        raise ValueError(f"Expected original MUCT image, not mirror/augmentation: {filename}")
                    identity = match[1]
                    self.cameras[row] = match[3]
                else:
                    match = re.fullmatch(r"(\d+)_(\d+)_(\d+)", stem)
                    if not match:
                        raise ValueError(f"Unrecognized fingerprint filename: {filename}")
                    identity = (match[1], match[2])
                groups[identity].append(row)
            for identity in groups:
                groups[identity].sort(key=lambda row: files[row])
            self.arrays[modality], self.files[modality], self.groups[modality] = x, files, dict(groups)
        if len(self.files["face"]) != 3755 or len(self.groups["face"]) != 276:
            raise ValueError("This experiment requires all 3,755 original MUCT images / 276 people")
        fingers = self.groups["fingerprint"]
        counts = Counter(person for person, finger in fingers)
        expected = {person: (5 if person == "057" else 10)
                    for person in ("012", "013", "017", "022", "027", "057", "076")}
        if len(self.files["fingerprint"]) != 520 or dict(counts) != expected:
            raise ValueError("This experiment requires all 520 UareU images / 65 fingers / 7 people")
        for finger, rows in fingers.items():
            captures = {int(Path(self.files["fingerprint"][row]).stem.split("_")[2]) for row in rows}
            if len(rows) != 8 or captures != set(range(1, 9)):
                raise ValueError(f"Incomplete or duplicate fingerprint captures: {finger}")

    def assert_unchanged(self):
        if any(sha256(path) != expected for path, expected in self.hashes.items()):
            raise RuntimeError("Input embedding archives changed during the run")


def make_users(data, seed):
    """Split real people BEFORE creating balanced, fixed synthetic associations."""
    rng = np.random.default_rng(seed)
    faces = sorted(data.groups["face"])
    people = sorted({person for person, finger in data.groups["fingerprint"]})
    rng.shuffle(faces)
    rng.shuffle(people)
    face_splits = dict(zip(SPLITS, (faces[:220], faces[220:248], faces[248:])))
    fingerprint_splits = dict(zip(SPLITS, (people[:5], people[5:6], people[6:])))
    users = []
    for split in SPLITS:
        fingers = sorted(finger for finger in data.groups["fingerprint"]
                         if finger[0] in fingerprint_splits[split])
        rng.shuffle(fingers)
        for i, person in enumerate(face_splits[split]):
            fingerprint_person, finger_id = fingers[i % len(fingers)]
            users.append(VirtualUser(split, person, fingerprint_person, finger_id))
        assigned = {user.finger for user in users if user.split == split}
        if assigned != set(fingers):
            raise RuntimeError("Every finger must be represented by at least one virtual user")
    # All faces occur once; all fingers of a real person remain in one split.
    if len({user.face_person for user in users}) != 276:
        raise RuntimeError("Face identities crossed partitions or were lost")
    person_partitions = defaultdict(set)
    for user in users:
        person_partitions[user.fingerprint_person].add(user.split)
    if any(len(splits) != 1 for splits in person_partitions.values()):
        raise RuntimeError("Fingerprint participant leakage across splits")
    return users, {"face_people": face_splits, "fingerprint_people": fingerprint_splits}


def make_pairs(data, users, split, seed, epoch=0):
    """Train: every face once as query; held-out: fixed enrollment, other faces as queries.

    Finger queries cycle through captures, covering all eight in training. In
    validation/test, capture 1 is enrollment and 2..8 cycle as queries. Camera a
    is preferred for face enrollment. No quality score or session is inferred.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, SPLITS.index(split)]))
    pairs = []
    for ui, user in enumerate(users):
        if user.split != split:
            continue
        faces = data.groups["face"][user.face_person]
        fingers = data.groups["fingerprint"][user.finger]
        frontal = [i for i in faces if data.cameras[i] == "a"] or faces
        if split == "train":
            q_faces = rng.permutation(faces).tolist()
            q_fingers = rng.permutation(fingers).tolist()
            for position, qf in enumerate(q_faces):
                candidates = [i for i in frontal if i != qf] or [i for i in faces if i != qf]
                ef = int(rng.choice(candidates))
                qp = q_fingers[(position + epoch + ui) % len(q_fingers)]
                ep = int(rng.choice([i for i in fingers if i != qp]))
                pairs.append((ef, qf, ep, qp, ui))
        else:
            ef, ep = frontal[0], fingers[0]
            q_fingers = [i for i in fingers if i != ep]
            for position, qf in enumerate(i for i in faces if i != ef):
                qp = q_fingers[(position + ui) % len(q_fingers)]
                pairs.append((ef, qf, ep, qp, ui))
    rows = np.asarray(pairs, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 5 or len(rows) < 2:
        raise ValueError(f"Insufficient genuine pairs in {split}")
    if np.any(rows[:, EF] == rows[:, QF]) or np.any(rows[:, EP] == rows[:, QP]):
        raise RuntimeError("Enrollment and query must use different captures")
    for modality, columns, identities in (
        ("face", (EF, QF), {u.face_person for u in users if u.split == split}),
        ("fingerprint", (EP, QP), {u.finger for u in users if u.split == split}),
    ):
        required = {i for identity in identities for i in data.groups[modality][identity]}
        covered = set(rows[:, columns].ravel().tolist())
        if required != covered:
            raise RuntimeError(f"Not all {split} {modality} images are used in enrollment/query")
    return rows


def finger_labels(rows, users):
    return [users[int(i)].finger for i in rows[:, USER]]


def negative_permutation(rows, users):
    """Group-and-rotate derangement excluding shared fingers AND shared faces."""
    labels = finger_labels(rows, users)
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    largest = max(map(len, groups.values()))
    if 2 * largest > len(rows):
        raise ValueError("A fingerprint identity dominates this batch; a valid negative permutation is impossible")
    order = np.asarray([i for label in sorted(groups) for i in groups[label]], dtype=np.int64)
    negative = np.empty(len(rows), dtype=np.int64)
    negative[order] = np.roll(order, -largest)
    for i, j in enumerate(negative):
        left, right = users[int(rows[i, USER])], users[int(rows[j, USER])]
        if left.face_person == right.face_person or left.finger == right.finger:
            raise RuntimeError("Invalid negative: shared face or shared fingerprint identity")
    return negative


def training_batches(rows, users, batch_size, rng):
    """Do not drop data. Pad rare unbalanced batches with valid contrastive rows.

    Padding is counted in the epoch log. It does not change the user mapping;
    all original training pairs are visited before any supplementary sampling.
    """
    order = rng.permutation(len(rows))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        tail = chunks.pop()
        chunks[-1] = np.concatenate((chunks[-1], tail))
    all_labels = finger_labels(rows, users)
    batches, padded = [], 0
    for indices in chunks:
        labels = [all_labels[int(i)] for i in indices]
        dominant, count = Counter(labels).most_common(1)[0]
        extra = max(0, 2 * count - len(indices))
        if extra:
            choices = [i for i, label in enumerate(all_labels) if label != dominant]
            if not choices:
                raise ValueError("At least two different fingerprints are required")
            indices = np.concatenate((indices, rng.choice(choices, size=extra, replace=True)))
            padded += extra
        batch = rows[indices]
        batches.append((batch, negative_permutation(batch, users)))
    return batches, padded


def model_inputs(cache, rows):
    index = torch.as_tensor(rows[:, :4], dtype=torch.long, device=cache["face"].device)
    return (cache["face"][index[:, EF]], cache["fingerprint"][index[:, EP]],
            cache["face"][index[:, QF]], cache["fingerprint"][index[:, QP]])


def forward_pair(model, cache, rows):
    ef, ep, qf, qp = model_inputs(cache, rows)
    # Two independent forwards: enrollment never receives query embeddings.
    return model(ef, ep), model(qf, qp)
