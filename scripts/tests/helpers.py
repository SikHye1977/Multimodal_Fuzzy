"""Shared synthetic inputs; never reads biometric datasets."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from scripts import common

def synthetic_data():
    rng = np.random.default_rng(91)
    users, groups, files, arrays, cameras = [], {}, {}, {}, []
    for m in common.MODALITIES:
        groups[m], files[m], arrays[m] = {}, [], []
        for split_index, split in enumerate(common.SPLITS):
            for person in range(3):
                name = f"{split_index}{person}"
                identity = name if m == "face" else (name, "1")
                center = rng.normal(size=512)
                indices = []
                for capture in range(4):
                    indices.append(len(arrays[m]))
                    x = center + rng.normal(scale=.45, size=512)
                    arrays[m].append((x / np.linalg.norm(x)).astype(np.float32))
                    files[m].append(f"synthetic_{m}_{name}_{capture}")
                    if m == "face":
                        cameras.append("a" if capture == 0 else "b")
                groups[m][identity] = indices
                if m == "face":
                    users.append(common.VirtualUser(split, name, name, "1"))
        arrays[m] = np.stack(arrays[m])
    original = {m: a.copy() for m, a in arrays.items()}

    def assert_unchanged():
        for m in common.MODALITIES:
            np.testing.assert_array_equal(original[m], arrays[m])

    data = SimpleNamespace(groups=groups, files=files, arrays=arrays, cameras=cameras,
                           hashes={}, assert_unchanged=assert_unchanged)
    pairs = {s: common.make_pairs(data, users, s, 17, epoch=1 if s == "train" else 0)
             for s in common.SPLITS}
    return data, users, pairs


def multisample_data():
    rng = np.random.default_rng(104)
    users, arrays, files, groups = [], {}, {}, {}
    for s, split in enumerate(common.SPLITS):
        for i in range(4):
            users.append(common.VirtualUser(split, f"f{s}-{i}", f"p{s}-{i//2}", str(i % 2)))
    for m in common.MODALITIES:
        arrays[m], files[m], groups[m] = [], [], {}
        for u in users:
            identity = u.face_person if m == "face" else u.finger
            center = rng.normal(size=512)
            ids = []
            for k in range(10 if m == "face" else 8):
                ids.append(len(arrays[m]))
                x = center+rng.normal(scale=.6, size=512)
                arrays[m].append((x/np.linalg.norm(x)).astype(np.float32))
                files[m].append(f"synthetic-{m}-{identity}-{k}")
            groups[m][identity] = ids
        arrays[m] = np.stack(arrays[m])
    original = deepcopy(arrays)
    def assert_unchanged():
        for m in common.MODALITIES:
            np.testing.assert_array_equal(original[m], arrays[m])
    return SimpleNamespace(arrays=arrays, files=files, groups=groups, hashes={},
                           assert_unchanged=assert_unchanged), users
