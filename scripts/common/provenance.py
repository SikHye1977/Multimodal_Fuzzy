"""Validate historical/new source runs without weakening participant/data checks."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
from scripts import common
from .paths import ROOT

HISTORICAL_TRAINER_SHA256 = "0278837bbcc58250a86776c46b93af6c95576e7a63d3015becce937b50bdb448"
HISTORICAL_TRAINER = ROOT / "scripts/common/provenance_snapshots/train_thresholding.py.txt"
SOURCE_FORMAT = "thresholding-bootstrap-v2"


def source_code_paths(entrypoint):
    """Hash the entry point and every shared dependency, including C++ sources."""
    paths = {Path(entrypoint).resolve(), ROOT / "scripts/__init__.py", HISTORICAL_TRAINER,
             ROOT / "Thresholding Model.py", ROOT / "Thresholding Model V3.py",
             ROOT / "Thresholding_Model_512.py", ROOT / "Thresholding_Model_Set.py",
             ROOT / "thresholding/variability_model.py", ROOT / "thresholding/distance_reliability.py",
             ROOT / "scripts/common/bch_bridge.cpp", ROOT / "BCH/BCH.cpp", ROOT / "BCH/BCH.hpp"}
    paths.update((ROOT / "scripts/common").glob("*.py"))
    paths.add(Path(entrypoint).resolve().parent / "__init__.py")
    return sorted(paths)


def code_hashes(entrypoint, *, relative=False):
    return {str(p.relative_to(ROOT) if relative else p): common.sha256(p)
            for p in source_code_paths(entrypoint)}


def validate_source_code(config):
    if common.sha256(ROOT / "Thresholding Model.py") != config["model_sha256"]:
        raise ValueError("Source experiment model has changed")
    if common.sha256(HISTORICAL_TRAINER) != HISTORICAL_TRAINER_SHA256:
        raise ValueError("Historical trainer snapshot has changed")
    expected = config["trainer_sha256"]
    if expected == HISTORICAL_TRAINER_SHA256 and "source_format" not in config:
        return
    trainer = ROOT / "scripts/bootstrap/train.py"
    if (config.get("source_format") != SOURCE_FORMAT
            or expected != common.sha256(trainer)
            or config.get("code_sha256") != code_hashes(trainer, relative=True)):
        raise ValueError("Source experiment code/dependencies have changed")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def prepare_source(args):
    """Refuse incompatible sources rather than silently change the comparison."""
    source = args.source_run
    config = json.loads((source / "config.json").read_text())
    if json.loads((source / "status.json").read_text()).get("state") != "completed":
        raise ValueError("--source-run must be a completed original Thresholding experiment")
    validate_source_code(config)
    embedding_dir = args.embedding_dir or Path(config["embedding_dir"])
    data = common.Embeddings(embedding_dir)
    expected_hashes = {Path(path).name: digest for path, digest in config["input_sha256"].items()}
    if {Path(path).name: digest for path, digest in data.hashes.items()} != expected_hashes:
        raise ValueError("Embeddings differ from the source experiment")
    users, partitions = common.make_users(data, int(config["split_seed"]))
    expected_users = [{"virtual_user_index": str(i), **asdict(user)} for i, user in enumerate(users)]
    if read_csv(source / "virtual_users.csv") != expected_users:
        raise ValueError("Source virtual-user mapping cannot be reproduced exactly")
    saved_splits = json.loads((source / "splits.json").read_text())
    if any(saved_splits[key] != value for key, value in partitions.items()):
        raise ValueError("Source participant splits differ")
    pairs = {}
    for split in common.SPLITS:
        rows = common.make_pairs(data, users, split, int(config["seed"]), epoch=1 if split == "train" else 0)
        expected = [{"virtual_user_index": str(int(row[common.USER])),
                     "enrollment_face": data.files["face"][int(row[common.EF])],
                     "query_face": data.files["face"][int(row[common.QF])],
                     "enrollment_fingerprint": data.files["fingerprint"][int(row[common.EP])],
                     "query_fingerprint": data.files["fingerprint"][int(row[common.QP])]}
                    for row in rows]
        if read_csv(source / f"{split}_pairs.csv") != expected:
            raise ValueError(f"Source {split} pairs cannot be reproduced exactly")
        pairs[split] = rows
    filenames = ("config.json", "status.json", "splits.json", "virtual_users.csv", "samples.csv",
                 "train_pairs.csv", "validation_pairs.csv", "test_pairs.csv", "best_threshold.pt", "final_model.pt")
    source_hashes = {str(source / name): common.sha256(source / name) for name in filenames}
    return data, users, pairs, config, source_hashes
