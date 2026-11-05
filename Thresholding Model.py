"""Joint scalar threshold and per-bit reliability for frozen 512-D embeddings.

One forward call takes ONLY face and fingerprint embeddings from the same
virtual-user observation. Enrollment and query use separate forward calls with
the same weights; no query information enters the enrollment forward call.

Run a small wiring/optimization check (not a recognition experiment):
  bash feature_extractors/run.sh 'Thresholding Model.py' --smoke-test --device cpu
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ThresholdingConfig:
    # Starting settings for normalized embeddings, not validated hyperparameters.
    temperature: float = 0.02
    tau_bound: float = 0.1

    def __post_init__(self):
        for name in ("temperature", "tau_bound"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


class ThresholdingModel(nn.Module):
    """512 -> 128 per modality; 256 -> 64 shared; one scalar tau per row.

    The 128-D hidden features only predict tau/reliability. The ORIGINAL 512-D
    input is thresholded, preserving every BCH input coordinate. Embeddings
    should already be L2 normalized by the frozen feature extractors; this class
    does not silently normalize, project, or change them.
    """
    embedding_dim = 512
    checkpoint_format = "multimodal-thresholding-v1"

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ThresholdingConfig()
        self.face_mlp = nn.Sequential(nn.Linear(512, 128), nn.GELU())
        self.fingerprint_mlp = nn.Sequential(nn.Linear(512, 128), nn.GELU())
        self.shared_mlp = nn.Sequential(nn.Linear(256, 64), nn.GELU())
        self.threshold_head = nn.Linear(64, 1)
        self.face_reliability_head = nn.Sequential(
            nn.Linear(1088, 256), nn.GELU(), nn.Linear(256, 512))
        self.fingerprint_reliability_head = nn.Sequential(
            nn.Linear(1088, 256), nn.GELU(), nn.Linear(256, 512))
        # Near-zero signed threshold, while retaining gradients to upstream MLPs.
        nn.init.normal_(self.threshold_head.weight, std=0.01)
        nn.init.zeros_(self.threshold_head.bias)
        self.stage = "joint"

    def _validate_inputs(self, face, fingerprint):
        reference = self.threshold_head.weight
        for name, value in (("face", face), ("fingerprint", fingerprint)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 2 or value.shape[1] != 512 or value.shape[0] < 1:
                raise ValueError(f"{name} must have shape (B, 512), with B >= 1")
            if not value.is_floating_point():
                raise TypeError(f"{name} must contain floating-point embeddings")
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"{name} must match model dtype/device: {reference.dtype}, {reference.device}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if face.shape != fingerprint.shape:
            raise ValueError("Face and fingerprint batch shapes must match")

    def forward(self, face_embedding, fingerprint_embedding):
        self._validate_inputs(face_embedding, fingerprint_embedding)
        context = self.shared_mlp(torch.cat((self.face_mlp(face_embedding),
                                            self.fingerprint_mlp(fingerprint_embedding)), dim=1))
        tau = self.config.tau_bound * torch.tanh(self.threshold_head(context))
        output = {"tau": tau, "context": context}
        for modality, embedding, head in (
            ("face", face_embedding, self.face_reliability_head),
            ("fingerprint", fingerprint_embedding, self.fingerprint_reliability_head),
        ):
            logits = (embedding - tau) / self.config.temperature
            distance = torch.abs(embedding - tau)
            reliability_logits = head(torch.cat((embedding, distance, context), dim=1))
            output.update({
                f"{modality}_soft_logits": logits,
                f"{modality}_soft_bits": torch.sigmoid(logits),
                f"{modality}_bits": (embedding >= tau).to(torch.uint8),
                f"{modality}_distance": distance,
                f"{modality}_reliability_logits": reliability_logits,
                f"{modality}_reliability": torch.sigmoid(reliability_logits),
            })
        return output

    def set_training_stage(self, stage):
        """threshold: only tau path; reliability: only reliability heads; inference: none.

        Construct a NEW optimizer from trainable_parameters() after each switch.
        Clearing old gradients prevents stale updates to newly frozen parameters.
        """
        if stage not in {"threshold", "reliability", "joint", "inference"}:
            raise ValueError("Unknown training stage")
        for name, parameter in self.named_parameters():
            is_reliability = "reliability_head" in name
            trainable = (stage == "joint" or stage == "reliability" and is_reliability
                         or stage == "threshold" and not is_reliability)
            parameter.requires_grad_(trainable)
            parameter.grad = None
        self.stage = stage
        self.train(stage != "inference")
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path, purpose="smoke_test_only_not_for_enrollment"):
        torch.save({"format": self.checkpoint_format, "config": asdict(self.config),
                    "state_dict": self.state_dict(), "purpose": purpose}, path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format") != cls.checkpoint_format:
            raise ValueError("Incompatible thresholding checkpoint")
        model = cls(ThresholdingConfig(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return model.to(device).set_training_stage("inference")


def soft_hamming(left, right):
    """Per-row soft mismatch fraction; equals Hamming/512 for hard binary inputs."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("soft_hamming requires equally shaped (B, D) tensors")
    return (left * (1 - right) + (1 - left) * right).mean(dim=1)


def threshold_loss(enrollment, query, negative_indices, *, margin=0.25,
                   different_weight=1.0, balance_weight=0.1):
    """Same-person face and same-finger genuine pairs in matching batch rows.

    negative_indices maps each enrollment row to a query from another identity.
    Caller MUST verify that those rows have different face people AND different
    fingerprint (person, finger) identities. IDs are sampling metadata, not model
    features. A derangement alone cannot establish that identities differ.
    """
    if not math.isfinite(margin) or not 0 < margin <= 1:
        raise ValueError("margin must be in (0, 1]")
    if not all(math.isfinite(x) and x >= 0 for x in (different_weight, balance_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    reference = enrollment["face_soft_bits"]
    batch = reference.shape[0]
    if batch < 2:
        raise ValueError("Threshold contrastive loss needs at least two identities")
    if not isinstance(negative_indices, torch.Tensor) or negative_indices.dtype != torch.long:
        raise TypeError("negative_indices must be a torch.long tensor")
    if negative_indices.shape != (batch,) or negative_indices.device != reference.device:
        raise ValueError("negative_indices must have shape (B,) on the model device")
    rows = torch.arange(batch, device=reference.device)
    if not torch.equal(torch.sort(negative_indices).values, rows) or torch.any(negative_indices == rows):
        raise ValueError("negative_indices must be a permutation without self pairs")
    same_terms, different_terms, balance_terms = [], [], []
    for modality in ("face", "fingerprint"):
        e, q = enrollment[f"{modality}_soft_bits"], query[f"{modality}_soft_bits"]
        if e.shape != (batch, 512) or q.shape != e.shape:
            raise ValueError("Expected matching (B, 512) soft bits")
        same_terms.append(soft_hamming(e, q).mean())
        different_terms.append(F.relu(margin - soft_hamming(e, q[negative_indices])).square().mean())
        # Balance each coordinate across identities/observations, not within one code.
        balance_terms.append((torch.cat((e, q), dim=0).mean(dim=0) - 0.5).square().mean())
    same, different, balance = [torch.stack(terms).mean()
                                for terms in (same_terms, different_terms, balance_terms)]
    total = same + different_weight * different + balance_weight * balance
    return {"total": total, "same": same, "different": different, "balance": balance}


def reliability_targets(enrollment, query):
    """Use ONLY genuine repeated captures. Labels mean bit agreement, not identity."""
    targets = {}
    for modality in ("face", "fingerprint"):
        e, q = enrollment[f"{modality}_bits"], query[f"{modality}_bits"]
        if e.shape != q.shape:
            raise ValueError("Mismatched enrollment/query bit shapes")
        targets[modality] = (e == q).to(torch.float32).detach()
    return targets


def reliability_loss(enrollment, query, targets):
    """Average BCE over both modalities and enrollment/query observations.

    Set model.set_training_stage('reliability') first, so these losses cannot
    change tau or shared features to make their own labels easier to predict.
    """
    terms = []
    for modality in ("face", "fingerprint"):
        target = targets[modality].detach()
        for output in (enrollment, query):
            logits = output[f"{modality}_reliability_logits"]
            if target.shape != logits.shape or not torch.isfinite(target).all():
                raise ValueError("Invalid reliability targets")
            if not torch.all((target == 0) | (target == 1)):
                raise ValueError("Reliability targets must be binary bit-agreement labels")
            terms.append(F.binary_cross_entropy_with_logits(logits, target.to(logits)))
    return torch.stack(terms).mean()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoke_batch(embedding_dir, batch_size, device):
    """Provisional TRAIN/reserved groups only; no validation/test measurement.

    Use one finger from each selected person, one distinct face person per finger.
    Every other finger of a selected fingerprint person stays in the TRAIN pool.
    This manifest is local to this smoke run, not a final research split.
    """
    arrays, files, groups, excluded = {}, {}, {}, {}
    for modality in ("face", "fingerprint"):
        with np.load(embedding_dir / f"{modality}_embeddings.npz", allow_pickle=False) as saved:
            arrays[modality] = saved["embeddings"].copy()
            files[modality] = saved["files"].tolist()
        if arrays[modality].shape != (len(files[modality]), 512) or not np.isfinite(arrays[modality]).all():
            raise ValueError(f"Invalid cached {modality} embeddings")
        if not np.allclose(np.linalg.norm(arrays[modality], axis=1), 1, atol=1e-4):
            raise ValueError("Smoke test expects cached L2-normalized embeddings")
        groups[modality], excluded[modality] = {}, []
        for i, name in enumerate(files[modality]):
            stem = Path(name).stem
            if modality == "face":
                if stem.startswith("ir"):
                    excluded[modality].append(name)
                    continue
                match = re.fullmatch(r"i(\d+)([a-z])([a-e])-(.+)", stem)
                if not match:
                    raise ValueError(f"Unparsed face filename: {name}")
                key = match[1]
            else:
                match = re.fullmatch(r"(\d+)_(\d+)_(\d+)", stem)
                if not match:
                    raise ValueError(f"Unparsed fingerprint filename: {name}")
                key = (match[1], match[2])
            groups[modality].setdefault(key, []).append(i)
    face_people = sorted(groups["face"])
    finger_people = sorted({key[0] for key in groups["fingerprint"]})
    if not 2 <= batch_size < len(finger_people) or batch_size >= len(face_people):
        raise ValueError("Smoke batch must use >=2 identities and leave people reserved in both datasets")
    selected_faces, selected_people = face_people[:batch_size], finger_people[:batch_size]
    indices = {f"{role}_{m}": [] for role in ("enrollment", "query") for m in ("face", "fingerprint")}
    mapping = []
    for j, (face_person, finger_person) in enumerate(zip(selected_faces, selected_people)):
        finger_key = min((k for k in groups["fingerprint"] if k[0] == finger_person), key=lambda k: int(k[1]))
        # Prefer two frontal observations; fall back to other original cameras if needed.
        face_rows = sorted(groups["face"][face_person], key=lambda i: (
            re.fullmatch(r"i(\d+)([a-z])([a-e])-(.+)", Path(files["face"][i]).stem)[3] != "a",
            files["face"][i]))
        finger_rows = sorted(groups["fingerprint"][finger_key],
                             key=lambda i: int(Path(files["fingerprint"][i]).stem.split("_")[2]))
        if len(face_rows) < 2 or len(finger_rows) < 2:
            raise ValueError("Smoke pairs need distinct repeated captures")
        entry = {"virtual_user": f"smoke_{j}", "face_person": face_person,
                 "fingerprint_person": finger_person, "finger_id": finger_key[1], "split": "smoke_train"}
        for modality, rows in (("face", face_rows), ("fingerprint", finger_rows)):
            for role, i in zip(("enrollment", "query"), rows[:2]):
                indices[f"{role}_{modality}"].append(i)
                entry[f"{role}_{modality}"] = files[modality][i]
        mapping.append(entry)
    tensors = {name: torch.tensor(arrays[name.split("_", 1)[1]][idx], dtype=torch.float32, device=device)
               for name, idx in indices.items()}
    negatives = torch.roll(torch.arange(batch_size, device=device), 1)
    for i, j in enumerate(negatives.cpu().tolist()):
        if mapping[i]["face_person"] == mapping[j]["face_person"] or (
            mapping[i]["fingerprint_person"], mapping[i]["finger_id"]) == (
                mapping[j]["fingerprint_person"], mapping[j]["finger_id"]):
            raise ValueError("Negative pairs must differ in both modalities")
    manifest = {"purpose": "smoke training only; synthetic cross-dataset pairing; no real same-person relation",
                "not_a_final_research_split": True, "mapping": mapping,
                "smoke_train_face_people": selected_faces, "reserved_face_people": face_people[batch_size:],
                "smoke_train_fingerprint_people": selected_people, "reserved_fingerprint_people": finger_people[batch_size:],
                "all_fingers_of_each_person_share_partition": True,
                "excluded_mirrored_files": excluded["face"], "negative_query_indices": negatives.cpu().tolist()}
    return tensors, negatives, manifest


def _snapshot(model, reliability):
    return {name: value.detach().clone() for name, value in model.named_parameters()
            if ("reliability_head" in name) == reliability}


def _unchanged(model, snapshot):
    current = dict(model.named_parameters())
    return all(torch.equal(value, current[name].detach()) for name, value in snapshot.items())


def _gradient_norms(model):
    norms = {}
    for name, value in model.named_parameters():
        if value.requires_grad:
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise RuntimeError(f"Missing/nonfinite gradient: {name}")
            block = name.split(".", 1)[0]
            norms[block] = norms.get(block, 0.0) + float(value.grad.norm())
        elif value.grad is not None:
            raise RuntimeError(f"Frozen parameter has a gradient: {name}")
    if not norms or any(value <= 0 for value in norms.values()):
        raise RuntimeError("At least one trainable block received zero gradient")
    return norms


def run_smoke_test(embedding_dir, output_dir, device="cpu", batch_size=4, steps=10, seed=17):
    root = Path(__file__).resolve().parent
    embedding_dir, output_dir = Path(embedding_dir).resolve(), Path(output_dir).resolve()
    for protected in (root / "muct-master", root / "UareU", root / "feature_extractors", root / "BCH"):
        if output_dir.is_relative_to(protected.resolve()):
            raise ValueError("Smoke outputs must be separate from datasets and existing modules")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"} or device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device unavailable: {device}")
    if steps < 1:
        raise ValueError("Smoke steps must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_files = [embedding_dir / f"{m}_embeddings.npz" for m in ("face", "fingerprint")]
    original_hashes = {str(p): _sha256(p) for p in source_files}
    data, negatives, manifest = _smoke_batch(embedding_dir, batch_size, device)
    (output_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = ThresholdingModel().to(device)

    def forward_pair():
        return (model(data["enrollment_face"], data["enrollment_fingerprint"]),
                model(data["query_face"], data["query_fingerprint"]))

    checks, histories, gradient_reports = {}, {}, {}
    model.set_training_stage("threshold")
    frozen_heads, initial_tau = _snapshot(model, True), _snapshot(model, False)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = threshold_loss(*forward_pair(), negatives)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("Nonfinite threshold loss")
        history.append({key: float(value.detach()) for key, value in losses.items()})
        losses["total"].backward()
        norms = _gradient_norms(model)
        if step == 0:
            gradient_reports["threshold"] = norms
        optimizer.step()
    with torch.no_grad():
        history.append({key: float(value) for key, value in threshold_loss(*forward_pair(), negatives).items()})
    histories["threshold"] = history
    checks["threshold_loss_decreased_on_smoke_batch"] = history[-1]["total"] < history[0]["total"]
    checks["threshold_parameters_updated"] = not _unchanged(model, initial_tau)
    checks["reliability_heads_frozen_in_threshold_stage"] = _unchanged(model, frozen_heads)

    model.set_training_stage("reliability")
    frozen_tau, initial_heads = _snapshot(model, False), _snapshot(model, True)
    with torch.no_grad():
        enrollment_before, query_before = forward_pair()
        targets = reliability_targets(enrollment_before, query_before)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = reliability_loss(*forward_pair(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite reliability loss")
        history.append(float(loss.detach()))
        loss.backward()
        norms = _gradient_norms(model)
        if step == 0:
            gradient_reports["reliability"] = norms
        optimizer.step()
    with torch.no_grad():
        history.append(float(reliability_loss(*forward_pair(), targets)))
    histories["reliability"] = history
    checks["reliability_loss_decreased_on_smoke_batch"] = history[-1] < history[0]
    checks["reliability_parameters_updated"] = not _unchanged(model, initial_heads)
    checks["tau_path_frozen_in_reliability_stage"] = _unchanged(model, frozen_tau)

    model.set_training_stage("inference")
    with torch.no_grad():
        enrollment, query = forward_pair()
        single = model(data["query_face"][:1], data["query_fingerprint"][:1])
    checks["finite_outputs"] = all(torch.isfinite(v).all().item() for o in (enrollment, query) for v in o.values())
    checks["per_sample_scalar_tau"] = enrollment["tau"].shape == query["tau"].shape == (batch_size, 1)
    checks["singleton_matches_batch"] = all(torch.allclose(single[k], query[k][:1], atol=1e-6, rtol=1e-5) for k in query)
    for name, before, after in (("enrollment", enrollment_before, enrollment), ("query", query_before, query)):
        checks[f"{name}_tau_unchanged_by_reliability_training"] = torch.equal(before["tau"], after["tau"])
    hard_summary = {}
    for modality in ("face", "fingerprint"):
        hard = query[f"{modality}_bits"]
        reliability = query[f"{modality}_reliability"]
        checks[f"{modality}_hard_bits_valid"] = (hard.shape == (batch_size, 512) and hard.dtype == torch.uint8
                                                 and bool(torch.all((hard == 0) | (hard == 1))))
        checks[f"{modality}_reliability_in_range"] = bool(torch.all((reliability >= 0) & (reliability <= 1)))
        checks[f"{modality}_nonconstant_codes"] = (torch.unique(hard, dim=0).shape[0] > 1
                                                    and bool(torch.all((hard.sum(1) > 0) & (hard.sum(1) < 512))))
        checks[f"{modality}_distance_uses_current_tau"] = torch.equal(query[f"{modality}_distance"],
                                                   (data[f"query_{modality}"] - query["tau"]).abs())
        hard_summary[modality] = {"shape": list(hard.shape), "ones_fraction": float(hard.float().mean()),
                                  "genuine_bit_agreement_fraction": float(targets[modality].mean())}
    checkpoint = output_dir / "smoke_only.pt"
    model.save_checkpoint(checkpoint)
    restored = ThresholdingModel.load_checkpoint(checkpoint, device)
    with torch.no_grad():
        reloaded = restored(data["query_face"], data["query_fingerprint"])
    checks["checkpoint_roundtrip_exact"] = all(torch.equal(query[k], reloaded[k]) for k in query)
    checks["cached_embeddings_unchanged"] = all(_sha256(p) == original_hashes[str(p)] for p in source_files)
    np.savez_compressed(output_dir / "smoke_outputs.npz", **{
        f"{role}_{key}": tensor.cpu().numpy()
        for role, output in (("enrollment", enrollment), ("query", query))
        for key, tensor in output.items()})
    report = {"purpose": "smoke test only: short optimization on one fixed training batch; not generalization or FE evaluation",
              "passed": all(checks.values()), "checks": checks, "device": device,
              "torch": str(torch.__version__), "seed": seed, "batch_size": batch_size,
              "steps_per_stage": steps, "config": asdict(model.config),
              "loss_settings": {"margin": 0.25, "different_weight": 1.0, "balance_weight": 0.1,
                                "optimizer": "Adam", "learning_rate": 0.001},
              "gradients_first_step": gradient_reports, "loss_history": histories,
              "smoke_batch_only_hard_bit_diagnostics": hard_summary,
              "tau_enrollment": enrollment["tau"].flatten().cpu().tolist(),
              "tau_query": query["tau"].flatten().cpu().tolist(),
              "input_archive_sha256": original_hashes, "source_sha256": _sha256(__file__),
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "device": device,
                      "threshold_loss": [histories["threshold"][i]["total"] for i in (0, -1)],
                      "reliability_loss": [histories["reliability"][i] for i in (0, -1)],
                      "report": str(output_dir / "report.json")}), flush=True)
    if not report["passed"]:
        raise RuntimeError(f"Failed smoke checks: {[k for k, v in checks.items() if not v]}")
    return report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--embedding-dir", type=Path,
                        default=root / "feature_extractors/results/full_dataset_check")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not args.smoke_test:
        parser.print_help()
        return
    output = args.output_dir or root / "thresholding/results" / datetime.now().strftime("smoke_%Y%m%d_%H%M%S_%f")
    run_smoke_test(args.embedding_dir, output, args.device, args.batch_size, args.steps, args.seed)


if __name__ == "__main__":
    main()
