"""Joint, input-dependent thresholds for two frozen 512-D embeddings.

This module does not load data or train at import time. Inputs are already
L2-normalized embeddings; they are not reprojected before binarization.
Enrollment and query must call the same frozen model separately.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn

MODALITIES = ("face", "fingerprint")
MODES = ("shared_scalar", "separate_scalar", "shared_512", "separate_512")


@dataclass(frozen=True)
class ThresholdingV3Config:
    mode: str = "separate_512"
    hidden_dim: int = 128
    context_dim: int = 64
    temperature: float = .02
    tau_base_bound: float = .01
    tau_delta_bound: float = .01

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        for name in ("hidden_dim", "context_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("temperature", "tau_base_bound", "tau_delta_bound"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def tau_dim(self):
        return 1 if self.mode.endswith("scalar") else 512

    @property
    def shared(self):
        return self.mode.startswith("shared")

    @property
    def tau_bound(self):
        return self.tau_base_bound + self.tau_delta_bound


def encoder(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                         nn.GELU(), nn.Linear(hidden_dim, output_dim),
                         nn.LayerNorm(output_dim), nn.GELU())


class ThresholdHead(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        self.base_logit = nn.Parameter(torch.zeros(config.tau_dim))
        self.context = nn.Sequential(nn.Linear(input_dim, config.hidden_dim),
                                     nn.LayerNorm(config.hidden_dim), nn.GELU())
        self.output = nn.Linear(config.hidden_dim, config.tau_dim)
        # Identical tau=0 bits for every architecture at epoch zero.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        base = (self.config.tau_base_bound * self.base_logit.tanh()).expand(len(x), -1)
        delta = self.config.tau_delta_bound * self.output(self.context(x)).tanh()
        return base + delta, base, delta


class ThresholdingModelV3(nn.Module):
    """Shared modes return tau (B,1)/(B,512); separate modes return two taus.

All modes also expose face_tau/fingerprint_tau for a uniform training API.
Hard bits have shape (B,512), regardless of the threshold dimension.
No user identifiers, enrollment templates, helper data or keys are inputs.
"""
    checkpoint_format = "multimodal-threshold-v3-v1"

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ThresholdingV3Config()
        c = self.config
        self.branches = nn.ModuleDict({m: encoder(512, c.hidden_dim, c.context_dim) for m in MODALITIES})
        self.fusion = encoder(2*c.context_dim, c.hidden_dim, c.context_dim)
        names = ("shared",) if c.shared else MODALITIES
        self.heads = nn.ModuleDict({name: ThresholdHead(c.context_dim if c.shared else 2*c.context_dim, c)
                                   for name in names})

    def _validate(self, face, fingerprint):
        reference = next(self.parameters())
        for x in (face, fingerprint):
            if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 512 or len(x) == 0:
                raise ValueError("Expected nonempty (batch,512) floating-point embeddings")
            if x.dtype != reference.dtype or x.device != reference.device or not bool(torch.isfinite(x).all()):
                raise ValueError("Inputs must be finite and match model dtype/device")
        if face.shape != fingerprint.shape:
            raise ValueError("Face and fingerprint batches must have the same shape")

    def forward(self, face_embedding, fingerprint_embedding):
        self._validate(face_embedding, fingerprint_embedding)
        embeddings = dict(zip(MODALITIES, (face_embedding, fingerprint_embedding)))
        hidden = {m: self.branches[m](embeddings[m]) for m in MODALITIES}
        joint = self.fusion(torch.cat([hidden[m] for m in MODALITIES], dim=1))
        if self.config.shared:
            shared = self.heads["shared"](joint)
            thresholds = {m: shared for m in MODALITIES}
        else:
            thresholds = {m: self.heads[m](torch.cat((hidden[m], joint), dim=1)) for m in MODALITIES}
        result = {}
        for m in MODALITIES:
            tau, base, delta = thresholds[m]
            margin = embeddings[m] - tau
            values = {"tau": tau, "base_tau": base, "delta_tau": delta, "margin": margin,
                      "soft_bits": torch.sigmoid(margin / self.config.temperature),
                      "bits": (margin >= 0).to(torch.uint8)}
            result.update({f"{m}_{k}": v for k, v in values.items()})
        if self.config.shared:
            result["tau"] = shared[0]
        return result

    def save_checkpoint(self, path, metadata):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format": self.checkpoint_format, "config": asdict(self.config),
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    "metadata": metadata}, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != cls.checkpoint_format:
            raise ValueError("Not a Thresholding V3 checkpoint")
        model = cls(ThresholdingV3Config(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).eval(), payload["metadata"]
