"""512 coordinate thresholds for frozen face/fingerprint embeddings.

Each modality processes its own current observation. No identity, paired sample,
enrollment record or helper is an input. Training and data loading are external.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn

from thresholding.variability_model import ModelConfig, ReliabilityHead, MODALITIES

MODES = ("zero", "fixed", "dynamic")


@dataclass(frozen=True)
class Thresholding512Config:
    mode: str = "dynamic"
    temperature: float = .02
    tau_base_bound: float = .01
    tau_delta_bound: float = .01
    hidden_dim: int = 32

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        for key in ("temperature", "tau_base_bound", "tau_delta_bound"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not isinstance(self.hidden_dim, int) or self.hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")

    @property
    def tau_bound(self):
        return self.tau_base_bound + self.tau_delta_bound


class CoordinateThreshold(nn.Module):
    """Global 512-vector plus bounded, observation-dependent 512-vector.

fixed uses the same total bound as dynamic. Parameters are shared by ALL users.
There is no post-threshold bit permutation or embedding projection.
"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_logit = nn.Parameter(torch.zeros(512))
        if config.mode == "dynamic":
            self.context = nn.Sequential(nn.Linear(512, config.hidden_dim), nn.GELU())
            self.delta = nn.Linear(config.hidden_dim, 512)
            # All modes begin at the EXACT same zero-threshold bit baseline.
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        if self.config.mode == "zero":
            base, delta = torch.zeros_like(x), torch.zeros_like(x)
        elif self.config.mode == "fixed":
            base = (self.config.tau_bound * self.base_logit.tanh()).expand_as(x)
            delta = torch.zeros_like(x)
        else:
            base = (self.config.tau_base_bound * self.base_logit.tanh()).expand_as(x)
            delta = self.config.tau_delta_bound * self.delta(self.context(x)).tanh()
        return base + delta, base, delta


class ThresholdingModel512(nn.Module):
    """Forward returns face_tau and fingerprint_tau with shape (batch,512).

Reliability is a marginal genuine-pair bit-agreement probability, learned AFTER
freezing thresholds. It is neither an identity score nor an FE decoder output.
The reliability architecture is the same B head as the preceding ABC experiment.
"""
    checkpoint_format = "multimodal-threshold-512-v1"

    def __init__(self, config=None):
        super().__init__()
        self.config = config or Thresholding512Config()
        self.thresholds = nn.ModuleDict({m: CoordinateThreshold(self.config) for m in MODALITIES})
        self.reliability = nn.ModuleDict({m: ReliabilityHead(ModelConfig(arm="B")) for m in MODALITIES})
        self.register_buffer("reliability_fitted", torch.tensor(False))
        self.set_stage("threshold" if self.config.mode != "zero" else "inference")

    def quantize(self, modality, x):
        reference = self.thresholds[modality].base_logit
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 512 or len(x) == 0:
            raise ValueError("Expected nonempty floating-point (batch,512) embeddings")
        if x.dtype != reference.dtype or x.device != reference.device or not bool(torch.isfinite(x).all()):
            raise ValueError("Input must be finite and match model dtype/device")
        tau, base, delta = self.thresholds[modality](x)
        margin = x - tau
        return {"tau": tau, "base_tau": base, "delta_tau": delta, "margin": margin,
                "distance": margin.abs(), "soft_bits": torch.sigmoid(margin / self.config.temperature),
                "bits": (margin >= 0).to(torch.uint8)}

    def forward(self, face_embedding, fingerprint_embedding):
        if not isinstance(face_embedding, torch.Tensor) or not isinstance(fingerprint_embedding, torch.Tensor):
            raise ValueError("Inputs must be tensors")
        if face_embedding.shape != fingerprint_embedding.shape:
            raise ValueError("Face and fingerprint batch shapes must match")
        result = {}
        for m, x in zip(MODALITIES, (face_embedding, fingerprint_embedding)):
            values = self.quantize(m, x)
            logits = self.reliability[m](x.detach(), values["margin"].detach())
            values.update(reliability_logits=logits, reliability=logits.sigmoid())
            result.update({f"{m}_{k}": v for k, v in values.items()})
        return result

    def set_stage(self, stage):
        if stage not in ("threshold", "reliability", "inference"):
            raise ValueError("stage must be threshold, reliability or inference")
        if stage == "threshold" and self.config.mode == "zero":
            raise ValueError("zero mode has no trainable thresholds")
        prefix = "thresholds." if stage == "threshold" else "reliability."
        for name, p in self.named_parameters():
            p.requires_grad_(stage != "inference" and name.startswith(prefix))
            p.grad = None
        if stage == "threshold":
            self.reliability_fitted.fill_(False)
        return self.train(stage != "inference")

    def fixed_reliability_state(self):
        buffers = set(dict(self.named_buffers()))
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith("thresholds.") or k in buffers}

    def save_checkpoint(self, path, metadata):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save({"format": self.checkpoint_format, "config": asdict(self.config),
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    "metadata": metadata}, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != cls.checkpoint_format:
            raise ValueError("This loader accepts only Thresholding_Model_512 checkpoints")
        model = cls(Thresholding512Config(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).set_stage("inference"), payload["metadata"]
