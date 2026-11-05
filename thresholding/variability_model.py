"""Independent scalar thresholds and capture-variability-aware reliability.

No data loading or training on import. Each observation is processed alone;
identity labels and the other observation of a genuine pair never enter forward.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("face", "fingerprint")


@dataclass(frozen=True)
class ModelConfig:
    arm: str = "B"
    temperature: float = 0.02
    tau_base_bound: float = 0.01
    tau_delta_bound: float = 0.01
    context_dim: int = 32
    coordinate_dim: int = 4

    def __post_init__(self):
        if self.arm not in ("A", "B", "C"):
            raise ValueError("arm must be A, B or C")
        for key in ("temperature", "tau_base_bound", "tau_delta_bound"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("context_dim", "coordinate_dim"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")

    @property
    def tau_bound(self):
        return self.tau_base_bound + self.tau_delta_bound


class ScalarThreshold(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_logit = nn.Parameter(torch.zeros(()))
        self.context = nn.Sequential(nn.Linear(512, config.context_dim), nn.GELU())
        self.delta = nn.Linear(config.context_dim, 1)
        # An exact zero-threshold initial/fallback policy for all three arms.
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        if self.config.arm != "C":
            return x.new_zeros((len(x), 1))
        return (self.config.tau_base_bound * self.base_logit.tanh()
                + self.config.tau_delta_bound * self.delta(self.context(x)).tanh())


class ReliabilityHead(nn.Module):
    """A: two learned scalars. B/C: same prior plus a small per-bit residual.

The coordinate embedding identifies a feature position, NEVER a person/finger.
The residual is not constrained to be monotone or to predict r >= 0.5.
"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.raw_slope = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.intercept = nn.Parameter(torch.zeros(()))
        self.register_buffer("distance_scale", torch.ones(()))
        self.register_buffer("variability_scale", torch.ones(512))
        self.register_buffer("constant_probability", torch.tensor(0.5))
        if config.arm != "A":
            self.context = nn.Sequential(nn.Linear(512, config.context_dim), nn.GELU())
            self.coordinates = nn.Parameter(torch.empty(512, config.coordinate_dim))
            nn.init.normal_(self.coordinates, std=0.01)
            self.residual = nn.Sequential(
                nn.Linear(4 + config.context_dim + config.coordinate_dim, 32),
                nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x, margin):
        scaled = margin / self.distance_scale
        logits = (F.softplus(self.raw_slope) + 1e-6) * scaled.abs() + self.intercept
        if self.config.arm != "A":
            standardized = (margin / self.variability_scale).clamp(-20, 20)
            local = torch.stack((scaled, scaled.abs(), standardized, standardized.abs()), dim=2)
            context = self.context(x)[:, None, :].expand(-1, 512, -1)
            coordinate = self.coordinates[None].expand(len(x), -1, -1)
            logits = logits + self.residual(torch.cat((local, context, coordinate), dim=2)).squeeze(2)
        return logits

    @torch.no_grad()
    def set_statistics(self, stats):
        distance = float(stats["mean_distance"])
        probability = float(stats["agreement_rate"])
        scale = torch.as_tensor(stats["variability_scale"], device=self.variability_scale.device,
                                dtype=self.variability_scale.dtype)
        if not math.isfinite(distance) or distance <= 0 or not 0 < probability < 1:
            raise ValueError("Invalid training distance/agreement statistics")
        if scale.shape != (512,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("Expected positive finite 512-D variability scales")
        self.distance_scale.fill_(distance)
        self.variability_scale.copy_(scale)
        self.constant_probability.fill_(probability)
        self.raw_slope.fill_(math.log(math.expm1(1.0)))
        self.intercept.fill_(math.log(probability / (1 - probability))
                             - float(F.softplus(self.raw_slope) + 1e-6))


class VariabilityThresholdModel(nn.Module):
    checkpoint_format = "multimodal-threshold-reliability-abc-v1"

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.thresholds = nn.ModuleDict({m: ScalarThreshold(self.config) for m in MODALITIES})
        self.reliability = nn.ModuleDict({m: ReliabilityHead(self.config) for m in MODALITIES})
        self.set_stage("inference")

    def quantize(self, modality, x):
        reference = self.thresholds[modality].base_logit
        if x.ndim != 2 or x.shape[1] != 512 or len(x) < 1:
            raise ValueError("Expected nonempty (B,512) embeddings")
        if x.dtype != reference.dtype or x.device != reference.device or not bool(torch.isfinite(x).all()):
            raise ValueError("Inputs must be finite and match model dtype/device")
        tau = self.thresholds[modality](x)
        margin = x - tau
        return {"tau": tau, "margin": margin, "distance": margin.abs(),
                "soft_bits": torch.sigmoid(margin / self.config.temperature),
                "bits": (margin >= 0).to(torch.uint8)}

    def forward(self, face_embedding, fingerprint_embedding):
        if face_embedding.shape != fingerprint_embedding.shape:
            raise ValueError("Face and fingerprint batch shapes must match")
        output = {}
        for m, x in zip(MODALITIES, (face_embedding, fingerprint_embedding)):
            values = self.quantize(m, x)
            # Reliability cannot change the quantizer, including accidental joint backprop.
            logits = self.reliability[m](x.detach(), values["margin"].detach())
            values.update(reliability_logits=logits, reliability=torch.sigmoid(logits))
            output.update({f"{m}_{key}": value for key, value in values.items()})
        return output

    def set_stage(self, stage):
        if stage not in ("threshold", "reliability", "inference"):
            raise ValueError("Invalid training stage")
        if stage == "threshold" and self.config.arm != "C":
            raise ValueError("Only C trains a threshold")
        for name, p in self.named_parameters():
            p.requires_grad_(stage != "inference" and name.startswith(
                "thresholds." if stage == "threshold" else "reliability."))
            p.grad = None
        self.train(stage != "inference")
        return self

    def fixed_reliability_state(self):
        """All quantizer state plus fitted buffers must remain fixed in stage 2."""
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
            raise ValueError("Use the matching ABC checkpoint loader")
        model = cls(ModelConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).set_stage("inference"), payload["metadata"]
