"""Thresholds from sets of frozen biometric embeddings, without enrollment lookup.

Each invocation sees ONLY its own face and fingerprint captures. Enrollment and
query invoke the same model independently. No dataset/identity IDs are inputs.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("face", "fingerprint")
STAT_DIM = 1026  # normalized mean, sample standard deviation, log count, availability


def set_statistics(x):
    """Order-invariant statistics of a nonempty (B,N,512) set.

The one-capture std placeholder is zero, but availability=0 explicitly marks
it as unobserved, not evidence of perfect stability. No across-user statistics.
"""
    if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3
            or x.shape[2] != 512 or not x.shape[0] or not x.shape[1]
            or not bool(torch.isfinite(x).all())):
        raise ValueError("Expected finite floating-point (batch,captures,512) embeddings")
    if not torch.allclose(x.norm(dim=2), torch.ones_like(x[:, :, 0]), atol=1e-4, rtol=1e-4):
        raise ValueError("Each source embedding must be L2 normalized")
    mean = x.mean(1)
    if bool((mean.norm(dim=1) < 1e-8).any()):
        raise ValueError("Near-zero mean embedding cannot be normalized")
    representative = F.normalize(mean, dim=1)
    n = x.shape[1]
    std = x.std(1, unbiased=True) if n > 1 else torch.zeros_like(mean)
    count = torch.full_like(mean[:, :1], math.log1p(n))
    available = torch.full_like(count, float(n > 1))
    return torch.cat((representative, std, count, available), dim=1)


@dataclass(frozen=True)
class SetThresholdConfig:
    hidden_dim: int = 128
    context_dim: int = 64
    temperature: float = .02
    tau_base_bound: float = .01
    tau_delta_bound: float = .01

    def __post_init__(self):
        for name in ("hidden_dim", "context_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("temperature", "tau_base_bound", "tau_delta_bound"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def tau_bound(self):
        return self.tau_base_bound + self.tau_delta_bound


def encoder(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                         nn.GELU(), nn.Linear(hidden_dim, output_dim),
                         nn.LayerNorm(output_dim), nn.GELU())


class SetThresholdHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_logit = nn.Parameter(torch.zeros(512))
        self.context = nn.Sequential(nn.Linear(2*config.context_dim, config.hidden_dim),
                                     nn.LayerNorm(config.hidden_dim), nn.GELU())
        self.output = nn.Linear(config.hidden_dim, 512)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class SetThresholdingModel(nn.Module):
    """Per-modality 512-D tau; hard bits compare the normalized set mean to tau."""
    checkpoint_format = "multimodal-set-threshold-v1"

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SetThresholdConfig()
        c = self.config
        self.branches = nn.ModuleDict({m: encoder(STAT_DIM, c.hidden_dim, c.context_dim) for m in MODALITIES})
        self.fusion = encoder(2*c.context_dim, c.hidden_dim, c.context_dim)
        self.heads = nn.ModuleDict({m: SetThresholdHead(c) for m in MODALITIES})

    def forward(self, face_sets, fingerprint_sets):
        return self.forward_statistics(set_statistics(face_sets), set_statistics(fingerprint_sets))

    def forward_statistics(self, face_stats, fingerprint_stats):
        """Equivalent fast path for statistics precomputed from frozen embeddings."""
        reference = next(self.parameters())
        for x in (face_stats, fingerprint_stats):
            if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != STAT_DIM
                    or not len(x) or x.dtype != reference.dtype or x.device != reference.device
                    or not bool(torch.isfinite(x).all())):
                raise ValueError("Statistics must have shape (B,1026), finite values and model dtype/device")
        if face_stats.shape != fingerprint_stats.shape:
            raise ValueError("Face and fingerprint batch sizes must agree")
        stats = dict(zip(MODALITIES, (face_stats, fingerprint_stats)))
        hidden = {m: self.branches[m](stats[m]) for m in MODALITIES}
        joint = self.fusion(torch.cat([hidden[m] for m in MODALITIES], dim=1))
        result = {}
        for m in MODALITIES:
            head = self.heads[m]
            base = (self.config.tau_base_bound*head.base_logit.tanh()).expand(len(face_stats), -1)
            delta = self.config.tau_delta_bound*head.output(head.context(torch.cat((hidden[m], joint), 1))).tanh()
            tau = base + delta
            representative = stats[m][:, :512]
            margin = representative-tau
            values = {"tau": tau, "base_tau": base, "delta_tau": delta, "representative": representative,
                      "soft_bits": (margin/self.config.temperature).sigmoid(),
                      "bits": (margin >= 0).to(torch.uint8)}
            result.update({m+"_"+key: value for key, value in values.items()})
        return result

    def save_checkpoint(self, path, metadata):
        path = Path(path)
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save({"format": self.checkpoint_format, "config": asdict(self.config),
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    "metadata": metadata}, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != cls.checkpoint_format:
            raise ValueError("Not a set-threshold checkpoint")
        model = cls(SetThresholdConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).eval(), payload["metadata"]
