"""Enrollment-only thresholds for frozen face/fingerprint embeddings.

forward accepts ONLY five enrollment captures per modality. Training queries
belong in the loss, never in the threshold predictor or fixed anchor. There is
no per-user optimizer, cross-modal fusion, or learned reliability head here.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

MODES = ("global_scalar", "global_vector", "user_scalar", "user_vector")
ARCHITECTURES = ("legacy_mlp", "coordinate_mlp")


@dataclass(frozen=True)
class V5Config:
    mode: str = "user_vector"
    architecture: str = "coordinate_mlp"
    tau_bound: float = 0.02
    temperature: float = 0.02
    coordinate_hidden: int = 32

    def __post_init__(self):
        if self.mode not in MODES or self.architecture not in ARCHITECTURES:
            raise ValueError("Unknown tau mode or architecture")
        for name in ("tau_bound", "temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.coordinate_hidden, int) or self.coordinate_hidden < 1:
            raise ValueError("coordinate_hidden must be a positive integer")


class ThresholdingModelV5(nn.Module):
    """Two independent modality heads; canonical bits are sign(enrollment mean).

legacy_mlp: [mean(512), sample_std(512)] -> 128 -> 64 -> 1 or 512.
coordinate_mlp: [sqrt(512)*mean_i, sqrt(512)*std_i, 2*p_i-1]
    -> H -> H -> 1, with weights shared across all 512 coordinates, where p_i
    is the fraction of enrollment values >= 0. Scalar mode pools these logits.
Global modes contain only learned scalar/vector parameters, as in V4.
"""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or V5Config()
        if not isinstance(self.config, V5Config):
            self.config = V5Config(**vars(self.config))
        self.output_dim = 1 if self.config.mode.endswith("scalar") else 512
        self.tau_bound = self.config.tau_bound
        if self.config.mode.startswith("global"):
            self.face_logit = nn.Parameter(torch.zeros(self.output_dim))
            self.finger_logit = nn.Parameter(torch.zeros(self.output_dim))
        else:
            for name in ("face", "finger"):
                if self.config.architecture == "legacy_mlp":
                    mlp = nn.Sequential(nn.Linear(1024, 128), nn.GELU(),
                                        nn.Linear(128, 64), nn.GELU(),
                                        nn.Linear(64, self.output_dim))
                else:
                    h = self.config.coordinate_hidden
                    mlp = nn.Sequential(nn.Linear(3, h), nn.GELU(),
                                        nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
                # Exact tau=0 baseline. Hidden-layer gradients start after the
                # output weight's first update; this does not freeze the head.
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
                setattr(self, name+"_mlp", mlp)

    def _validate(self, x, name, *, enrollment):
        reference = next(self.parameters())
        valid_shape = (x.ndim == 3 and x.shape[1:] == (5, 512)) if enrollment else (
            x.ndim in (2, 3) and x.shape[-1] == 512 and (x.ndim != 3 or x.shape[1] > 0))
        if not valid_shape or x.shape[0] < 1:
            raise ValueError(f"Invalid {name} shape; enrollment must be [B,5,512]")
        if x.dtype != reference.dtype or x.device != reference.device or not x.is_floating_point():
            raise ValueError(f"{name} must match model dtype/device")
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f"Nonfinite {name}")

    def forward(self, face_enroll, fingerprint_enroll):
        self._validate(face_enroll, "face enrollment", enrollment=True)
        self._validate(fingerprint_enroll, "fingerprint enrollment", enrollment=True)
        if face_enroll.shape != fingerprint_enroll.shape:
            raise ValueError("Enrollment batches must have the same shape")
        outputs = {}
        for name, enroll in (("face", face_enroll), ("finger", fingerprint_enroll)):
            mean = enroll.mean(1)
            std = enroll.std(1, unbiased=True)
            positive_fraction = (enroll >= 0).to(enroll.dtype).mean(1)
            if self.config.mode.startswith("global"):
                logit = getattr(self, name+"_logit")[None].expand(len(enroll), -1)
            elif self.config.architecture == "legacy_mlp":
                logit = getattr(self, name+"_mlp")(torch.cat((mean, std), dim=1))
            else:
                stats = torch.stack((math.sqrt(512)*mean, math.sqrt(512)*std,
                                     2*positive_fraction-1), dim=-1)
                logit = getattr(self, name+"_mlp")(stats).squeeze(-1)
                if self.output_dim == 1:
                    logit = logit.mean(1, keepdim=True)
            tau = (self.tau_bound*torch.tanh(logit)).expand(-1, 512)
            bits = self.apply_tau(enroll, tau)
            outputs.update({name+"_tau": tau,
                            name+"_mean": mean, name+"_std": std,
                            name+"_positive_fraction": positive_fraction,
                            name+"_target_bits": (mean >= 0).to(enroll.dtype).detach(),
                            name+"_bit_logits": bits["logits"],
                            name+"_soft_bits": bits["soft_bits"],
                            name+"_hard_bits": bits["hard_bits"]})
        return outputs

    def apply_tau(self, query_embedding, tau):
        """Binarize queries with the CLAIMANT's enrollment tau; never refit it."""
        self._validate(query_embedding, "query", enrollment=False)
        if tau.shape != (len(query_embedding), 512):
            raise ValueError("tau must have shape [B,512], including expanded scalar tau")
        if (tau.dtype != query_embedding.dtype or tau.device != query_embedding.device
                or not bool(torch.isfinite(tau).all())):
            raise ValueError("Invalid tau dtype/device/values")
        if bool((tau.abs() > self.tau_bound+1e-7).any()):
            raise ValueError("tau exceeds the configured bound")
        margin = query_embedding - (tau[:, None] if query_embedding.ndim == 3 else tau)
        logits = margin/self.config.temperature
        return {"logits": logits, "soft_bits": logits.sigmoid(),
                "hard_bits": (margin >= 0).to(torch.uint8)}
