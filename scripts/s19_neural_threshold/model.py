"""Enrollment-only neural thresholds; no per-user fitting or cross-modal fusion."""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn

from scripts.s11_threshold_v5.model import ThresholdingModelV5, V5Config

FORMAT = "threshold-neural-set-v1"


@dataclass(frozen=True)
class Config:
    architecture: str = "coordinate"
    coordinate_hidden: int = 128
    set_context: int = 16
    set_hidden: int = 64
    tau_bound: float = .02
    temperature: float = .02

    def __post_init__(self):
        if self.architecture not in ("coordinate", "set"):
            raise ValueError("Unknown architecture")
        for field in ("coordinate_hidden", "set_context", "set_hidden"):
            value = getattr(self, field)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        V5Config(coordinate_hidden=self.coordinate_hidden, tau_bound=self.tau_bound,
                 temperature=self.temperature)


class SetHead(nn.Module):
    """Shared capture encoder + content attention + coordinate-shared head.

    No positional encodings: permutations of the five captures leave tau
    unchanged up to floating-point summation. Attention changes tau features,
    NEVER the fixed arithmetic-mean enrollment anchor.
    """
    def __init__(self, context, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(512, context), nn.GELU(),
                                     nn.Linear(context, context), nn.GELU())
        self.attention = nn.Linear(context, 1)
        self.head = nn.Sequential(nn.Linear(context+3, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, enrollment, local):
        captures = self.encoder(math.sqrt(512)*enrollment)
        weights = self.attention(captures).softmax(dim=1)
        context = (weights*captures).sum(1)
        features = torch.cat((local, context[:, None].expand(-1, 512, -1)), -1)
        return self.head(features).squeeze(-1), weights.squeeze(-1)


class NeuralThreshold(ThresholdingModelV5):
    def __init__(self, config=None):
        config = config or Config()
        super().__init__(V5Config(coordinate_hidden=config.coordinate_hidden,
                                 tau_bound=config.tau_bound, temperature=config.temperature))
        self.config = config
        if config.architecture == "set":
            self.face_mlp = SetHead(config.set_context, config.set_hidden)
            self.finger_mlp = SetHead(config.set_context, config.set_hidden)

    def forward(self, face_enroll, fingerprint_enroll):
        self._validate(face_enroll, "face enrollment", enrollment=True)
        self._validate(fingerprint_enroll, "fingerprint enrollment", enrollment=True)
        if face_enroll.shape != fingerprint_enroll.shape:
            raise ValueError("Enrollment batches must have equal shapes")
        out = {}
        for name, enrollment in (("face", face_enroll), ("finger", fingerprint_enroll)):
            mean, std = enrollment.mean(1), enrollment.std(1, unbiased=True)
            fraction = (enrollment >= 0).to(enrollment.dtype).mean(1)
            local = torch.stack((math.sqrt(512)*mean, math.sqrt(512)*std, 2*fraction-1), -1)
            if self.config.architecture == "set":
                raw, attention = getattr(self, name+"_mlp")(enrollment, local)
                out[name+"_attention"] = attention  # Offline diagnostic, not a public payload.
            else:
                raw = getattr(self, name+"_mlp")(local).squeeze(-1)
            tau = self.tau_bound*raw.tanh()
            bits = self.apply_tau(enrollment, tau)
            out.update({name+"_tau": tau, name+"_target_bits": (mean >= 0).float().detach(),
                        name+"_mean": mean, name+"_std": std, name+"_positive_fraction": fraction,
                        name+"_bit_logits": bits["logits"], name+"_soft_bits": bits["soft_bits"],
                        name+"_hard_bits": bits["hard_bits"]})
        return out


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(model, path, metadata):
    temporary = path.with_suffix(".tmp")
    torch.save({"format": FORMAT, "config": vars(model.config), "metadata": metadata,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format"] != FORMAT:
        raise ValueError("Expected an S19 neural checkpoint")
    model = NeuralThreshold(Config(**saved["config"])).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved["metadata"]
