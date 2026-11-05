"""Frozen original tau model with monotone distance-to-reliability calibration.

The original Thresholding Model.py and its checkpoint format are unchanged.
This variant has four trainable scalars: slope/intercept for each modality.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


_PATH = Path(__file__).resolve().parents[1] / "Thresholding Model.py"
_SPEC = importlib.util.spec_from_file_location("_distance_reliability_original", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load original threshold model: {_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)


class MonotoneDistanceHead(nn.Module):
    """logit = positive_slope * abs(embedding - tau) / train_scale + intercept.

    Accepts the original head's concat(embedding, distance, context) interface.
    Uses ONLY its distance slice. Slope/intercept are shared across all 512 bits;
    therefore the distance ordering is retained within each modality.
    """
    def __init__(self):
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.intercept = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("distance_scale", torch.tensor(1.0))

    @property
    def positive_slope(self):
        return F.softplus(self.raw_slope) + 1e-6

    def forward(self, combined):
        if combined.ndim != 2 or combined.shape[1] != 1088:
            raise ValueError("Expected concat(512-D embedding, 512-D distance, 64-D context)")
        distance = combined[:, 512:1024]
        return self.positive_slope * (distance / self.distance_scale) + self.intercept

    @torch.no_grad()
    def initialize_from_training(self, mean_distance, agreement_rate):
        if not math.isfinite(mean_distance) or mean_distance <= 0:
            raise ValueError("Training distance scale must be positive and finite")
        if not math.isfinite(agreement_rate) or not 0 < agreement_rate < 1:
            raise ValueError("Training pairs must contain both agreeing and disagreeing bits")
        self.distance_scale.fill_(mean_distance)
        self.raw_slope.fill_(math.log(math.expm1(1.0)))
        # Initialize the logit at mean distance to the TRAIN-only marginal logit.
        self.intercept.fill_(math.log(agreement_rate / (1 - agreement_rate)) - float(self.positive_slope))

    @torch.no_grad()
    def settings(self):
        return {"normalized_positive_slope": float(self.positive_slope),
                "distance_scale_from_train": float(self.distance_scale),
                "slope_in_original_distance_units": float(self.positive_slope / self.distance_scale),
                "intercept": float(self.intercept)}


class DistanceThresholdingModel(_BASE.ThresholdingModel):
    """Only reliability/inference modes are supported; the entire tau path is fixed."""
    checkpoint_format = "multimodal-thresholding-distance-reliability-v1"

    def __init__(self, config=None):
        super().__init__(config)
        self.face_reliability_head = MonotoneDistanceHead()
        self.fingerprint_reliability_head = MonotoneDistanceHead()
        self.set_training_stage("inference")

    def set_training_stage(self, stage):
        if stage not in {"reliability", "inference"}:
            raise ValueError("Distance calibration permits reliability or inference only; tau must remain frozen")
        return super().set_training_stage(stage)

    @classmethod
    def from_threshold_checkpoint(cls, path, device="cpu"):
        original = _BASE.ThresholdingModel.load_checkpoint(path, device="cpu")
        model = cls(original.config)
        trunk = {name: value for name, value in original.state_dict().items()
                 if "reliability_head" not in name}
        result = model.load_state_dict(trunk, strict=False)
        expected_missing = {name for name in model.state_dict() if "reliability_head" in name}
        if result.unexpected_keys or set(result.missing_keys) != expected_missing:
            raise RuntimeError("The original tau path does not match this model")
        return model.to(device).set_training_stage("inference")

    @torch.no_grad()
    def head_settings(self):
        return {modality: getattr(self, f"{modality}_reliability_head").settings()
                for modality in ("face", "fingerprint")}
