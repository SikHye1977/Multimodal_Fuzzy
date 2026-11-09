"""Gaussian-CDF quantile helpers preserving c=1[enrollment mean >= 0].

This is a MODEL-BASED adaptation, not a zero-leakage guarantee for correlated
512-D embeddings. Population CDF and query-noise parameters use training people
only. C1/C2 integrate likelihood over the STORED code interval, not over a hidden
unquantized helper. Their inference is a global lookup followed by thresholding.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn

from .protocol import MODEL_NAMES

FORMAT = "threshold-quantile-gaussian-v1"


@dataclass(frozen=True)
class QuantileConfig:
    helper_bits: int = 0  # 0 denotes the float32 diagnostic, not a zero-bit payload.
    tau_bound: float = .02
    temperature: float = .02
    quadrature: int = 16
    cdf_epsilon: float = 1e-6
    min_interval_mass: float = 1e-4

    def __post_init__(self):
        if self.helper_bits not in (0, 4, 8) or self.quadrature < 4:
            raise ValueError("Use float32/4-bit/8-bit helpers and quadrature >=4")
        if not 0 < self.cdf_epsilon < self.min_interval_mass < .5:
            raise ValueError("Invalid CDF numerical limits")
        if not all(math.isfinite(x) and x > 0 for x in (self.tau_bound, self.temperature)):
            raise ValueError("Bounds and temperature must be positive and finite")


def pack_codes(code, bits):
    if bits not in (4, 8) or code.ndim != 2 or code.shape[1] != 512:
        raise ValueError("Expected [N,512] finite code indices")
    if code.dtype != torch.uint8 or not bool((code.to(torch.int16) < 2**bits).all()):
        raise ValueError("Invalid quantile codes")
    return code.clone() if bits == 8 else (code[:, ::2] << 4) | code[:, 1::2]


def unpack_codes(packed, bits):
    width = 512 if bits == 8 else 256
    if bits not in (4, 8) or packed.ndim != 2 or packed.shape[1] != width or packed.dtype != torch.uint8:
        raise ValueError("Invalid quantile payload")
    if bits == 8:
        return packed.clone()
    return torch.stack((packed >> 4, packed & 15), -1).reshape(-1, 512)


class QuantileThreshold(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or QuantileConfig()
        levels = 2**self.config.helper_bits if self.config.helper_bits else 0
        for name in MODEL_NAMES.values():
            for key, initial in (("location", 0.), ("scale", 1.), ("slope", 1.),
                                 ("intercept", 0.), ("noise_var", 1.), ("p0", .5)):
                self.register_buffer(name+"_"+key, torch.full((512,), initial, dtype=torch.float64))
            self.register_buffer(name+"_active", torch.ones(512, dtype=torch.bool))
            self.register_buffer(name+"_table", torch.zeros(512, levels, dtype=torch.float32))
        self.register_buffer("fitted", torch.tensor(False))

    def params(self, name):
        return {k: getattr(self, name+"_"+k) for k in
                ("location", "scale", "slope", "intercept", "noise_var", "p0", "active")}

    @torch.no_grad()
    def fit(self, bank, plan):
        if plan["split"] != "train":
            raise ValueError("Population statistics MUST use the training split")
        diagnostics = {}
        for m, name in MODEL_NAMES.items():
            data = bank[m]
            means = data["enrollment"].double().mean(1)
            queries, mask = data["queries"].double(), data["query_mask"]
            counts = mask.sum(1).double()
            # Equal registration weight, independent of number of queries.
            y_mean = (queries*mask[..., None]).sum(1)/counts[:, None]
            y_second = (queries.square()*mask[..., None]).sum(1)/counts[:, None]
            identity = data["identity"]
            _, inverse, frequencies = torch.unique(identity, return_inverse=True, return_counts=True)
            weights = frequencies[inverse].double().reciprocal()
            weights /= weights.sum()
            average = lambda value: (value*weights[:, None]).sum(0)
            location = average(means)
            variance = average((means-location).square())
            scale = variance.clamp_min(1e-10).sqrt()
            ey = average(y_mean)
            slope_raw = average((means-location)*(y_mean-ey))/variance.clamp_min(1e-10)
            slope = slope_raw.clamp_min(1e-4)
            intercept = ey-slope*location
            fitted_y = slope*means+intercept
            noise_var = average(y_second-2*fitted_y*y_mean+fitted_y.square()).clamp_min(1e-10)
            p0 = torch.special.ndtr(-location/scale)
            active = ((p0 > self.config.min_interval_mass) &
                      (p0 < 1-self.config.min_interval_mass) & (variance > 1e-10) & (slope_raw > 0))
            values = dict(location=location, scale=scale, slope=slope, intercept=intercept,
                          noise_var=noise_var, p0=p0, active=active)
            for key, value in values.items():
                getattr(self, name+"_"+key).copy_(value)
            if self.config.helper_bits:
                self.build_table(name)
            diagnostics[m] = {"registrations": len(means), "identities": len(identity.unique()),
                "people": len({r["person"] for r in plan["modalities"][m]}),
                "inactive_coordinates": int((~active).sum()), "p0": p0.cpu().tolist(),
                "slope": slope.cpu().tolist(), "noise_std": noise_var.sqrt().cpu().tolist(),
                "cdf_model": "Gaussian on five-capture means; estimated, not assumed exact",
                "noise_model": "query = slope * enrollment_mean + intercept + homoscedastic Gaussian noise"}
        self.fitted.fill_(True)
        return diagnostics

    def candidate_means(self, w, name):
        """w shape [512,...]; return query-space sibling means for c=0 and 1."""
        p = self.params(name)
        shape = (512,)+(1,)*(w.ndim-1)
        p0 = p["p0"].reshape(shape)
        eps = self.config.cdf_epsilon
        result = []
        for probability in (p0*w, p0+(1-p0)*w):
            x = p["location"].reshape(shape)+p["scale"].reshape(shape)*torch.special.ndtri(probability.clamp(eps, 1-eps))
            result.append(p["slope"].reshape(shape)*x+p["intercept"].reshape(shape))
        return result

    @torch.no_grad()
    def build_table(self, name):
        """MAP boundary of bin-averaged likelihoods, clipped to common tau bound.

        Ordered sibling supports and common Gaussian variance imply a monotone
        likelihood ratio. Gauss-Legendre quadrature approximates each integral;
        40 bisections find its boundary. Global tables are public model state.
        """
        levels = 2**self.config.helper_bits
        p = self.params(name)
        nodes, weights = np.polynomial.legendre.leggauss(self.config.quadrature)
        nodes = torch.as_tensor((nodes+1)/2, device=p["p0"].device)
        log_weights = torch.as_tensor(weights/2, device=nodes.device).log()
        prior = p["p0"].clamp(self.config.cdf_epsilon, 1-self.config.cdf_epsilon)
        log_prior_ratio = torch.log1p(-prior)-prior.log()
        table = getattr(self, name+"_table")
        for start in range(0, levels, 32):
            codes = torch.arange(start, min(start+32, levels), device=nodes.device)
            w = ((codes[None, :, None]+nodes[None, None, :])/levels).expand(512, -1, -1)
            mean0, mean1 = self.candidate_means(w, name)
            low = torch.full(mean0.shape[:2], -self.config.tau_bound, dtype=torch.float64, device=nodes.device)
            high = -low
            for _ in range(40):
                middle = (low+high)/2
                l0 = torch.logsumexp(-(middle[..., None]-mean0).square()/(2*p["noise_var"][:, None, None])+log_weights, -1)
                l1 = torch.logsumexp(-(middle[..., None]-mean1).square()/(2*p["noise_var"][:, None, None])+log_weights, -1)
                positive = (l1-l0+log_prior_ratio[:, None]) >= 0
                high = torch.where(positive, middle, high)
                low = torch.where(positive, low, middle)
            tau = (low+high)/2
            table[:, start:start+len(codes)].copy_(torch.where(p["active"][:, None], tau, 0).float())

    def helper(self, means, name):
        p = self.params(name)
        cdf = torch.special.ndtr((means.double()-p["location"])/p["scale"])
        p0 = p["p0"].clamp(self.config.cdf_epsilon, 1-self.config.cdf_epsilon)
        w = torch.where(means >= 0, (cdf-p0)/(1-p0), cdf/p0)
        w = torch.where(p["active"], w, 0).clamp(0., 1.-self.config.cdf_epsilon)
        if not self.config.helper_bits:
            return w.float()  # Exact persisted representation for C0.
        return torch.floor(w*(2**self.config.helper_bits)).to(torch.uint8)

    def tau_from_payload(self, payload, name):
        if self.config.helper_bits:
            codes = unpack_codes(payload, self.config.helper_bits)
            table = getattr(self, name+"_table")
            return table[torch.arange(512, device=table.device)[None], codes.long()]
        if payload.dtype != torch.float32 or payload.ndim != 2 or payload.shape[1] != 512:
            raise ValueError("C0 payload must be float32 [N,512]")
        if not bool(torch.isfinite(payload).all()) or not bool(((payload >= 0) & (payload < 1)).all()):
            raise ValueError("Invalid float helper")
        p = self.params(name)
        mean0, mean1 = self.candidate_means(payload.double().T, name)
        prior = p["p0"].clamp(self.config.cdf_epsilon, 1-self.config.cdf_epsilon)
        correction = p["noise_var"][:, None]*(prior.log()-torch.log1p(-prior))[:, None]/(mean1-mean0).clamp_min(1e-12)
        tau = ((mean0+mean1)/2+correction).T.clamp(-self.config.tau_bound, self.config.tau_bound)
        return torch.where(p["active"], tau, 0).float()

    def forward(self, face_enroll, fingerprint_enroll):
        if not bool(self.fitted):
            raise RuntimeError("Fit population parameters on training people first")
        out = {}
        for name, x in (("face", face_enroll), ("finger", fingerprint_enroll)):
            if (x.ndim != 3 or x.shape[1:] != (5, 512) or x.dtype != torch.float32
                    or x.device != self.fitted.device or not bool(torch.isfinite(x).all())):
                raise ValueError("Expected finite float32 [N,5,512] enrollment on model device")
            mean = x.mean(1)
            code = self.helper(mean, name)
            payload = pack_codes(code, self.config.helper_bits) if self.config.helper_bits else code
            tau = self.tau_from_payload(payload, name)
            w = (code.float()+.5)/(2**self.config.helper_bits) if self.config.helper_bits else code
            logits = (x-tau[:, None])/self.config.temperature
            out.update({name+"_tau": tau, name+"_target_bits": (mean >= 0).float(),
                        name+"_payload": payload, name+"_public_aux": w,
                        name+"_coordinate_aux": w, name+"_bit_logits": logits,
                        name+"_soft_bits": logits.sigmoid(), name+"_hard_bits": (logits >= 0).byte()})
        return out

    def storage(self):
        return {m: {"payload_bits": 512*(self.config.helper_bits or 32),
                    "encoding": f"{self.config.helper_bits}-bit packed quantile codes" if self.config.helper_bits else "float32 quantile helpers",
                    "derived_tau_stored": False} for m in MODEL_NAMES}

    def checkpoint(self, metadata):
        return {"format": FORMAT, "config": asdict(self.config), "metadata": metadata,
                "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}
