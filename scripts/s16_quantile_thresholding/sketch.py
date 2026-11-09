"""Exact public BCH syndrome map; matches the existing C++ implementation."""
from functools import lru_cache
import numpy as np
import torch


@lru_cache(maxsize=16)
def sketch_matrix(t):
    """Exact binary linear map for BCH.cpp compute_syndrome/pack_syndromes.

Bit i represents x**i; shortened-away coefficients are zero. GF polynomial
0x409, GF(2**10). Columns are S1..S2t, each MSB first. At t=32 this is 640
stored syndrome bits, NOT 640 independent constraints (nor a complete FE P).
"""
    if not isinstance(t, int) or not 1 <= t <= 255:
        raise ValueError("BCH requires integer t in [1,255]")
    alpha, value = [], 1
    for _ in range(1023):
        alpha.append(value)
        value <<= 1
        if value & 1024:
            value ^= 0x409
    exponents = np.arange(512)[:, None]*np.arange(1, 2*t+1)[None, :] % 1023
    symbols = np.asarray(alpha)[exponents]
    bits = ((symbols[..., None] >> np.arange(9, -1, -1)) & 1).reshape(512, 20*t)
    return torch.from_numpy(bits.astype(np.float32))


def public_sketch(target, t):
    if target.ndim != 2 or target.shape[1] != 512 or not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("Sketch expects binary [N,512] targets")
    # Exact integer sums <=512 in float32, before parity reduction. No AMP.
    return (target.float() @ sketch_matrix(t).to(target.device)).remainder(2).detach()
