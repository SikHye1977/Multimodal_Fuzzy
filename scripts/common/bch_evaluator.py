"""Evaluate exact 512-bit recovery with the existing C++ BCH implementation.

This module depends on NumPy and the BCH sources, not on Fusion models or
training scripts. Importing it does not compile or run anything; construction
builds the library in the caller's output directory.

Outcomes are 0 (no candidate), 1 (exact enrolled bits), or 2 (wrong candidate).
This evaluates bit recovery, not cryptographic key generation or FE security.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .paths import ROOT


class BCHEvaluator:
    """Compile the existing C++ implementation into this experiment's directory."""

    def __init__(self, directory, t):
        self.t = t
        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError("g++ is required for exact BCH bit-recovery evaluation; nothing is installed automatically")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Retain the existing C bridge and ABI; neither imports a Fusion model.
        library = directory / "bch.so"
        command = [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I", str(ROOT / "BCH"),
                   str(ROOT / "scripts/common/bch_bridge.cpp"), str(ROOT / "BCH/BCH.cpp"), "-o", str(library)]
        built = subprocess.run(command, capture_output=True, text=True)
        (directory / "compiler.log").write_text(built.stdout + built.stderr, encoding="utf-8")
        if built.returncode:
            raise RuntimeError(f"BCH compilation failed; see {directory / 'compiler.log'}")
        self.library = ctypes.CDLL(str(library))
        self.call = self.library.fusion_bch_recover
        ptr = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.call.argtypes = (ctypes.c_int, ptr, ptr, ctypes.c_size_t, ptr)
        self.call.restype = ctypes.c_int

    def recover(self, enrolled, query):
        enrolled, query = np.asarray(enrolled), np.asarray(query)
        if enrolled.shape != query.shape or enrolled.ndim != 2 or enrolled.shape[1] != 512:
            raise ValueError("BCH expects matching (N,512) bit arrays")
        if not np.isin(enrolled, (0, 1)).all() or not np.isin(query, (0, 1)).all():
            raise ValueError("BCH inputs must be binary")
        e = np.ascontiguousarray(np.packbits(enrolled.astype(np.uint8), axis=1, bitorder="big"))
        q = np.ascontiguousarray(np.packbits(query.astype(np.uint8), axis=1, bitorder="big"))
        outcomes = np.empty(len(e), np.uint8)
        if self.call(self.t, e, q, len(e), outcomes):
            raise RuntimeError("C++ BCH evaluator rejected its inputs")
        return outcomes
