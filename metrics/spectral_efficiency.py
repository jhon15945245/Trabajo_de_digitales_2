"""Métrica RF4a — Eficiencia espectral."""

from __future__ import annotations

import math


def compute_spectral_efficiency(m: int, alpha: float) -> float:
    """η [bit/s/Hz] = log2(M)/(1+α)."""
    return math.log2(m) / (1.0 + alpha)


def compute_bit_rate(m: int, symbol_rate_baud: float) -> float:
    """R_b [bit/s] = log2(M)·R_s."""
    return math.log2(m) * symbol_rate_baud
