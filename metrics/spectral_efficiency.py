"""
spectral_efficiency.py
======================
Métrica RF4a — Eficiencia espectral.

    η = R_b / B  [bit/s/Hz],   R_b = k·R_s = log2(M)·R_s,   B = R_s·(1+α)

    η = log2(M) / (1+α)

B = (1+α)·Rs es el ancho de banda de la señal PASABANDA (doble banda lateral)
con filtro coseno alzado: el espectro en banda base ocupa ±(1+α)/2·Rs.  Aplica
igual a PAM, QAM y PSK modulados en pasabanda con una sola portadora.

Referencia: Proakis & Salehi, Cap. 8-9.
"""

from __future__ import annotations

import math


def compute_spectral_efficiency(m: int, alpha: float) -> float:
    """η [bit/s/Hz] = log2(M)/(1+α)."""
    return math.log2(m) / (1.0 + alpha)


def compute_bit_rate(m: int, symbol_rate_baud: float) -> float:
    """R_b [bit/s] = log2(M)·R_s."""
    return math.log2(m) * symbol_rate_baud
