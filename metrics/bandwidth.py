"""
bandwidth.py
============
Métrica RF4c — Ancho de banda.

Con filtro de coseno alzado, el ancho de banda de la señal PASABANDA
(nulo a nulo, ambos lados de la portadora) es

        B = R_s · (1 + α)

En unidades normalizadas B/Rs = 1+α.  Si se conoce Rs [baudios], B [Hz] = Rs·(1+α).
(En banda base la señal ocupa ±B/2.)

Referencia: Proakis & Salehi, Cap. 9 (espectro de coseno alzado).
"""

from __future__ import annotations


def compute_bandwidth_normalized(alpha: float) -> float:
    """B/Rs = 1+α."""
    return 1.0 + alpha


def compute_bandwidth_hz(alpha: float, symbol_rate_baud: float) -> float:
    """B [Hz] = Rs·(1+α)."""
    return symbol_rate_baud * (1.0 + alpha)
