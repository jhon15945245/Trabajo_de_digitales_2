"""
symbol_energy.py
================
Métrica RF4b — Energía de símbolo (Es) y de bit (Eb).

Energía de símbolo
  • Teórica (de la constelación):   Es = (1/M)·Σ|s_i|²   (=1 por normalización)
  • Medida (de la señal transmitida): el filtro SRRC tiene energía unitaria y
    cada símbolo dura sps muestras, de modo que

        Es_medida = P_bb · sps,     P_bb = E|s[n]|²  (envolvente compleja)

    Si el sistema está bien implementado ambas coinciden (salvo la desviación
    estadística de la muestra de símbolos).

Energía de bit:   Eb = Es / log2(M).

Unidades: energía normalizada (u.e.), con Es teórica = 1 u.e.

Referencia: Proakis & Salehi, Cap. 4.
"""

from __future__ import annotations

import math

import numpy as np


def compute_symbol_energy(constellation: np.ndarray) -> float:
    """Es teórica: media de |s_i|² sobre los M puntos."""
    return float(np.mean(np.abs(constellation) ** 2))


def compute_symbol_energy_measured(baseband_signal: np.ndarray, sps: int) -> float:
    """Es medida sobre la señal en banda base conformada: P_bb · sps."""
    return float(np.mean(np.abs(baseband_signal) ** 2)) * sps


def compute_bit_energy(es: float, m: int) -> float:
    """Eb = Es / log2(M)."""
    return es / math.log2(m)
