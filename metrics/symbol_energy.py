

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
