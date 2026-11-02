

from __future__ import annotations

import numpy as np


def compute_signal_power(signal: np.ndarray) -> float:
    """Potencia media por muestra  E[|x|²]  (sin factores ocultos)."""
    return float(np.mean(np.abs(signal) ** 2))


def compute_normalized_power(signal: np.ndarray, sps: int) -> float:
    """Potencia en unidades de Es/Ts:  E[|x|²]·sps."""
    return compute_signal_power(signal) * sps


def theoretical_passband_power(es: float) -> float:
    """P_pb teórica en Es/Ts:  Es/2 (portadora real)."""
    return es / 2.0


def power_to_db(power: float) -> float:
    """10·log10(P) (relativa a 1 u.e./Ts); -inf si P ≤ 0."""
    return float("-inf") if power <= 0 else float(10.0 * np.log10(power))
