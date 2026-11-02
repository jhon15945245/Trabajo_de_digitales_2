"""
signal_power.py
===============
Métrica RF4d — Potencia de la señal modulada.

Potencia media discreta (por muestra):   P = (1/N)·Σ|x[n]|²

Para x[n] REAL en pasabanda esta fórmula ya incluye el factor ½ de la
portadora (cos² promedia ½); NO se debe dividir otra vez por 2 (error de la
versión anterior, que mostraba la mitad de la potencia real).

Relación teórica:   P_pb = P_bb / 2,    P_bb = Es / Ts

Se reporta la potencia normalizada al tiempo de símbolo (unidades de Es/Ts):

        P_norm = P_por_muestra · sps        [u.e. / Ts]

y, si se conoce Rs, P = P_norm · Rs  [u.e./s]  (u.e. = unidad de energía de Es).
Teóricamente, con Es = 1: P_bb = 1 y P_pb = 0.5 (en Es/Ts).
"""

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
