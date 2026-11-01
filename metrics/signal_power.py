"""
signal_power.py
===============
Métrica RF4d — Potencia de la señal modulada.

La potencia promedio de una señal discreta se calcula como:

    P = (1/N) · Σ |x[n]|²

Para una señal real en pasabanda con modulación IQ:

    P = (1/N) · Σ (I[n]² + Q[n]²) / 2

El factor 1/2 aparece porque la señal pasabanda real tiene la mitad
de la potencia de la señal en banda base equivalente (la portadora
promedia a 1/2 en el tiempo).

Referencia: Proakis, Cap. 2.4 — Signal Power.
"""

import numpy as np


def compute_signal_power(signal: np.ndarray, is_passband: bool = False) -> float:
    """
    Potencia promedio de la señal.

    Parámetros
    ----------
    signal      : ndarray — señal (real o compleja).
    is_passband : bool    — True si la señal es pasabanda real (incluye
                            el factor 1/2 de la portadora).

    Retorna
    -------
    power : float — potencia promedio (unidades de amplitud²).
    """
    power = float(np.mean(np.abs(signal) ** 2))
    if is_passband:
        power /= 2.0    # compensar factor portadora
    return power


def compute_power_dbm(signal: np.ndarray, impedance_ohm: float = 50.0,
                      is_passband: bool = True) -> float:
    """
    Potencia de la señal en dBm, asumiendo una impedancia de referencia.

    P_dBm = 10 · log10(P_W / 1e-3)

    Parámetros
    ----------
    signal        : ndarray — señal.
    impedance_ohm : float   — impedancia de carga (default 50 Ω, estándar RF).
    is_passband   : bool    — si es señal pasabanda.

    Retorna
    -------
    p_dbm : float — potencia en dBm.

    Nota
    ----
    El resultado es simbólico/normalizado ya que la amplitud de la señal
    simulada no tiene unidades físicas reales. Sirve para comparaciones
    relativas entre esquemas.
    """
    p_linear = compute_signal_power(signal, is_passband)
    if p_linear <= 0:
        return float("-inf")
    # Interpretar amplitud² como V² → P = V²/R
    p_watts = p_linear / impedance_ohm
    return 10.0 * np.log10(p_watts / 1e-3)
