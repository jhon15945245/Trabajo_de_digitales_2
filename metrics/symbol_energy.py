"""
symbol_energy.py
================
Métrica RF4b — Energía de símbolo (Es) y energía de bit (Eb).

La energía promedio de símbolo Es se calcula directamente
sobre la constelación:

    Es = (1/M) · Σ |s_i|²    (media sobre los M símbolos)

La energía de bit se obtiene dividiendo por k = log2(M):

    Eb = Es / log2(M)

Para constelaciones normalizadas (energía unitaria), Es = 1 por
construcción; el módulo calculado acá confirma esto experimentalmente
y muestra el valor si el usuario no normalizó.

Referencia: Proakis, Cap. 5 — Signal Energy.
"""

import numpy as np


def compute_symbol_energy(constellation: np.ndarray) -> float:
    """
    Energía promedio de símbolo de la constelación.

    Parámetros
    ----------
    constellation : ndarray complejo — todos los símbolos del alfabeto.

    Retorna
    -------
    Es : float — energía promedio por símbolo (adimensional, normalizada).
    """
    return float(np.mean(np.abs(constellation) ** 2))


def compute_bit_energy(constellation: np.ndarray, m: int) -> float:
    """
    Energía promedio por bit.

    Parámetros
    ----------
    constellation : ndarray complejo — constelación.
    m             : int — orden de modulación.

    Retorna
    -------
    Eb : float — energía por bit = Es / log2(M).
    """
    es = compute_symbol_energy(constellation)
    k  = np.log2(m)
    return es / k
