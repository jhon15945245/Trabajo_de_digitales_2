"""
spectral_efficiency.py
======================
Métrica RF4a — Eficiencia espectral.

La eficiencia espectral η mide cuántos bits por segundo se transmiten
por cada Hz de ancho de banda:

    η = R_b / B   [bits/s/Hz]

donde:
  R_b = k · R_s = log2(M) · (1/T_s)   — tasa de bits
  B   = R_s · (1 + α)                  — ancho de banda de Nyquist
                                           con filtro coseno alzado

Sustituyendo:
    η = log2(M) / (1 + α)   [bits/s/Hz]

Para α=0 (filtro ideal): η = log2(M)  (límite de Shannon para PAM).
Para α=1 (roll-off máximo): η = log2(M) / 2.

Referencia: Proakis, Cap. 9.3 — Spectral Efficiency.
"""


def compute_spectral_efficiency(m: int, alpha: float) -> float:
    """
    Calcula la eficiencia espectral del sistema.

    Parámetros
    ----------
    m     : int   — orden de modulación (número de símbolos).
    alpha : float — factor roll-off ∈ [0, 1].

    Retorna
    -------
    eta : float — eficiencia espectral en bits/s/Hz.
    """
    import math
    k = math.log2(m)              # bits por símbolo
    eta = k / (1.0 + alpha)       # bits/s/Hz
    return eta
