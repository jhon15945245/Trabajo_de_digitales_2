"""
bandwidth.py
============
Métrica RF4c — Ancho de banda del sistema.

El ancho de banda nulo-a-nulo del filtro coseno alzado (RC) es:

    B = R_s · (1 + α)   [Hz]

donde R_s = 1/T_s es la tasa de símbolo.

Como en la simulación trabajamos con frecuencia normalizada
(R_s = 1 símbolo/muestra antes de oversample), el ancho de banda
normalizado respecto a la frecuencia de símbolo es simplemente:

    B_norm = (1 + α)   [en unidades de Rs]

Si el usuario provee la tasa de símbolo en baudios, se puede
calcular el ancho de banda absoluto en Hz.

Referencia: Proakis, Cap. 9.2 — Raised Cosine Spectrum.
"""


def compute_bandwidth_normalized(alpha: float) -> float:
    """
    Ancho de banda normalizado (en unidades de la tasa de símbolo Rs).

    B_norm = 1 + α

    Parámetros
    ----------
    alpha : float — factor roll-off ∈ [0, 1].

    Retorna
    -------
    B_norm : float — ancho de banda normalizado en [Rs].
    """
    return 1.0 + alpha


def compute_bandwidth_hz(alpha: float, symbol_rate_baud: float) -> float:
    """
    Ancho de banda absoluto en Hz, dado el roll-off y la tasa de símbolo.

    B = Rs · (1 + α)   [Hz]

    Parámetros
    ----------
    alpha            : float — factor roll-off ∈ [0, 1].
    symbol_rate_baud : float — tasa de símbolo en baudios (símbolos/s).

    Retorna
    -------
    B : float — ancho de banda en Hz.
    """
    return symbol_rate_baud * (1.0 + alpha)
