"""
ber_calculator.py
=================
BER teórica (fórmulas analíticas) y BER simulada (conteo de errores).

Q(x) = P(X > x), X ~ N(0,1)  =  ½·erfc(x/√2)   (erfc de la librería estándar `math`).

BER teórica en AWGN con codificación Gray (k = log2 M, γb = Eb/N0 lineal):

  M-PAM (exacta para M=2, aprox. Gray para M>2):
      Pb ≈ [2(M-1)/(M·k)] · Q( √( 6·k·γb / (M²-1) ) )

  M-QAM cuadrada:
      Pb ≈ [4(1-1/√M)/k] · Q( √( 3·k·γb / (M-1) ) )
      (cada eje es un √M-PAM; la suma de las dos ramas da el factor 4(1-1/√M)/k)

  M-PSK:
      M = 2 (BPSK, exacta):  Pb = Q( √(2·γb) )
      M ≥ 4 (aproximación, solo errores a símbolos vecinos, Gray):
      Pb ≈ (2/k) · Q( √(2·k·γb) · sin(π/M) )

Para M=4: 4-QAM y 4-PSK dan exactamente la misma Pb = Q(√(2γb)), porque son la
misma constelación rotada 45°.

Referencia: Proakis & Salehi, Cap. 4 y 8; Simon & Alouini.
"""

from __future__ import annotations

from math import erfc

import numpy as np


def q_function(x):
    """Función Q(x) = ½·erfc(x/√2) para escalares o arreglos."""
    arr = np.asarray(x, dtype=float)
    out = 0.5 * np.vectorize(lambda v: erfc(v / np.sqrt(2.0)), otypes=[float])(arr)
    return float(out) if np.isscalar(x) else out


class BERCalculator:
    """BER simulada y teórica para M-PAM, M-QAM y M-PSK."""

    # ── BER simulada ─────────────────────────────────────────────────────────

    @staticmethod
    def count_errors(bits_tx: np.ndarray, bits_rx: np.ndarray) -> tuple[int, int]:
        """Retorna (errores, bits comparados)."""
        n_total = min(len(bits_tx), len(bits_rx))
        if n_total == 0:
            return 0, 0
        n_errors = int(np.sum(bits_tx[:n_total] != bits_rx[:n_total]))
        return n_errors, n_total

    @staticmethod
    def ber_simulated(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
        """BER = errores / bits  (0.0 si no hay bits)."""
        n_errors, n_total = BERCalculator.count_errors(bits_tx, bits_rx)
        return n_errors / n_total if n_total else 0.0

    # ── BER teórica ──────────────────────────────────────────────────────────

    @staticmethod
    def ber_theoretical(scheme: str, m: int, eb_n0_db) -> np.ndarray:
        """BER teórica para Eb/N0 en dB (escalar o arreglo)."""
        eb_n0_db = np.asarray(eb_n0_db, dtype=float)
        eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)          # dB → lineal
        k = np.log2(m)
        dispatch = {
            "M-PAM": BERCalculator._ber_pam,
            "M-QAM": BERCalculator._ber_qam,
            "M-PSK": BERCalculator._ber_psk,
        }
        if scheme not in dispatch:
            raise ValueError(f"Esquema no reconocido: {scheme}")
        return dispatch[scheme](m, k, eb_n0_lin)

    @staticmethod
    def _ber_pam(m, k, eb_n0_lin):
        arg = np.sqrt(6.0 * k * eb_n0_lin / (m ** 2 - 1))
        return (2.0 * (m - 1) / (m * k)) * q_function(arg)

    @staticmethod
    def _ber_qam(m, k, eb_n0_lin):
        sqrt_m = np.sqrt(m)
        arg = np.sqrt(3.0 * k * eb_n0_lin / (m - 1))
        return (4.0 * (1.0 - 1.0 / sqrt_m) / k) * q_function(arg)

    @staticmethod
    def _ber_psk(m, k, eb_n0_lin):
        if m == 2:                                      # BPSK: exacta
            return q_function(np.sqrt(2.0 * eb_n0_lin))
        arg = np.sqrt(2.0 * k * eb_n0_lin) * np.sin(np.pi / m)
        return (2.0 / k) * q_function(arg)
