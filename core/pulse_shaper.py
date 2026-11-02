

from __future__ import annotations

import numpy as np

from utils.dsp import fir_filter_centered
from utils.validators import validate_roll_off


class PulseShaper:
    """Genera y aplica el filtro SRRC (transmisor)."""

    def __init__(self, alpha: float, span: int = 20, sps: int = 8):
        ok, alpha_v, msg = validate_roll_off(alpha)
        if not ok:
            raise ValueError(msg)
        if span < 2 or span % 2 != 0:
            raise ValueError("El span del filtro debe ser un entero par ≥ 2.")
        if sps < 2:
            raise ValueError("Se requieren al menos 2 muestras por símbolo.")
        self.alpha = alpha_v
        self.span = span
        self.sps = sps
        self._coeffs = self._design_srrc()

    def shape(self, symbols: np.ndarray) -> np.ndarray:
        """
        1. Sobremuestreo: inserta (sps-1) ceros entre símbolos.
        2. Convolución con el SRRC y compensación del retardo de grupo.

        La salida tiene len(symbols)·sps muestras y el símbolo j queda
        centrado en la muestra j·sps.  Para no truncar las colas de los
        primeros/últimos símbolos, el llamador debe rellenar con símbolos
        nulos (ver core/link.py).
        """
        upsampled = np.zeros(len(symbols) * self.sps, dtype=complex)
        upsampled[:: self.sps] = symbols
        return fir_filter_centered(upsampled, self._coeffs)

    def get_coefficients(self) -> np.ndarray:
        return self._coeffs.copy()

    def _design_srrc(self) -> np.ndarray:
        alpha, sps, span = self.alpha, self.sps, self.span
        half = (span * sps) // 2
        t = np.arange(-half, half + 1) / sps               # unidades de T
        coeffs = np.zeros(len(t))

        for idx, ti in enumerate(t):
            if ti == 0.0:
                coeffs[idx] = (1.0 - alpha) + 4.0 * alpha / np.pi
            elif alpha > 0 and np.isclose(abs(ti), 1.0 / (4.0 * alpha), rtol=0, atol=1e-9):
                coeffs[idx] = (alpha / np.sqrt(2.0)) * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
                )
            else:
                num = (np.sin(np.pi * ti * (1.0 - alpha))
                       + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha)))
                den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
                coeffs[idx] = num / den

        coeffs /= np.sqrt(np.sum(coeffs ** 2))             # energía unitaria
        return coeffs


class MatchedFilter:
    """
    Receptor — filtro casado SRRC.

    filter(x)       : filtra a la tasa de muestreo completa (útil para el
                      diagrama del ojo).
    downsample(y)   : toma una muestra cada sps (instante óptimo, offset 0).
    filter_and_downsample(x) : ambas operaciones.
    """

    def __init__(self, alpha: float, span: int = 20, sps: int = 8):
        self._shaper = PulseShaper(alpha, span, sps)
        self.sps = sps

    def filter(self, received: np.ndarray) -> np.ndarray:
        return fir_filter_centered(received, self._shaper.get_coefficients())

    def downsample(self, filtered: np.ndarray, offset: int = 0) -> np.ndarray:
        return filtered[offset:: self.sps]

    def filter_and_downsample(self, received: np.ndarray) -> np.ndarray:
        return self.downsample(self.filter(received))
