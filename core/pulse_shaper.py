"""
pulse_shaper.py
===============
Transmisor — Etapa 2: Conformación de pulso (pulse shaping) y filtro casado.

Se usa el filtro de raíz de coseno alzado (SRRC).  TX y RX usan el mismo
SRRC, de modo que la cascada TX⊛RX es un coseno alzado (RC) que cumple el
criterio de Nyquist (ISI nula en los instantes de muestreo) y el filtro del
receptor es el filtro casado (máxima SNR en el instante de decisión).

Parámetros
  alpha : factor de roll-off, 0 ≤ α ≤ 1  (ancho de banda = (1+α)·Rs)
  span  : duración del filtro en símbolos
  sps   : muestras por símbolo

SRRC en tiempo discreto (t en unidades de T, t = n/sps):

    h(t) = [ sin(π t (1-α)) + 4α t cos(π t (1+α)) ] / [ π t (1 - (4α t)²) ]

  t = 0        →  h(0) = 1 - α + 4α/π
  t = ±1/(4α)  →  h = (α/√2)·[ (1+2/π) sin(π/(4α)) + (1-2/π) cos(π/(4α)) ]

Los coeficientes se normalizan a energía unitaria (Σh² = 1), de modo que el
filtro no altera la energía de símbolo: Es = (potencia por muestra)·sps.

Referencia: Proakis & Salehi, Cap. 9 (Raised cosine / square-root RC).
"""

from __future__ import annotations

import numpy as np

from utils.dsp import filter_decimate_centered, fir_filter_centered, upsample_filter_centered
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
        return upsample_filter_centered(np.asarray(symbols, dtype=complex),
                                        self._coeffs, self.sps)

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

    def sample_symbols(self, received: np.ndarray, start: int, count: int) -> np.ndarray:
        """
        Filtro casado evaluado SOLO en los instantes de símbolo start, start+sps, …
        (count muestras).  Igual a filter(received)[start::sps][:count] pero sin
        calcular las muestras intermedias (más rápido; no sirve para el ojo).
        """
        return filter_decimate_centered(received, self._shaper.get_coefficients(),
                                        self.sps, start, count)

    def filter_and_downsample(self, received: np.ndarray) -> np.ndarray:
        return self.downsample(self.filter(received))
