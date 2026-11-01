"""
pulse_shaper.py
===============
Transmisor — Etapa 2: Conformación de pulso (Pulse Shaping).

Implementa el filtro de coseno alzado con raíz cuadrada (SRRC — Square
Root Raised Cosine).  Se usa SRRC y no RC completo porque el filtro
casado en el receptor también es SRRC: la convolución de dos SRRC da
un RC, minimizando la ISI (Intersymbol Interference).

Parámetros del filtro:
  alpha  : factor de roll-off ∈ [0, 1]   (controla el exceso de BW)
  span   : duración del filtro en símbolos (ej. 10 → ±5 símbolos)
  sps    : muestras por símbolo (oversampling factor)

El filtro se construye evaluando la fórmula analítica del SRRC,
manejando los dos casos singulares (t=0 y t=±T/(2α)) con límites
de L'Hôpital.

Referencia: Proakis & Salehi, Cap. 9.2 — Pulse Shaping.
"""

import numpy as np


class PulseShaper:
    """
    Genera y aplica el filtro SRRC (Square-Root Raised Cosine).

    Parámetros
    ----------
    alpha : float — factor roll-off ∈ [0, 1]
    span  : int   — duración del filtro en símbolos
    sps   : int   — muestras por símbolo
    """

    def __init__(self, alpha: float, span: int = 10, sps: int = 8):
        self.alpha = alpha
        self.span  = span
        self.sps   = sps
        self._coeffs = self._design_srrc()

    # ── API pública ──────────────────────────────────────────────────────────

    def shape(self, symbols: np.ndarray) -> np.ndarray:
        """
        Inserta ceros entre símbolos (upsampling) y aplica el filtro SRRC.

        Pasos:
          1. Upsampling: inserta (sps-1) ceros entre cada símbolo.
          2. Convolución con los coeficientes SRRC.
          3. Recorta el retardo de grupo (span*sps//2 muestras de cada lado).

        Retorna
        -------
        signal : ndarray complejo, longitud ≈ n_symbols * sps.
        """
        # 1. Upsampling (interpolación por inserción de ceros)
        upsampled = np.zeros(len(symbols) * self.sps, dtype=complex)
        upsampled[:: self.sps] = symbols

        # 2. Filtrado (convolución lineal con los coeficientes SRRC)
        filtered = np.convolve(upsampled, self._coeffs)

        # 3. Compensar retardo de grupo del filtro FIR
        delay = (len(self._coeffs) - 1) // 2
        filtered = filtered[delay: delay + len(upsampled)]

        return filtered

    def get_coefficients(self) -> np.ndarray:
        """Retorna los coeficientes del filtro (útil para graficar la respuesta)."""
        return self._coeffs.copy()

    # ── Diseño del filtro SRRC ───────────────────────────────────────────────

    def _design_srrc(self) -> np.ndarray:
        """
        Calcula los coeficientes del filtro SRRC usando la fórmula analítica.

        La función SRRC h(t) en tiempo continuo es:

            h(t) = [ sin(π t/T (1-α)) + 4α t/T cos(π t/T (1+α)) ]
                   / [ π t/T (1 - (4α t/T)²) ]

        Casos especiales:
          t = 0         → h(0) = (1 - α + 4α/π)
          t = ±T/(4α)   → h(t) = (α/√2)[(1+2/π)sin(π/4α)+(1-2/π)cos(π/4α)]
        """
        alpha = self.alpha
        sps   = self.sps
        span  = self.span

        # Vector de tiempo discreto normalizado (unidades de período de símbolo T)
        n_taps = span * sps + 1                    # número de coeficientes (impar)
        t = np.arange(-(span * sps) // 2,
                       (span * sps) // 2 + 1) / sps  # t en unidades de T

        coeffs = np.zeros(n_taps)

        for idx, ti in enumerate(t):
            if ti == 0.0:
                # Límite cuando t → 0
                coeffs[idx] = (1.0 - alpha) + (4.0 * alpha / np.pi)

            elif alpha > 0 and abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-8:
                # Límite cuando t → ±1/(4α)  [singularidad de (4αt)²=1]
                a = alpha
                coeffs[idx] = (a / np.sqrt(2)) * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * a))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * a))
                )

            else:
                # Fórmula general
                num = (
                    np.sin(np.pi * ti * (1.0 - alpha))
                    + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
                )
                den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
                coeffs[idx] = num / den

        # Normalizar energía del filtro a 1
        coeffs /= np.sqrt(np.sum(coeffs ** 2))
        return coeffs


class MatchedFilter:
    """
    Receptor — filtro casado SRRC.
    El filtro casado es idéntico al filtro de transmisión (SRRC).
    La convolución TX·RX produce la respuesta de coseno alzado (RC),
    que tiene ISI nula en los instantes de muestreo.
    """

    def __init__(self, alpha: float, span: int = 10, sps: int = 8):
        # Reutilizamos PulseShaper para construir los mismos coeficientes
        self._shaper = PulseShaper(alpha, span, sps)
        self.sps = sps

    def filter_and_downsample(self, received: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro casado SRRC y hace downsampling (décima de n_sps).

        Retorna
        -------
        symbols_rx : ndarray complejo, un valor por símbolo.
        """
        coeffs = self._shaper.get_coefficients()

        # Convolución
        filtered = np.convolve(received, coeffs)

        # Compensar retardo del filtro
        delay = (len(coeffs) - 1) // 2
        filtered = filtered[delay: delay + len(received)]

        # Downsampling: tomar una muestra cada sps (instante óptimo)
        return filtered[:: self.sps]
