"""
modulator.py
============
Transmisor — Etapa 3: Modulación en pasabanda (RF).

Convierte la señal en banda base compleja s(t) = I(t) + jQ(t) en una
señal real en pasabanda x(t) mediante:

    x(t) = I(t)·cos(2π fc t) − Q(t)·sin(2π fc t)

donde fc es la frecuencia portadora (normalizada).

Esta representación corresponde a la modulación IQ exacta que usa
cualquier sistema real de radiocomunicaciones.

Referencia: Proakis & Salehi, Cap. 2.5 — Bandpass Signal Representation.
"""

import numpy as np
from utils.constants import CARRIER_FREQ_NORMALIZED


class Modulator:
    """
    Modulación en pasabanda a partir de la señal en banda base.

    Parámetros
    ----------
    fc_normalized : float
        Frecuencia portadora normalizada respecto a la frecuencia de muestreo
        (fc / fs).  Valor típico: 0.25 → fc = fs/4.
    """

    def __init__(self, fc_normalized: float = CARRIER_FREQ_NORMALIZED):
        self.fc_norm = fc_normalized

    def modulate(self, baseband_signal: np.ndarray) -> np.ndarray:
        """
        Modula la señal en banda base compleja a pasabanda real.

        Parámetros
        ----------
        baseband_signal : ndarray complejo
            Señal en banda base después de la conformación de pulso.

        Retorna
        -------
        rf_signal : ndarray real
            Señal en pasabanda lista para transmitir por el canal.
        """
        n = np.arange(len(baseband_signal))

        # Portadora coseno para la componente I (en fase)
        carrier_i = np.cos(2.0 * np.pi * self.fc_norm * n)

        # Portadora seno para la componente Q (en cuadratura)
        carrier_q = np.sin(2.0 * np.pi * self.fc_norm * n)

        # Señal en pasabanda real
        rf_signal = (
            np.real(baseband_signal) * carrier_i
            - np.imag(baseband_signal) * carrier_q
        )
        return rf_signal


class Demodulator:
    """
    Demodulación coherente: recupera la señal en banda base a partir
    de la señal de pasabanda recibida.

    El receptor multiplica la señal recibida por cos y −sin de la portadora
    (asume sincronización perfecta de fase y frecuencia) y luego aplica
    un filtro pasa-bajas ideal para eliminar el armónico en 2fc.

    Proceso:
      r(t) · cos(2π fc t) = I(t)/2 + I(t)·cos(4π fc t)/2 − Q(t)·sin(4π fc t)/2
      → filtro PB → I(t)/2

      r(t) · (−sin(2π fc t)) = Q(t)/2 + términos en 2fc
      → filtro PB → Q(t)/2

    El factor 1/2 se cancela normalizando por 2 al final.
    """

    def __init__(self, fc_normalized: float = CARRIER_FREQ_NORMALIZED):
        self.fc_norm = fc_normalized

    def demodulate(self, rf_signal: np.ndarray, sps: int) -> np.ndarray:
        """
        Demodula la señal de pasabanda y retorna la señal en banda base.

        Parámetros
        ----------
        rf_signal : ndarray real  — señal recibida después del canal.
        sps       : int           — muestras por símbolo (para el filtro PB).

        Retorna
        -------
        baseband_rx : ndarray complejo — señal en banda base recuperada.
        """
        n = np.arange(len(rf_signal))

        # Mezcla con portadora local (demodulación coherente)
        i_mixed = rf_signal * np.cos(2.0 * np.pi * self.fc_norm * n)
        q_mixed = rf_signal * (-np.sin(2.0 * np.pi * self.fc_norm * n))

        # Filtro pasa-bajas FIR de ventana rectangular simple
        # Ancho de banda de corte: 1/(2·sps) normalizado
        i_filtered = self._lowpass_filter(i_mixed, sps)
        q_filtered = self._lowpass_filter(q_mixed, sps)

        # Compensar el factor 1/2 de la demodulación coherente
        baseband_rx = 2.0 * (i_filtered + 1j * q_filtered)
        return baseband_rx

    @staticmethod
    def _lowpass_filter(signal: np.ndarray, sps: int) -> np.ndarray:
        """
        Filtro pasa-bajas FIR de ventana sinc.
        Frecuencia de corte normalizada: fc_cut = 1/(2·sps).
        Longitud: 4·sps + 1 coeficientes.

        Implementado desde cero: convolución con la ventana sinc.
        """
        fc_cut = 1.0 / (2.0 * sps)      # frecuencia de corte normalizada
        n_taps = 4 * sps + 1
        n_vec  = np.arange(-(n_taps // 2), n_taps // 2 + 1)

        # Coeficientes sinc con ventana de Hann para reducir lóbulos laterales
        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(
                n_vec == 0,
                2.0 * fc_cut,
                np.sin(2.0 * np.pi * fc_cut * n_vec) / (np.pi * n_vec),
            )

        # Ventana de Hann
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_taps) / (n_taps - 1))
        h *= window
        h /= np.sum(h)    # normalizar ganancia DC = 1

        # Convolución con compensación de retardo
        filtered = np.convolve(signal, h)
        delay = (n_taps - 1) // 2
        return filtered[delay: delay + len(signal)]
