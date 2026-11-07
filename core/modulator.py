"""Modulación y demodulación en pasabanda (modulación analógica IQ)."""

from __future__ import annotations

import numpy as np

from utils.constants import (
    CARRIER_FREQ_NORMALIZED, LPF_MARGIN_RS, LPF_TAPS_PER_SYMBOL,
)
from utils.dsp import fir_filter_centered


class Modulator:
    """Modulador IQ a pasabanda real."""

    def __init__(self, fc_normalized: float = CARRIER_FREQ_NORMALIZED):
        if not (0.0 < fc_normalized < 0.5):
            raise ValueError("fc normalizada debe estar en (0, 0.5) (Nyquist).")
        self.fc_norm = fc_normalized

    def modulate(self, baseband_signal: np.ndarray) -> np.ndarray:
        n = np.arange(len(baseband_signal))
        phase = 2.0 * np.pi * self.fc_norm * n
        return (np.real(baseband_signal) * np.cos(phase)
                - np.imag(baseband_signal) * np.sin(phase))


class Demodulator:
    """Demodulador coherente IQ."""

    def __init__(self, fc_normalized: float = CARRIER_FREQ_NORMALIZED,
                 alpha: float = 0.35):
        self.fc_norm = fc_normalized
        self.alpha = alpha

    def lowpass_cutoff_rs(self) -> float:
        """Frecuencia de corte del LPF en unidades de Rs."""
        return 0.5 * (1.0 + self.alpha) + LPF_MARGIN_RS

    def demodulate(self, rf_signal: np.ndarray, sps: int) -> np.ndarray:
        """Retorna la señal en banda base compleja (misma longitud que la entrada)."""
        n = np.arange(len(rf_signal))
        phase = 2.0 * np.pi * self.fc_norm * n
        h = self._lowpass_coefficients(sps)

        i_filtered = fir_filter_centered(rf_signal * np.cos(phase), h)
        q_filtered = fir_filter_centered(rf_signal * (-np.sin(phase)), h)
        return 2.0 * (i_filtered + 1j * q_filtered)

    def _lowpass_coefficients(self, sps: int) -> np.ndarray:
        """Diseña el FIR pasa-bajas: sinc enventanada con Hann y ganancia DC = 1."""
        cutoff = self.lowpass_cutoff_rs() / sps
        if cutoff >= 0.5:
            raise ValueError("El corte del LPF supera Nyquist: aumente sps.")
        n_taps = LPF_TAPS_PER_SYMBOL * sps + 1
        n_vec = np.arange(-(n_taps // 2), n_taps // 2 + 1)

        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(
                n_vec == 0,
                2.0 * cutoff,
                np.sin(2.0 * np.pi * cutoff * n_vec) / (np.pi * n_vec),
            )
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_taps) / (n_taps - 1))
        h = h * window
        return h / np.sum(h)
