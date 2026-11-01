"""
spectrum.py
===========
RF3d — Espectro de la señal en banda base (TX) y pasabanda (RX).

Calcula y visualiza la Densidad Espectral de Potencia (PSD) usando
la Transformada de Fourier Discreta (DFT/FFT) implementada con
numpy.fft (función de bajo nivel disponible en numpy, no de alto nivel
de scipy.signal):

    PSD[k] = |X[k]|² / N

donde X[k] = FFT(x[n]) es la DFT de la señal.

Se aplica ventana de Hann antes de la FFT para reducir la fuga espectral
(spectral leakage), implementada manualmente:

    w[n] = 0.5 − 0.5·cos(2π·n / (N−1))

Visualización:
  - Espectro en banda base: centrado en 0 Hz.
  - Espectro en pasabanda: centrado en fc (frecuencia portadora).

Referencia: Proakis, Cap. 2.2 — Fourier Spectrum.
"""

import numpy as np
from matplotlib.axes import Axes


def _compute_psd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la densidad espectral de potencia usando FFT con ventana Hann.

    Parámetros
    ----------
    signal : ndarray — señal a analizar (real o compleja).

    Retorna
    -------
    freqs : ndarray — frecuencias normalizadas ∈ [-0.5, 0.5].
    psd   : ndarray — PSD en dB, centrada en cero.
    """
    n = len(signal)

    # Ventana de Hann implementada manualmente
    n_vec  = np.arange(n)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n_vec / (n - 1))

    # Señal enventanada
    windowed = signal * window

    # FFT y corrimiento para centrar en cero (fftshift)
    spectrum = np.fft.fft(windowed, n=n)
    spectrum = np.fft.fftshift(spectrum)

    # PSD normalizada
    psd_linear = (np.abs(spectrum) ** 2) / n

    # Convertir a dB (añadir 1e-12 para evitar log(0))
    psd_db = 10.0 * np.log10(psd_linear + 1e-12)

    # Frecuencias normalizadas ∈ [-0.5, 0.5]
    freqs = np.fft.fftshift(np.fft.fftfreq(n))

    return freqs, psd_db


def plot_spectrum(ax: Axes, baseband_signal: np.ndarray,
                 passband_signal: np.ndarray,
                 fc_normalized: float = 0.25,
                 sps: int = 8) -> None:
    """
    Dibuja el espectro en banda base (TX) y pasabanda (RX) en un Axes.

    Muestra ambos espectros superpuestos para comparación directa.

    Parámetros
    ----------
    ax               : Axes   — eje de matplotlib.
    baseband_signal  : ndarray — señal en banda base (después del pulso shaping).
    passband_signal  : ndarray — señal en pasabanda (después de la modulación).
    fc_normalized    : float  — frecuencia portadora normalizada.
    sps              : int    — muestras por símbolo (para normalizar el eje).
    """
    ax.clear()

    freqs_bb, psd_bb = _compute_psd(baseband_signal)
    freqs_pb, psd_pb = _compute_psd(passband_signal)

    # Escalar frecuencias a unidades de Rs (tasa de símbolo)
    freqs_bb_rs = freqs_bb * sps
    freqs_pb_rs = freqs_pb * sps

    ax.plot(freqs_bb_rs, psd_bb, "b-", linewidth=1.5,
            label="Banda base (TX)", alpha=0.9)
    ax.plot(freqs_pb_rs, psd_pb, "r-", linewidth=1.5,
            label="Pasabanda (RX)", alpha=0.7)

    ax.set_title("Espectro de la señal", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frecuencia normalizada [f / Rs]", fontsize=9)
    ax.set_ylabel("PSD [dB]", fontsize=9)
    ax.set_ylim(bottom=max(psd_bb.min(), -80))   # limitar el piso de ruido en pantalla
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(x= fc_normalized * sps, color="orange", linestyle="--",
               linewidth=1, alpha=0.7, label=f"fc = {fc_normalized:.2f}·fs")
    ax.axvline(x=-fc_normalized * sps, color="orange", linestyle="--",
               linewidth=1, alpha=0.7)
