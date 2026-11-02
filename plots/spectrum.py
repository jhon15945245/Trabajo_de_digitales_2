"""
spectrum.py
===========
RF3d — Espectro de la señal en banda base (TRANSMISIÓN) y en pasabanda
(RECEPCIÓN, es decir, después del canal y con ruido).

PSD estimada con el método de Welch implementado desde cero:
  1. Se divide la señal en segmentos de longitud L con 50 % de solape.
  2. Cada segmento se multiplica por una ventana de Hann.
  3. Se promedian los periodogramas |FFT|²/(U·fs), U = Σ w².

El promediado reduce la varianza del periodograma simple.  Ambas curvas se
normalizan al máximo de la banda base (0 dB), de modo que se conservan las
potencias relativas: una señal real pasabanda reparte su potencia en dos
bandas (±fc), cada una con densidad pico de −6 dB respecto a la banda base.

Eje de frecuencia: f/Rs  (Rs = tasa de símbolo = fs/sps).
Marcas: bordes del espectro ±(1+α)/2·Rs en banda base y portadora ±fc.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from utils.dsp import fft, fft_freqs, fft_shift, next_power_of_two


def welch_psd(signal: np.ndarray, nperseg: int | None = None, overlap: float = 0.5):
    """
    PSD bilateral (Welch) de una señal real o compleja, frecuencias normalizadas
    a fs (rango [-0.5, 0.5)).  Retorna (freqs, psd_lineal).
    """
    x = np.asarray(signal)
    n = len(x)
    if nperseg is None:
        nperseg = 1 << max(8, int(np.floor(np.log2(max(n // 16, 256)))))
    nperseg = min(nperseg, next_power_of_two(n + 1) // 2)   # potencia de 2 ≤ n (FFT radix-2)
    step = max(1, int(nperseg * (1.0 - overlap)))

    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nperseg) / (nperseg - 1))
    u = np.sum(w ** 2)

    acc = np.zeros(nperseg)
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start: start + nperseg] * w
        acc += np.abs(fft(seg)) ** 2 / u
        count += 1
    psd = fft_shift(acc / max(count, 1))
    freqs = fft_shift(fft_freqs(nperseg))
    return freqs, psd


def plot_spectrum(ax: Axes, baseband_signal: np.ndarray, passband_rx: np.ndarray,
                  fc_normalized: float = 0.25, sps: int = 8, alpha: float = 0.35) -> None:
    ax.clear()
    f_bb, p_bb = welch_psd(baseband_signal)
    f_pb, p_pb = welch_psd(passband_rx)

    ref = np.max(p_bb)
    floor = 1e-14 * ref
    db_bb = 10 * np.log10(p_bb / ref + floor)
    db_pb = 10 * np.log10(p_pb / ref + floor)

    ax.plot(f_bb * sps, db_bb, "b-", linewidth=1.4, label="Banda base (TX)")
    ax.plot(f_pb * sps, db_pb, "r-", linewidth=1.2, alpha=0.8, label="Pasabanda (RX, con ruido)")

    edge = 0.5 * (1.0 + alpha)
    for sign in (1, -1):
        ax.axvline(sign * edge, color="blue", linestyle=":", linewidth=1, alpha=0.6)
        ax.axvline(sign * fc_normalized * sps, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax.plot([], [], "b:", label=f"±(1+α)/2·Rs = ±{edge:.3f}")
    ax.plot([], [], color="orange", linestyle="--", label=f"±fc = ±{fc_normalized * sps:.1f}·Rs")

    ax.set_title("Espectro de la señal (PSD, Welch)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frecuencia normalizada f / Rs", fontsize=9)
    ax.set_ylabel("PSD relativa [dB]  (0 dB = máx. banda base)", fontsize=9)
    ax.set_ylim(-80, 5)
    ax.set_xlim(-sps / 2, sps / 2)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
