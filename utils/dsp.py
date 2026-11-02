"""
dsp.py
======
Utilidades de procesamiento digital de señales implementadas desde cero.

Solo se usan primitivas numéricas de numpy (arreglos y FFT).  No se usa
scipy.signal ni funciones de comunicaciones de alto nivel (RNF2).

  convolve(x, h)              convolución lineal (directa o por FFT)
  fir_filter_centered(x, h)   filtra con FIR simétrico compensando su retardo
                              de grupo: la salida tiene la misma longitud y
                              queda alineada con la entrada.
"""

from __future__ import annotations

import numpy as np

# Umbral (nº de productos) a partir del cual conviene la convolución por FFT
_FFT_THRESHOLD = 2_000_000


def _convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución lineal directa  y[n] = Σ_k h[k]·x[n-k]  (O(N·L))."""
    return np.convolve(x, h)


def _convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Convolución lineal por FFT (teorema de convolución):
        y = IFFT( FFT(x, N) · FFT(h, N) ),   N ≥ len(x)+len(h)-1
    con relleno de ceros hasta una potencia de 2.
    """
    n_out = len(x) + len(h) - 1
    n_fft = 1 << (n_out - 1).bit_length()
    if np.iscomplexobj(x) or np.iscomplexobj(h):
        y = np.fft.ifft(np.fft.fft(x, n_fft) * np.fft.fft(h, n_fft))
    else:
        y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)
    return y[:n_out]


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución lineal (elige método directo o FFT según el tamaño)."""
    x = np.asarray(x)
    h = np.asarray(h)
    if len(x) * len(h) > _FFT_THRESHOLD:
        return _convolve_fft(x, h)
    return _convolve_direct(x, h)


def fir_filter_centered(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Filtra x con el FIR simétrico h (longitud impar) compensando el retardo
    de grupo (len(h)-1)/2 muestras.  La salida tiene len(x) muestras y la
    muestra n de la salida corresponde a la muestra n de la entrada.
    """
    delay = (len(h) - 1) // 2
    y = convolve(x, h)
    return y[delay: delay + len(x)]
