"""
dsp.py
======
Utilidades de procesamiento digital de señales implementadas desde cero (RNF2).

Solo se usan arreglos de numpy (indexación, suma, producto, exp).  NO se usan
np.convolve, np.fft.*, scipy.signal ni funciones de comunicaciones.

  fft(x), ifft(X)             FFT radix-2 (Cooley-Tukey iterativa, vectorizada)
  fft_shift(x), fft_freqs(n)  utilidades de espectro (centrado y eje de frecuencia)
  convolve(x, h)              convolución lineal (directa o por FFT)
  fir_filter_centered(x, h)   filtra con FIR simétrico compensando su retardo
                              de grupo: la salida tiene la misma longitud y
                              queda alineada con la entrada.
  upsample_filter_centered    sobremuestreo ×sps + filtrado (polifásico: no
                              multiplica por los ceros insertados).
  filter_decimate_centered    filtrado evaluado solo cada sps muestras.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Con filtros cortos la convolución directa (vectorizada) es más rápida que la FFT;
# a partir de este nº de coeficientes se usa la FFT (medido: cruce ≈ 400 coeficientes).
_DIRECT_MAX_TAPS = 400


# ── FFT radix-2 ──────────────────────────────────────────────────────────────

def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Menor potencia de 2 ≥ n."""
    return 1 << max(0, (int(n) - 1).bit_length())


@lru_cache(maxsize=32)
def _bit_reversal(n: int) -> np.ndarray:
    """Permutación de inversión de bits para una FFT de n puntos (n = 2^b)."""
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


@lru_cache(maxsize=32)
def _twiddles(size: int) -> np.ndarray:
    """Factores W_size^k = exp(-j2πk/size), k = 0 … size/2-1."""
    return np.exp(-2j * np.pi * np.arange(size // 2) / size)


def fft(x: np.ndarray) -> np.ndarray:
    """
    DFT por FFT radix-2 decimación en el tiempo:
        X[k] = Σ_n x[n]·exp(-j2πkn/N),   N = 2^b.
    En cada etapa se combinan pares (a, b·W):  (a + bW, a − bW).
    """
    a = np.asarray(x, dtype=complex)
    n = len(a)
    if not _is_power_of_two(n):
        raise ValueError("La FFT radix-2 requiere una longitud potencia de 2.")
    if n == 1:
        return a.copy()
    a = a[_bit_reversal(n)]
    size = 2
    while size <= n:
        half = size // 2
        blocks = a.reshape(-1, size)
        even = blocks[:, :half]
        odd = blocks[:, half:] * _twiddles(size)
        a = np.concatenate((even + odd, even - odd), axis=1).reshape(-1)
        size *= 2
    return a


def ifft(spectrum: np.ndarray) -> np.ndarray:
    """IFFT:  x = conj( FFT( conj(X) ) ) / N."""
    s = np.asarray(spectrum, dtype=complex)
    return np.conj(fft(np.conj(s))) / len(s)


def fft_shift(x: np.ndarray) -> np.ndarray:
    """Mueve la frecuencia 0 al centro (longitud par)."""
    return np.roll(x, len(x) // 2)


def fft_freqs(n: int) -> np.ndarray:
    """Frecuencias normalizadas a fs de una FFT de n puntos, en orden [0, …, -1/n]."""
    k = np.arange(n)
    k = np.where(k >= (n + 1) // 2, k - n, k)
    return k / n


# ── Convolución ──────────────────────────────────────────────────────────────

def _convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Convolución lineal directa  y[n] = Σ_k h[k]·x[n-k]  (O(N·L)).
    Se acumula h[k]·x desplazado k muestras (el bucle recorre la secuencia corta).
    """
    if len(h) > len(x):
        x, h = h, x
    dtype = np.result_type(x.dtype, h.dtype, np.float64)
    y = np.zeros(len(x) + len(h) - 1, dtype=dtype)
    for k, hk in enumerate(h):
        y[k: k + len(x)] += hk * x
    return y


def _convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Convolución lineal por FFT (teorema de convolución):
        y = IFFT( FFT(x, N) · FFT(h, N) ),   N ≥ len(x)+len(h)-1  (potencia de 2)
    """
    n_out = len(x) + len(h) - 1
    n_fft = next_power_of_two(n_out)
    xp = np.zeros(n_fft, dtype=complex)
    hp = np.zeros(n_fft, dtype=complex)
    xp[: len(x)] = x
    hp[: len(h)] = h
    y = ifft(fft(xp) * fft(hp))[:n_out]
    if not (np.iscomplexobj(x) or np.iscomplexobj(h)):
        y = y.real
    return y


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución lineal (elige método directo o FFT según el tamaño)."""
    x = np.asarray(x)
    h = np.asarray(h)
    if min(len(x), len(h)) > _DIRECT_MAX_TAPS:
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


def upsample_filter_centered(symbols: np.ndarray, h: np.ndarray, sps: int) -> np.ndarray:
    """
    Equivale a  fir_filter_centered( s↑sps , h )  donde s↑sps inserta (sps-1)
    ceros entre símbolos, pero sin operar sobre los ceros:

        y[j·sps + k] += h[k]·s[j]     para cada coeficiente k.

    La salida tiene len(symbols)·sps muestras y el símbolo j queda centrado en
    la muestra j·sps.
    """
    s = np.asarray(symbols)
    n_sym = len(s)
    dtype = np.result_type(s.dtype, np.asarray(h).dtype, np.float64)
    y = np.zeros(n_sym * sps + len(h) - 1, dtype=dtype)
    for k, hk in enumerate(h):
        y[k: k + sps * (n_sym - 1) + 1: sps] += hk * s
    delay = (len(h) - 1) // 2
    return y[delay: delay + n_sym * sps]


def filter_decimate_centered(x: np.ndarray, h: np.ndarray, sps: int,
                             start: int, count: int) -> np.ndarray:
    """
    Salidas  y_c[start + j·sps],  j = 0 … count-1,  del filtrado centrado
    (equivale a fir_filter_centered(x, h)[start::sps][:count], pero solo se
    calculan las muestras necesarias):

        y_c[n] = Σ_k h[k]·x[n + delay − k]

    Las muestras de x fuera de rango se consideran cero.
    """
    x = np.asarray(x)
    delay = (len(h) - 1) // 2
    dtype = np.result_type(x.dtype, np.asarray(h).dtype, np.float64)
    out = np.zeros(count, dtype=dtype)
    for k, hk in enumerate(h):
        base = start + delay - k                       # índice de x para j = 0
        j_lo = max(0, -((base) // sps)) if base < 0 else 0     # primer j con índice ≥ 0
        j_hi = min(count - 1, (len(x) - 1 - base) // sps)      # último j con índice < len(x)
        if j_hi < j_lo:
            continue
        first = base + sps * j_lo
        out[j_lo: j_hi + 1] += hk * x[first: first + sps * (j_hi - j_lo) + 1: sps]
    return out
