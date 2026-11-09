"""Utilidades de procesamiento digital de señales implementadas desde cero (RNF2)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_DIRECT_MAX_TAPS = 400


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
    """DFT por FFT radix-2 decimación en el tiempo: X[k] = Σ_n x[n]·exp(-j2πkn/N), N = 2^b."""
    a = np.asarray(x, dtype=complex)
    n = len(a)
    if not _is_power_of_two(n):
        raise ValueError("La FFT radix-2 requiere una longitud potencia de 2.")
    if n == 1:
        return a.copy()
    # Ordena las muestras para procesar la señal en etapas de tamaño creciente.
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
    """IFFT: x = conj( FFT( conj(X) ) ) / N."""
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


def _convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución lineal directa y[n] = Σ_k h[k]·x[n-k] (O(N·L))."""
    if len(h) > len(x):
        x, h = h, x
    dtype = np.result_type(x.dtype, h.dtype, np.float64)
    y = np.zeros(len(x) + len(h) - 1, dtype=dtype)
    for k, hk in enumerate(h):
        y[k: k + len(x)] += hk * x
    return y


def _convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución lineal por FFT: y = IFFT(FFT(x)·FFT(h))."""
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
    # Para filtros cortos conviene el cálculo directo; para largos, el de FFT.
    if min(len(x), len(h)) > _DIRECT_MAX_TAPS:
        return _convolve_fft(x, h)
    return _convolve_direct(x, h)


def fir_filter_centered(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Filtra x con el FIR simétrico h (longitud impar) compensando el retardo de grupo (len(h)-1)/2 muestras."""
    delay = (len(h) - 1) // 2
    y = convolve(x, h)
    return y[delay: delay + len(x)]


def upsample_filter_centered(symbols: np.ndarray, h: np.ndarray, sps: int) -> np.ndarray:
    """Sobremuestrea y filtra en un solo paso, equivalente a insertar ceros y filtrar."""
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
    """Filtra y submuestrea solo en los instantes de símbolo."""
    x = np.asarray(x)
    delay = (len(h) - 1) // 2
    dtype = np.result_type(x.dtype, np.asarray(h).dtype, np.float64)
    out = np.zeros(count, dtype=dtype)
    for k, hk in enumerate(h):
        base = start + delay - k
        j_lo = max(0, -((base) // sps)) if base < 0 else 0
        j_hi = min(count - 1, (len(x) - 1 - base) // sps)
        if j_hi < j_lo:
            continue
        first = base + sps * j_lo
        out[j_lo: j_hi + 1] += hk * x[first: first + sps * (j_hi - j_lo) + 1: sps]
    return out
