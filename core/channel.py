"""
channel.py
==========
Canal — Modelo AWGN (Additive White Gaussian Noise).

El canal más fundamental en comunicaciones digitales. Agrega ruido
gaussiano blanco a la señal transmitida:

    r(t) = s(t) + n(t)

donde n(t) ~ N(0, σ²) con varianza σ² calculada a partir de Eb/N0.

Relación entre σ² y Eb/N0:
    σ² = Eb / (2 · (Eb/N0))     ← ruido bilateral (potencia total = σ²)

donde Eb es la energía por bit medida directamente de la señal.

Referencia: Proakis, Cap. 4.3 — AWGN Channel Model.
"""

import numpy as np
from utils.constants import RANDOM_SEED


class AWGNChannel:
    """
    Canal AWGN: agrega ruido gaussiano blanco aditivo.

    Parámetros
    ----------
    seed : int — semilla aleatoria para reproducibilidad (RNF3).
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self._rng = np.random.default_rng(seed)

    def transmit(self, signal: np.ndarray, eb_n0_db: float,
                 bits_per_symbol: int, sps: int) -> np.ndarray:
        """
        Agrega ruido AWGN calibrado para un Eb/N0 dado.

        Proceso de cálculo de la varianza del ruido:
          1. Medir Eb (energía por bit) de la señal.
          2. Convertir Eb/N0 de dB a lineal.
          3. Calcular N0 = Eb / (Eb/N0)_lin.
          4. σ² = N0 / 2  (densidad espectral unilateral).

        Parámetros
        ----------
        signal        : ndarray — señal transmitida (real o compleja).
        eb_n0_db      : float   — relación Eb/N0 en dB.
        bits_per_symbol : int   — k = log2(M), bits por símbolo.
        sps           : int     — muestras por símbolo.

        Retorna
        -------
        received : ndarray — señal más ruido, mismo tipo que la entrada.
        """
        # Paso 1: energía por bit medida de la señal
        signal_power = np.mean(np.abs(signal) ** 2)
        eb = signal_power * sps / bits_per_symbol   # Eb = Ps·Ts / k

        # Paso 2: convertir Eb/N0 dB → lineal
        eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)

        # Paso 3: varianza del ruido
        n0 = eb / eb_n0_lin
        sigma2 = n0 / 2.0                           # varianza por componente

        # Paso 4: generar y agregar ruido
        noise = self._generate_noise(len(signal), sigma2, np.iscomplexobj(signal))
        return signal + noise

    def _generate_noise(self, length: int, sigma2: float,
                        is_complex: bool) -> np.ndarray:
        """
        Genera ruido gaussiano blanco.

        Para señal real   : n ~ N(0, σ²).
        Para señal compleja: n = nr + j·ni con nr,ni ~ N(0, σ²/2)
                             → potencia total = σ².
        """
        std = np.sqrt(sigma2)
        if is_complex:
            # Ruido complejo: igual potencia en I y Q
            noise_i = self._rng.normal(0.0, std / np.sqrt(2), length)
            noise_q = self._rng.normal(0.0, std / np.sqrt(2), length)
            return noise_i + 1j * noise_q
        else:
            return self._rng.normal(0.0, std, length)
