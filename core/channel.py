"""
channel.py
==========
Canal AWGN calibrado por Eb/N0.

Definiciones (todo en unidades "por muestra" con Ts = sps muestras):

    P_bb  = potencia de la envolvente compleja  (E|s[n]|²)
    Es    = P_bb · sps                  (energía por símbolo)
    Eb    = Es / k,  k = log2(M)        (energía por bit)
    N0    = Eb / (Eb/N0)_lineal         (densidad espectral de ruido)

Varianza del ruido que produce el Eb/N0 pedido
----------------------------------------------
El detector (filtro casado de energía unitaria + muestreo) necesita, en cada
componente (I y Q), una varianza de ruido de N0/2.  Como el filtro casado
tiene energía unitaria, no modifica la varianza de un ruido blanco, por lo que:

  • Señal COMPLEJA en banda base:  cada componente tiene varianza σ² = N0/2
    (ruido complejo de potencia total N0).
    (Error corregido: la versión anterior usaba N0/4 por componente, es decir
    3.01 dB menos ruido del pedido.)

  • Señal REAL en pasabanda:  P_bb = 2·P_pb (una portadora real tiene la
    mitad de la potencia de su envolvente compleja).  El demodulador
    multiplica por 2·cos y 2·sin y filtra: el ruido blanco real de varianza σ_r²
    produce en cada rama una varianza 2·σ_r².  Igualando a N0/2:

                          σ_r² = N0/4.

Referencia: Proakis & Salehi, Cap. 4 (AWGN) y Cap. 8.
"""

from __future__ import annotations

import numpy as np

from utils.constants import RANDOM_SEED


class AWGNChannel:
    """
    Canal AWGN.

    Parámetros
    ----------
    seed : entero o secuencia de enteros (RNF3: reproducibilidad).
    """

    def __init__(self, seed=RANDOM_SEED):
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def noise_variance(signal_power: float, is_complex: bool, eb_n0_db: float,
                       bits_per_symbol: int, sps: int) -> float:
        """
        Varianza del ruido (por componente real) para un Eb/N0 dado.

        signal_power : potencia media de la señal que entra al canal
                       (E|s|² si es compleja, E[x²] si es real pasabanda).
        """
        p_bb = signal_power if is_complex else 2.0 * signal_power
        es = p_bb * sps
        eb = es / bits_per_symbol
        eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)        # dB → lineal
        n0 = eb / eb_n0_lin
        return n0 / 2.0 if is_complex else n0 / 4.0

    def transmit(self, signal: np.ndarray, eb_n0_db: float,
                 bits_per_symbol: int, sps: int,
                 signal_power: float | None = None) -> np.ndarray:
        """
        Agrega ruido AWGN calibrado.

        signal_power : potencia a usar para calibrar (por defecto la medida
                       sobre toda la señal).  link.py pasa la potencia de la
                       región útil para no contar los símbolos de relleno.
        """
        is_complex = np.iscomplexobj(signal)
        if signal_power is None:
            signal_power = float(np.mean(np.abs(signal) ** 2))
        sigma2 = self.noise_variance(signal_power, is_complex, eb_n0_db,
                                     bits_per_symbol, sps)
        std = np.sqrt(sigma2)
        if is_complex:
            noise = (self._rng.normal(0.0, std, len(signal))
                     + 1j * self._rng.normal(0.0, std, len(signal)))
        else:
            noise = self._rng.normal(0.0, std, len(signal))
        return signal + noise
