"""Canal AWGN calibrado por Eb/N0."""

from __future__ import annotations

import numpy as np

from utils.constants import RANDOM_SEED


class AWGNChannel:
    """Canal AWGN."""

    def __init__(self, seed=RANDOM_SEED):
        # La semilla permite repetir la misma simulación cuando se usan los mismos datos.
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def noise_variance(signal_power: float, is_complex: bool, eb_n0_db: float,
                       bits_per_symbol: int, sps: int) -> float:
        """Varianza del ruido (por componente real) para un Eb/N0 dado."""
        # Convierte la potencia recibida en energía por bit y la ajusta al Eb/N0 solicitado.
        p_bb = signal_power if is_complex else 2.0 * signal_power
        es = p_bb * sps
        eb = es / bits_per_symbol
        eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)
        n0 = eb / eb_n0_lin
        return n0 / 2.0 if is_complex else n0 / 4.0

    def transmit(self, signal: np.ndarray, eb_n0_db: float,
                 bits_per_symbol: int, sps: int,
                 signal_power: float | None = None) -> np.ndarray:
        """Agrega ruido AWGN calibrado."""
        is_complex = np.iscomplexobj(signal)
        if signal_power is None:
            signal_power = float(np.mean(np.abs(signal) ** 2))
        sigma2 = self.noise_variance(signal_power, is_complex, eb_n0_db,
                                     bits_per_symbol, sps)
        std = np.sqrt(sigma2)
        # En una señal compleja se agrega ruido independiente a I y Q.
        if is_complex:
            noise = (self._rng.normal(0.0, std, len(signal))
                     + 1j * self._rng.normal(0.0, std, len(signal)))
        else:
            noise = self._rng.normal(0.0, std, len(signal))
        return signal + noise
