"""Transmisor — Etapa 1: Mapeo de bits a símbolos complejos."""

from __future__ import annotations

import numpy as np

from utils.constants import RANDOM_SEED
from utils.validators import validate_modulation_order


def gray_code(n: int) -> list[int]:
    """Primeros n enteros del código Gray: Gray(i) = i XOR (i >> 1)."""
    return [i ^ (i >> 1) for i in range(n)]


class Mapper:
    """Convierte una secuencia de bits en símbolos del alfabeto de modulación."""

    def __init__(self, scheme: str, m: int):
        ok, m_valid, msg = validate_modulation_order(scheme, m)
        if not ok:
            raise ValueError(msg)
        self.scheme = scheme
        self.m = m_valid
        self.bits_per_symbol = int(round(np.log2(self.m)))
        self._constellation = self._build_constellation()


    def map_bits(self, bits: np.ndarray) -> np.ndarray:
        """Convierte los bits en símbolos: agrupa de a k bits, obtiene la etiqueta y busca el punto."""
        k = self.bits_per_symbol
        n_symbols = len(bits) // k
        groups = np.asarray(bits[: n_symbols * k], dtype=np.int64).reshape(n_symbols, k)
        powers = 2 ** np.arange(k - 1, -1, -1)
        labels = groups @ powers
        return self._constellation[labels]

    def get_constellation(self) -> np.ndarray:
        """Constelación completa indexada por etiqueta (copia)."""
        return self._constellation.copy()

    def get_bits_per_symbol(self) -> int:
        return self.bits_per_symbol


    def _build_constellation(self) -> np.ndarray:
        builders = {
            "M-PAM": self._build_pam,
            "M-QAM": self._build_qam,
            "M-PSK": self._build_psk,
        }
        return builders[self.scheme]()

    @staticmethod
    def _pam_levels(n_levels: int) -> np.ndarray:
        """Niveles PAM sin normalizar: -(n-1), -(n-3), ..., (n-3), (n-1)."""
        return np.array([2 * p - (n_levels - 1) for p in range(n_levels)], dtype=float)

    @staticmethod
    def _gray_permute(values_by_position: np.ndarray) -> np.ndarray:
        """Devuelve el arreglo indexado por etiqueta Gray: out[Gray(p)] = values[p]."""
        out = np.zeros_like(values_by_position)
        for p, g in enumerate(gray_code(len(values_by_position))):
            out[g] = values_by_position[p]
        return out

    def _build_pam(self) -> np.ndarray:
        """M-PAM: a_p = (2p-(M-1)) / sqrt(Es0), Es0 = (M²-1)/3 → Es = 1."""
        levels = self._pam_levels(self.m)
        levels = levels / np.sqrt(np.mean(levels ** 2))
        return self._gray_permute(levels).astype(complex)

    def _build_qam(self) -> np.ndarray:
        """M-QAM cuadrada: dos PAM-Gray independientes (ejes I y Q), normalizada a Es = 1."""
        sqrt_m = int(round(np.sqrt(self.m)))
        levels = self._pam_levels(sqrt_m)
        levels = levels / np.sqrt(2.0 * np.mean(levels ** 2))
        pam_gray = self._gray_permute(levels)

        constellation = np.zeros(self.m, dtype=complex)
        for label in range(self.m):
            i_label = label // sqrt_m
            q_label = label % sqrt_m
            constellation[label] = pam_gray[i_label] + 1j * pam_gray[q_label]
        return constellation

    def _build_psk(self) -> np.ndarray:
        """M-PSK: s_p = exp(j·2π·p/M), con etiqueta Gray(p)."""
        positions = np.exp(1j * 2.0 * np.pi * np.arange(self.m) / self.m)
        return self._gray_permute(positions)


def generate_random_bits(n_symbols: int, bits_per_symbol: int,
                         seed=RANDOM_SEED) -> np.ndarray:
    """Genera bits aleatorios reproducibles (RNF3)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_symbols * bits_per_symbol)
