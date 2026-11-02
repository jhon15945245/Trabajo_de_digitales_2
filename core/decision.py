"""
decision.py
===========
Receptor — Decisor y demapeo.

Criterio de máxima verosimilitud (ML) en AWGN con símbolos equiprobables:
mínima distancia euclidiana al punto de la constelación,

        ŝ = argmin_{s_i ∈ C} |r − s_i|²

El índice del punto elegido ES la etiqueta de bits (la tabla del Mapper está
indexada por etiqueta), por lo que el demapeo es la expansión binaria del
índice (big-endian).

Referencia: Proakis & Salehi, Cap. 4 (detección ML).
"""

from __future__ import annotations

import numpy as np

_CHUNK_ELEMENTS = 4_000_000     # tamaño máximo de la matriz N×M procesada a la vez


class Decisor:
    """Decisor por mínima distancia y demapeo a bits."""

    def __init__(self, constellation: np.ndarray, bits_per_symbol: int):
        self._constellation = np.asarray(constellation)
        self._bits_per_symbol = bits_per_symbol
        self._m = len(constellation)

    def decide(self, received_symbols: np.ndarray) -> np.ndarray:
        """Índice del símbolo más cercano (procesa por bloques para limitar memoria)."""
        r = np.asarray(received_symbols)
        c_re, c_im = self._constellation.real, self._constellation.imag
        chunk = max(1, _CHUNK_ELEMENTS // self._m)
        out = np.empty(len(r), dtype=np.int64)
        for start in range(0, len(r), chunk):
            seg = r[start: start + chunk]
            d2 = ((seg.real[:, None] - c_re[None, :]) ** 2
                  + (seg.imag[:, None] - c_im[None, :]) ** 2)
            out[start: start + chunk] = np.argmin(d2, axis=1)
        return out

    def demap(self, decided_indices: np.ndarray) -> np.ndarray:
        """Índices → bits (vectorizado, big-endian: primer bit = más significativo)."""
        k = self._bits_per_symbol
        shifts = np.arange(k - 1, -1, -1)
        bits = (decided_indices[:, None] >> shifts[None, :]) & 1
        return bits.reshape(-1).astype(int)

    def decide_and_demap(self, received_symbols: np.ndarray) -> np.ndarray:
        return self.demap(self.decide(received_symbols))

    def get_decided_symbols(self, received_symbols: np.ndarray) -> np.ndarray:
        return self._constellation[self.decide(received_symbols)]
