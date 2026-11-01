"""
decision.py
===========
Receptor — Decisor y Demapeo.

El decisor implementa el criterio de máxima verosimilitud (ML) para
canal AWGN, que equivale a encontrar el símbolo de la constelación
más cercano al símbolo recibido en distancia euclidiana:

    ŝ = argmin_{s_i ∈ C} |r - s_i|²

Una vez elegido el símbolo más cercano, se recuperan los bits que
lo representan (demapeo inverso de la tabla de constelación).

Referencia: Proakis, Cap. 5.1 — Maximum-Likelihood Detection.
"""

import numpy as np


class Decisor:
    """
    Toma decisiones símbolo a símbolo sobre los símbolos recibidos.

    Parámetros
    ----------
    constellation : ndarray complejo — tabla de símbolos de la constelación.
    bits_per_symbol : int — k = log2(M).
    """

    def __init__(self, constellation: np.ndarray, bits_per_symbol: int):
        self._constellation = constellation
        self._bits_per_symbol = bits_per_symbol
        self._m = len(constellation)

    def decide(self, received_symbols: np.ndarray) -> np.ndarray:
        """
        Clasificación por mínima distancia euclidiana.

        Para cada símbolo recibido r, calcula |r - s_i|² para todos los
        s_i en la constelación y devuelve el índice del más cercano.

        Parámetros
        ----------
        received_symbols : ndarray complejo — símbolos en recepción.

        Retorna
        -------
        decided_indices : ndarray int — índice del símbolo decidido (∈ [0, M-1]).
        """
        # Diferencia entre cada símbolo recibido y toda la constelación
        # received_symbols: (N,) → (N,1); constellation: (M,) → distancias (N,M)
        diff = received_symbols[:, np.newaxis] - self._constellation[np.newaxis, :]
        distances_sq = np.real(diff * np.conj(diff))   # |r - s_i|²

        return np.argmin(distances_sq, axis=1)

    def demap(self, decided_indices: np.ndarray) -> np.ndarray:
        """
        Convierte índices de símbolos decididos en bits (demapeo).

        Usa representación big-endian: el primer bit es el más significativo.

        Parámetros
        ----------
        decided_indices : ndarray int — índices ∈ [0, M-1].

        Retorna
        -------
        bits_rx : ndarray int (0/1) — bits recuperados.
        """
        k = self._bits_per_symbol
        n_symbols = len(decided_indices)
        bits_rx = np.zeros(n_symbols * k, dtype=int)

        for i, idx in enumerate(decided_indices):
            # Convertir índice a k bits (big-endian)
            for bit_pos in range(k):
                bits_rx[i * k + (k - 1 - bit_pos)] = (idx >> bit_pos) & 1

        return bits_rx

    def decide_and_demap(self, received_symbols: np.ndarray) -> np.ndarray:
        """
        Atajo: decide y luego desmapea en un solo llamado.
        """
        indices = self.decide(received_symbols)
        return self.demap(indices)

    def get_decided_symbols(self, received_symbols: np.ndarray) -> np.ndarray:
        """
        Retorna los símbolos de la constelación decididos (no los índices).
        Útil para el diagrama de constelación en recepción.
        """
        indices = self.decide(received_symbols)
        return self._constellation[indices]
