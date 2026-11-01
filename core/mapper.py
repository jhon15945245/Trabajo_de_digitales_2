"""
mapper.py
=========
Transmisor — Etapa 1: Mapeo de bits a símbolos complejos.

Implementa desde cero (sin funciones de alto nivel) los constelaciones:
  - M-PAM : símbolos reales en {-(M-1), -(M-3), ..., (M-3), (M-1)}
  - M-PSK : símbolos en el círculo unitario, ángulos equiespaciados
  - M-QAM : retícula cuadrada, combina dos PAM de sqrt(M) puntos

Usa codificación Gray para minimizar BER: bits adyacentes en la
constelación difieren en un solo bit.

Referencia teórica:
  Proakis & Salehi, "Digital Communications", 5ª ed., Cap. 6-7.
"""

import numpy as np
from utils.constants import RANDOM_SEED


def _gray_code(n: int) -> list[int]:
    """
    Genera los primeros n enteros en código Gray.
    Gray(i) = i XOR (i >> 1)
    """
    return [i ^ (i >> 1) for i in range(n)]


class Mapper:
    """
    Convierte una secuencia de bits en símbolos del alfabeto de modulación.

    Parámetros
    ----------
    scheme : str
        Esquema de modulación: 'M-PAM', 'M-QAM' o 'M-PSK'.
    m : int
        Orden de la modulación (número de símbolos en la constelación).
    """

    def __init__(self, scheme: str, m: int):
        self.scheme = scheme
        self.m = m
        self.bits_per_symbol = int(np.log2(m))   # k = log2(M)
        self._constellation = self._build_constellation()

    # ── API pública ──────────────────────────────────────────────────────────

    def map_bits(self, bits: np.ndarray) -> np.ndarray:
        """
        Convierte un array de bits en símbolos complejos.

        Proceso:
          1. Agrupa bits en bloques de k = log2(M).
          2. Convierte cada bloque a índice entero (big-endian).
          3. Busca el símbolo en la tabla de constelación usando Gray.

        Parámetros
        ----------
        bits : ndarray de int (0/1), longitud múltiplo de k.

        Retorna
        -------
        symbols : ndarray complejo, longitud = len(bits) / k.
        """
        k = self.bits_per_symbol
        n_symbols = len(bits) // k
        bits_trimmed = bits[: n_symbols * k]

        # Agrupa en bloques de k bits y convierte a índice decimal
        groups = bits_trimmed.reshape(n_symbols, k)
        # big-endian: el primer bit es el más significativo
        powers = 2 ** np.arange(k - 1, -1, -1)
        indices = groups @ powers          # índice en [0, M-1]

        # Mapea índice → símbolo usando la tabla (Gray aplicado en _build)
        return self._constellation[indices]

    def get_constellation(self) -> np.ndarray:
        """Retorna la constelación completa (útil para graficar)."""
        return self._constellation.copy()

    def get_bits_per_symbol(self) -> int:
        return self.bits_per_symbol

    # ── Construcción de constelaciones ──────────────────────────────────────

    def _build_constellation(self) -> np.ndarray:
        """Despacha la construcción según el esquema."""
        builders = {
            "M-PAM": self._build_pam,
            "M-QAM": self._build_qam,
            "M-PSK": self._build_psk,
        }
        return builders[self.scheme]()

    def _build_pam(self) -> np.ndarray:
        """
        M-PAM: símbolos reales en {±1, ±3, ..., ±(M-1)}.
        Escala para energía promedio unitaria: dividir entre
        sqrt(E_s) donde E_s = (M²-1)/3.

        Codificación Gray: re-ordena las amplitudes según el código Gray
        para que palabras de código adyacentes difieran en un bit.
        """
        m = self.m
        # Amplitudes base: -(M-1), -(M-3), ..., (M-3), (M-1)
        amplitudes = np.array([2 * i - (m - 1) for i in range(m)], dtype=float)

        # Normalización: energía promedio = 1
        e_s = np.mean(amplitudes ** 2)
        amplitudes /= np.sqrt(e_s)

        # Aplicar permutación Gray
        gray = _gray_code(m)
        constellation = np.zeros(m, dtype=complex)
        for bit_index, gray_index in enumerate(gray):
            constellation[bit_index] = amplitudes[gray_index]

        return constellation

    def _build_qam(self) -> np.ndarray:
        """
        M-QAM cuadrada: combina dos constelaciones PAM de sqrt(M) puntos
        en los ejes I y Q.

        Condición: M debe ser potencia de 2 con raíz entera exacta.

        La codificación Gray se aplica independientemente en I y Q
        (bits de mayor orden → I, bits de menor orden → Q).
        """
        m = self.m
        sqrt_m = int(np.sqrt(m))
        # Amplitudes de un PAM de sqrt(M) puntos (sin normalizar aún)
        pam_vals = np.array([2 * i - (sqrt_m - 1) for i in range(sqrt_m)], dtype=float)
        e_s = np.mean(pam_vals ** 2) * 2          # factor 2 por dos ejes
        pam_vals /= np.sqrt(e_s)

        gray = _gray_code(sqrt_m)
        pam_gray = np.zeros(sqrt_m)
        for bit_idx, g_idx in enumerate(gray):
            pam_gray[bit_idx] = pam_vals[g_idx]

        # Construir retícula: índice i → (i // sqrt_m, i % sqrt_m)
        constellation = np.zeros(m, dtype=complex)
        for i in range(m):
            i_idx = i // sqrt_m    # coordenada I (bits más significativos)
            q_idx = i % sqrt_m     # coordenada Q (bits menos significativos)
            constellation[i] = pam_gray[i_idx] + 1j * pam_gray[q_idx]

        return constellation

    def _build_psk(self) -> np.ndarray:
        """
        M-PSK: símbolos sobre el círculo unitario.
        Ángulo del k-ésimo símbolo (con Gray): θ_k = 2π·Gray(k)/M.

        La energía es constante (= 1) para todos los símbolos.
        """
        m = self.m
        gray = _gray_code(m)
        constellation = np.zeros(m, dtype=complex)
        for bit_index, gray_index in enumerate(gray):
            theta = 2 * np.pi * gray_index / m
            constellation[bit_index] = np.cos(theta) + 1j * np.sin(theta)
        return constellation


def generate_random_bits(n_symbols: int, bits_per_symbol: int,
                          seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Genera bits aleatorios reproducibles.
    Usar siempre esta función (con seed) garantiza RNF3.

    Retorna
    -------
    bits : ndarray de int (0/1), longitud = n_symbols * bits_per_symbol.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_symbols * bits_per_symbol)
