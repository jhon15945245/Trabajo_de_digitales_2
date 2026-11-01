"""
ber_calculator.py
=================
Cálculo de BER simulada y BER teórica.

BER SIMULADA:
  Comparación bit a bit entre los bits transmitidos y los recibidos.
  BER_sim = (número de bits en error) / (número total de bits)

BER TEÓRICA:
  Fórmulas analíticas exactas para canal AWGN, derivadas de la función
  Q(x) = P(X > x) para X ~ N(0,1):

  M-PAM:
    BER = (2(M-1)/M) · (1/log2(M)) · Q(√(6·log2(M)·Eb/N0 / (M²-1)))

  M-QAM cuadrada:
    BER ≈ (4(√M-1)/√M) · (1/log2(M)) · Q(√(3·log2(M)·Eb/N0 / (M-1)))

    [Cada eje I o Q es un PAM de √M puntos; factor 4 por los 4 vecinos]

  M-PSK:
    BER ≈ (2/log2(M)) · Q(√(2·log2(M)·Eb/N0) · sin(π/M))

    [Válida exactamente para M≤4 y como aproximación para M>4 con
     probabilidad de error dominada por los dos vecinos más cercanos]

Función Q implementada sin scipy: Q(x) = 0.5·erfc(x/√2)
donde erfc es la función de error complementaria (disponible en numpy
como math.erfc o se puede calcular con la serie).

Referencia: Proakis, Cap. 5.2–5.4; Simon & Alouini, Cap. 8.
"""

import numpy as np
from math import erfc


def q_function(x: float | np.ndarray) -> float | np.ndarray:
    """
    Función Q: probabilidad de que una variable Normal(0,1) supere x.

    Q(x) = (1/2) · erfc(x / √2)

    Implementada con erfc de la librería math (sin scipy).
    Soporta escalares y arrays de numpy.
    """
    return 0.5 * np.vectorize(lambda v: erfc(v / np.sqrt(2)))(x)


class BERCalculator:
    """
    Calcula la BER simulada y la BER teórica para M-PAM, M-QAM y M-PSK.
    """

    # ── BER simulada ─────────────────────────────────────────────────────────

    @staticmethod
    def ber_simulated(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
        """
        Calcula la BER comparando bits transmitidos y recibidos.

        Parámetros
        ----------
        bits_tx : ndarray int — bits originales enviados.
        bits_rx : ndarray int — bits recuperados después del decisor.

        Retorna
        -------
        ber : float — tasa de error de bit (0 a 1). Si no hay errores,
              retorna 0.0 (no NaN).
        """
        n_total = min(len(bits_tx), len(bits_rx))
        if n_total == 0:
            return 0.0
        n_errors = int(np.sum(bits_tx[:n_total] != bits_rx[:n_total]))
        return n_errors / n_total

    # ── BER teórica ──────────────────────────────────────────────────────────

    @staticmethod
    def ber_theoretical(scheme: str, m: int,
                        eb_n0_db_array: np.ndarray) -> np.ndarray:
        """
        Calcula la curva de BER teórica para un rango de Eb/N0.

        Parámetros
        ----------
        scheme        : str     — 'M-PAM', 'M-QAM' o 'M-PSK'.
        m             : int     — orden de la modulación.
        eb_n0_db_array: ndarray — valores de Eb/N0 en dB.

        Retorna
        -------
        ber_theory : ndarray — BER teórica para cada Eb/N0.
        """
        eb_n0_lin = 10.0 ** (eb_n0_db_array / 10.0)   # dB → lineal
        k = np.log2(m)                                  # bits por símbolo

        dispatchers = {
            "M-PAM": BERCalculator._ber_pam,
            "M-QAM": BERCalculator._ber_qam,
            "M-PSK": BERCalculator._ber_psk,
        }
        return dispatchers[scheme](m, k, eb_n0_lin)

    @staticmethod
    def _ber_pam(m: int, k: float, eb_n0_lin: np.ndarray) -> np.ndarray:
        """
        BER teórica para M-PAM en AWGN.

        Derivación:
          - La distancia mínima entre símbolos: d_min = 2·√(3·k·Eb / (M²-1))
          - Probabilidad de error de símbolo:
            Ps = 2(M-1)/M · Q(d_min/(2σ))
          - Con Gray code: Pb ≈ Ps / log2(M)

        BER = (2(M-1)/(M·k)) · Q(√(6·k·Eb/N0 / (M²-1)))
        """
        arg = np.sqrt(6.0 * k * eb_n0_lin / (m ** 2 - 1))
        return (2.0 * (m - 1) / (m * k)) * q_function(arg)

    @staticmethod
    def _ber_qam(m: int, k: float, eb_n0_lin: np.ndarray) -> np.ndarray:
        """
        BER teórica para M-QAM cuadrada en AWGN.

        M-QAM se descompone en dos M^(1/2)-PAM independientes en I y Q.
        La BER total es la suma de las BER de cada eje (con Gray):

        BER ≈ (4(√M-1)/(√M·k)) · Q(√(3·k·Eb/N0 / (M-1)))
        """
        sqrt_m = np.sqrt(m)
        arg    = np.sqrt(3.0 * k * eb_n0_lin / (m - 1))
        return (4.0 * (sqrt_m - 1) / (sqrt_m * k)) * q_function(arg)

    @staticmethod
    def _ber_psk(m: int, k: float, eb_n0_lin: np.ndarray) -> np.ndarray:
        """
        BER teórica para M-PSK en AWGN.

        Para M-PSK la probabilidad de error de símbolo se aproxima
        considerando solo los dos vecinos inmediatos (válido para Eb/N0 alto):

        Ps ≈ 2·Q(√(2·k·Eb/N0)·sin(π/M))

        Con Gray: BER ≈ Ps / log2(M)

        Para M=2 (BPSK): es exacta. Para M=4 (QPSK): equivale a 4-QAM.
        """
        arg = np.sqrt(2.0 * k * eb_n0_lin) * np.sin(np.pi / m)
        return (2.0 / k) * q_function(arg)

    # ── Curva completa para un rango de Eb/N0 ────────────────────────────────

    @staticmethod
    def compute_ber_curve(scheme: str, m: int, sps: int,
                          eb_n0_db_array: np.ndarray,
                          n_symbols: int, seed: int) -> np.ndarray:
        """
        Calcula la BER simulada punto a punto para toda la curva.

        Esto ejecuta la simulación completa (TX → canal → RX) para cada
        valor de Eb/N0 y mide la BER resultante.

        Parámetros
        ----------
        scheme        : str    — esquema de modulación.
        m             : int    — orden.
        sps           : int    — muestras por símbolo.
        eb_n0_db_array: ndarray — puntos Eb/N0 en dB.
        n_symbols     : int    — símbolos por punto.
        seed          : int    — semilla para reproducibilidad.

        Retorna
        -------
        ber_sim_array : ndarray — BER simulada para cada Eb/N0.
        """
        # Import local para evitar dependencias circulares
        from core.mapper       import Mapper, generate_random_bits
        from core.pulse_shaper import PulseShaper, MatchedFilter
        from core.channel      import AWGNChannel
        from core.decision     import Decisor

        mapper  = Mapper(scheme, m)
        shaper  = PulseShaper(alpha=0.35, sps=sps)
        mf      = MatchedFilter(alpha=0.35, sps=sps)
        channel = AWGNChannel(seed=seed)
        decisor = Decisor(mapper.get_constellation(), mapper.get_bits_per_symbol())

        bits_tx = generate_random_bits(n_symbols, mapper.get_bits_per_symbol(), seed)
        symbols_tx = mapper.map_bits(bits_tx)
        shaped     = shaper.shape(symbols_tx)

        ber_sim = np.zeros(len(eb_n0_db_array))
        for i, eb_n0_db in enumerate(eb_n0_db_array):
            # Canal: agrega ruido sobre la señal en banda base
            received = channel.transmit(
                shaped, eb_n0_db, mapper.get_bits_per_symbol(), sps
            )
            # Filtro casado + downsampling
            symbols_rx = mf.filter_and_downsample(received)
            symbols_rx = symbols_rx[:n_symbols]   # ajustar longitud

            # Decisión y demapeo
            bits_rx = decisor.decide_and_demap(symbols_rx)

            ber_sim[i] = BERCalculator.ber_simulated(bits_tx, bits_rx)

        return ber_sim
