"""
simulation_runner.py
====================
Orquestador de la simulación completa.

Conecta todas las etapas del sistema de comunicaciones en el orden correcto:

    Bits → Mapper → PulseShaper → Modulator → Canal AWGN
                                                    ↓
    Bits ← Demapper ← Decisor ← MatchedFilter ← Demodulator

Este módulo no conoce la GUI — solo recibe parámetros y retorna resultados.
Esto permite pruebas unitarias independientes de la interfaz.
"""

import numpy as np
from dataclasses import dataclass, field

from core.mapper        import Mapper, generate_random_bits
from core.pulse_shaper  import PulseShaper, MatchedFilter
from core.modulator     import Modulator, Demodulator
from core.channel       import AWGNChannel
from core.decision      import Decisor
from core.ber_calculator import BERCalculator
from metrics.spectral_efficiency import compute_spectral_efficiency
from metrics.symbol_energy       import compute_symbol_energy, compute_bit_energy
from metrics.bandwidth           import compute_bandwidth_normalized
from metrics.signal_power        import compute_signal_power
from utils.constants import (
    DEFAULT_SAMPLES_PER_SYM, DEFAULT_NUM_SYMBOLS,
    DEFAULT_EB_N0_MIN, DEFAULT_EB_N0_MAX, DEFAULT_EB_N0_STEPS,
    CARRIER_FREQ_NORMALIZED, RANDOM_SEED, FILTER_SPAN,
)


@dataclass
class SimulationParams:
    """
    Parámetros de entrada de la simulación.
    La GUI llena este dataclass y lo pasa a SimulationRunner.
    """
    scheme:      str   = "M-QAM"
    m:           int   = 4
    alpha:       float = 0.35
    sps:         int   = DEFAULT_SAMPLES_PER_SYM
    n_symbols:   int   = DEFAULT_NUM_SYMBOLS
    eb_n0_db:    float = 10.0          # punto de operación para las gráficas
    eb_n0_min:   float = DEFAULT_EB_N0_MIN
    eb_n0_max:   float = DEFAULT_EB_N0_MAX
    eb_n0_steps: int   = DEFAULT_EB_N0_STEPS


@dataclass
class SimulationResults:
    """
    Resultados de una simulación completa.
    La GUI lee este dataclass para actualizar todas las gráficas y métricas.
    """
    # ── Señales ───────────────────────────────────────────────────────────
    symbols_tx:      np.ndarray = field(default_factory=lambda: np.array([]))
    baseband_shaped: np.ndarray = field(default_factory=lambda: np.array([]))
    passband_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    baseband_rx:     np.ndarray = field(default_factory=lambda: np.array([]))
    symbols_rx:      np.ndarray = field(default_factory=lambda: np.array([]))

    # ── BER en el punto de operación ─────────────────────────────────────
    ber_simulated_point: float  = 0.0

    # ── Curvas BER ────────────────────────────────────────────────────────
    eb_n0_array:     np.ndarray = field(default_factory=lambda: np.array([]))
    ber_theoretical: np.ndarray = field(default_factory=lambda: np.array([]))
    ber_simulated:   np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Constelación ─────────────────────────────────────────────────────
    ideal_constellation: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Métricas ──────────────────────────────────────────────────────────
    spectral_efficiency: float = 0.0
    symbol_energy:       float = 0.0
    bit_energy:          float = 0.0
    bandwidth_norm:      float = 0.0
    signal_power:        float = 0.0


class SimulationRunner:
    """
    Ejecuta la simulación de comunicaciones digitales de extremo a extremo.

    Uso:
        runner  = SimulationRunner()
        params  = SimulationParams(scheme="M-QAM", m=16, alpha=0.25, ...)
        results = runner.run(params)
    """

    def run(self, params: SimulationParams) -> SimulationResults:
        """
        Ejecuta la simulación completa y retorna todos los resultados.

        Parámetros
        ----------
        params : SimulationParams — configuración del sistema.

        Retorna
        -------
        results : SimulationResults — señales, métricas y curvas BER.
        """
        results = SimulationResults()

        # ── 1. Construir bloques ──────────────────────────────────────────
        mapper      = Mapper(params.scheme, params.m)
        shaper      = PulseShaper(params.alpha, FILTER_SPAN, params.sps)
        mf          = MatchedFilter(params.alpha, FILTER_SPAN, params.sps)
        modulator   = Modulator(CARRIER_FREQ_NORMALIZED)
        demodulator = Demodulator(CARRIER_FREQ_NORMALIZED)
        channel     = AWGNChannel(seed=RANDOM_SEED)
        decisor     = Decisor(mapper.get_constellation(),
                               mapper.get_bits_per_symbol())

        # ── 2. Transmisor ─────────────────────────────────────────────────
        bits_tx        = generate_random_bits(
            params.n_symbols, mapper.get_bits_per_symbol(), RANDOM_SEED
        )
        symbols_tx     = mapper.map_bits(bits_tx)
        baseband_shaped = shaper.shape(symbols_tx)
        passband_signal = modulator.modulate(baseband_shaped)

        # ── 3. Canal ──────────────────────────────────────────────────────
        passband_rx = channel.transmit(
            passband_signal, params.eb_n0_db,
            mapper.get_bits_per_symbol(), params.sps
        )

        # ── 4. Receptor ───────────────────────────────────────────────────
        baseband_rx = demodulator.demodulate(passband_rx, params.sps)
        symbols_rx  = mf.filter_and_downsample(baseband_rx)
        symbols_rx  = symbols_rx[: params.n_symbols]    # recortar al tamaño original

        bits_rx     = decisor.decide_and_demap(symbols_rx)

        # ── 5. BER en el punto de operación ──────────────────────────────
        ber_point = BERCalculator.ber_simulated(bits_tx, bits_rx)

        # ── 6. Curva BER ──────────────────────────────────────────────────
        eb_n0_array = np.linspace(
            params.eb_n0_min, params.eb_n0_max, params.eb_n0_steps
        )
        ber_theory = BERCalculator.ber_theoretical(
            params.scheme, params.m, eb_n0_array
        )
        ber_sim = BERCalculator.compute_ber_curve(
            params.scheme, params.m, params.sps,
            eb_n0_array, params.n_symbols, RANDOM_SEED
        )

        # ── 7. Métricas ───────────────────────────────────────────────────
        constellation  = mapper.get_constellation()
        spec_eff       = compute_spectral_efficiency(params.m, params.alpha)
        sym_energy     = compute_symbol_energy(constellation)
        bit_energy_val = compute_bit_energy(constellation, params.m)
        bw_norm        = compute_bandwidth_normalized(params.alpha)
        sig_power      = compute_signal_power(passband_signal, is_passband=True)

        # ── 8. Empaquetar resultados ──────────────────────────────────────
        results.symbols_tx           = symbols_tx
        results.baseband_shaped      = baseband_shaped
        results.passband_signal      = passband_signal
        results.baseband_rx          = baseband_rx
        results.symbols_rx           = symbols_rx
        results.ber_simulated_point  = ber_point
        results.eb_n0_array          = eb_n0_array
        results.ber_theoretical      = ber_theory
        results.ber_simulated        = ber_sim
        results.ideal_constellation  = constellation
        results.spectral_efficiency  = spec_eff
        results.symbol_energy        = sym_energy
        results.bit_energy           = bit_energy_val
        results.bandwidth_norm       = bw_norm
        results.signal_power         = sig_power

        return results
