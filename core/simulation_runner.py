"""
simulation_runner.py
====================
Orquestador de la simulación.  No conoce la GUI: recibe parámetros y retorna
resultados (se puede usar en scripts y pruebas).

Dos etapas (para que la interfaz responda de inmediato, RF1):

  run_point(params)  → señales, constelación, ojo, espectro, métricas, BER
                       teórica y BER puntual.  Tarda decenas de milisegundos.
  run_curve(...)     → curva BER simulada por Monte Carlo (más lenta, con
                       progreso y cancelación).
  run(params)        → ambas etapas.

Curva BER por Monte Carlo
-------------------------
Para cada Eb/N0 se simulan bloques de `CURVE_BLOCK_SYMBOLS` símbolos hasta
acumular `min_errors` errores o llegar a `max_symbols` símbolos.  Cada bloque
usa semillas derivadas (seed, índice de punto, bloque) → resultados
reproducibles y puntos independientes entre sí (RNF3).  Un punto sin errores
no es "BER = 0": solo indica BER < 1/bits_simulados (se grafica como cota).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core.ber_calculator import BERCalculator
from core.link import CommLink
from core.mapper import generate_random_bits
from metrics.bandwidth import compute_bandwidth_hz, compute_bandwidth_normalized
from metrics.signal_power import (
    compute_normalized_power, power_to_db, theoretical_passband_power,
)
from metrics.spectral_efficiency import compute_bit_rate, compute_spectral_efficiency
from metrics.symbol_energy import (
    compute_bit_energy, compute_symbol_energy, compute_symbol_energy_measured,
)
from utils.constants import (
    CURVE_BLOCK_SYMBOLS, CURVE_MAX_SYMBOLS, CURVE_MIN_ERRORS,
    DEFAULT_EB_N0_DB, DEFAULT_EB_N0_MAX, DEFAULT_EB_N0_MIN, DEFAULT_EB_N0_STEPS,
    DEFAULT_NUM_SYMBOLS, DEFAULT_SAMPLES_PER_SYM, DEFAULT_SYMBOL_RATE_BAUD,
    RANDOM_SEED,
)
from utils.validators import validate_simulation_params


@dataclass
class SimulationParams:
    """Parámetros de entrada (la GUI llena este dataclass)."""
    scheme:      str   = "M-QAM"
    m:           int   = 4
    alpha:       float = 0.35
    sps:         int   = DEFAULT_SAMPLES_PER_SYM
    n_symbols:   int   = DEFAULT_NUM_SYMBOLS
    eb_n0_db:    float = DEFAULT_EB_N0_DB      # punto de operación [dB]
    eb_n0_min:   float = DEFAULT_EB_N0_MIN
    eb_n0_max:   float = DEFAULT_EB_N0_MAX
    eb_n0_steps: int   = DEFAULT_EB_N0_STEPS
    symbol_rate_baud: float = DEFAULT_SYMBOL_RATE_BAUD
    min_errors:  int   = CURVE_MIN_ERRORS
    max_symbols: int   = CURVE_MAX_SYMBOLS
    seed:        int   = RANDOM_SEED

    def validate(self) -> None:
        """Lanza ValueError con todos los problemas encontrados."""
        _, errors = validate_simulation_params(
            self.scheme, self.m, self.alpha, self.n_symbols, self.eb_n0_db,
            self.eb_n0_min, self.eb_n0_max, self.eb_n0_steps,
            self.symbol_rate_baud, self.min_errors, self.max_symbols, self.sps,
        )
        if errors:
            raise ValueError("\n".join(errors.values()))


def _empty() -> np.ndarray:
    return np.array([])


@dataclass
class SimulationResults:
    """Resultados de una simulación (la GUI los lee para graficar)."""
    # Señales (región útil)
    symbols_tx:      np.ndarray = field(default_factory=_empty)
    symbols_rx:      np.ndarray = field(default_factory=_empty)
    baseband_shaped: np.ndarray = field(default_factory=_empty)   # TX banda base
    passband_signal: np.ndarray = field(default_factory=_empty)   # TX pasabanda
    passband_rx:     np.ndarray = field(default_factory=_empty)   # RX pasabanda (con ruido)
    baseband_rx:     np.ndarray = field(default_factory=_empty)   # tras demodulación
    mf_output:       np.ndarray = field(default_factory=_empty)   # tras filtro casado (tasa completa)
    ideal_constellation: np.ndarray = field(default_factory=_empty)

    # BER en el punto de operación
    ber_simulated_point: float = 0.0
    errors_point: int = 0
    bits_point:   int = 0
    ber_theoretical_point: float = 0.0
    evm_rms: float = 0.0

    # Curvas BER
    eb_n0_array:     np.ndarray = field(default_factory=_empty)
    ber_theoretical: np.ndarray = field(default_factory=_empty)
    ber_simulated:   np.ndarray = field(default_factory=_empty)
    curve_errors:    np.ndarray = field(default_factory=_empty)
    curve_bits:      np.ndarray = field(default_factory=_empty)

    # Métricas
    spectral_efficiency: float = 0.0     # bit/s/Hz
    bit_rate_bps:        float = 0.0     # bit/s
    symbol_energy:       float = 0.0     # Es teórica [u.e.]
    symbol_energy_measured: float = 0.0  # Es medida en la señal [u.e.]
    bit_energy:          float = 0.0     # Eb = Es/k [u.e.]
    bandwidth_norm:      float = 0.0     # B/Rs
    bandwidth_hz:        float = 0.0     # B [Hz]
    signal_power:        float = 0.0     # P_pb [Es/Ts]  (medida)
    signal_power_theory: float = 0.0     # Es/2 [Es/Ts]
    signal_power_db:     float = 0.0
    signal_power_per_s:  float = 0.0     # P_pb · Rs [u.e./s]


ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


class SimulationRunner:
    """Ejecuta la simulación de extremo a extremo."""

    # ── Etapa 1: punto de operación ──────────────────────────────────────────

    def run_point(self, params: SimulationParams) -> SimulationResults:
        params.validate()
        link = CommLink(params.scheme, params.m, params.alpha, params.sps)
        k = link.k

        bits = generate_random_bits(params.n_symbols, k, params.seed)
        out = link.run(bits, params.eb_n0_db, [params.seed, 1])

        res = SimulationResults()
        res.symbols_tx = out.symbols_tx
        res.symbols_rx = out.symbols_rx
        res.baseband_shaped = out.baseband_tx
        res.passband_signal = out.passband_tx
        res.passband_rx = out.passband_rx
        res.baseband_rx = out.baseband_rx
        res.mf_output = out.mf_output
        res.ideal_constellation = link.mapper.get_constellation()

        res.errors_point, res.bits_point = out.n_errors, out.n_bits
        res.ber_simulated_point = out.n_errors / out.n_bits
        res.ber_theoretical_point = float(
            BERCalculator.ber_theoretical(params.scheme, params.m, params.eb_n0_db))
        res.evm_rms = float(np.sqrt(np.mean(np.abs(out.symbols_rx - out.symbols_tx) ** 2)))

        # Curva teórica (instantánea) y ejes de la curva simulada
        res.eb_n0_array = np.linspace(params.eb_n0_min, params.eb_n0_max, params.eb_n0_steps)
        res.ber_theoretical = BERCalculator.ber_theoretical(
            params.scheme, params.m, res.eb_n0_array)
        res.ber_simulated = np.full(len(res.eb_n0_array), np.nan)
        res.curve_errors = np.zeros(len(res.eb_n0_array), dtype=int)
        res.curve_bits = np.zeros(len(res.eb_n0_array), dtype=int)

        # Métricas
        constellation = res.ideal_constellation
        res.symbol_energy = compute_symbol_energy(constellation)
        res.symbol_energy_measured = compute_symbol_energy_measured(out.baseband_tx, params.sps)
        res.bit_energy = compute_bit_energy(res.symbol_energy, params.m)
        res.spectral_efficiency = compute_spectral_efficiency(params.m, params.alpha)
        res.bit_rate_bps = compute_bit_rate(params.m, params.symbol_rate_baud)
        res.bandwidth_norm = compute_bandwidth_normalized(params.alpha)
        res.bandwidth_hz = compute_bandwidth_hz(params.alpha, params.symbol_rate_baud)
        res.signal_power = compute_normalized_power(out.passband_tx, params.sps)
        res.signal_power_theory = theoretical_passband_power(res.symbol_energy)
        res.signal_power_db = power_to_db(res.signal_power)
        res.signal_power_per_s = res.signal_power * params.symbol_rate_baud
        return res

    # ── Etapa 2: curva BER simulada ──────────────────────────────────────────

    def run_curve(self, params: SimulationParams, results: SimulationResults,
                  progress: Optional[ProgressCallback] = None,
                  cancelled: Optional[CancelCallback] = None) -> bool:
        """
        Llena results.ber_simulated / curve_errors / curve_bits.
        Retorna False si se canceló antes de terminar.
        """
        ber, errors, bits = self.simulate_ber_curve(
            params.scheme, params.m, params.alpha, results.eb_n0_array,
            params.sps, params.seed, params.min_errors, params.max_symbols,
            progress, cancelled)
        if ber is None:
            return False
        results.ber_simulated, results.curve_errors, results.curve_bits = ber, errors, bits
        return True

    @staticmethod
    def simulate_ber_curve(scheme: str, m: int, alpha: float, eb_n0_db_array,
                           sps: int = DEFAULT_SAMPLES_PER_SYM, seed: int = RANDOM_SEED,
                           min_errors: int = CURVE_MIN_ERRORS,
                           max_symbols: int = CURVE_MAX_SYMBOLS,
                           progress: Optional[ProgressCallback] = None,
                           cancelled: Optional[CancelCallback] = None):
        """
        Monte Carlo punto a punto con el MISMO sistema que el punto de operación.

        Retorna (ber, errores, bits) o (None, None, None) si se canceló.
        ber[i] = NaN si no se simuló ningún bit.
        """
        link = CommLink(scheme, m, alpha, sps)
        k = link.k
        pts = np.asarray(eb_n0_db_array, dtype=float)
        ber = np.full(len(pts), np.nan)
        errors = np.zeros(len(pts), dtype=int)
        bits_count = np.zeros(len(pts), dtype=int)
        n_blocks_max = max(1, int(np.ceil(max_symbols / CURVE_BLOCK_SYMBOLS)))
        block = min(CURVE_BLOCK_SYMBOLS, max_symbols)

        for i, eb_n0 in enumerate(pts):
            n_err = n_bits = 0
            for b in range(n_blocks_max):
                if cancelled is not None and cancelled():
                    return None, None, None
                bits = generate_random_bits(block, k, [seed, 10 + i, b])
                e, n = link.count_errors(bits, float(eb_n0), [seed, 1000 + i, b])
                n_err += e
                n_bits += n
                if n_err >= min_errors:
                    break
            errors[i], bits_count[i] = n_err, n_bits
            ber[i] = n_err / n_bits if n_bits else np.nan
            if progress is not None:
                progress(i + 1, len(pts))
        return ber, errors, bits_count

    # ── Todo junto ───────────────────────────────────────────────────────────

    def run(self, params: SimulationParams,
            progress: Optional[ProgressCallback] = None) -> SimulationResults:
        results = self.run_point(params)
        self.run_curve(params, results, progress)
        return results
