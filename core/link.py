"""Enlace extremo a extremo: transmisor, canal AWGN y receptor en un solo camino."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.channel import AWGNChannel
from core.decision import Decisor
from core.mapper import Mapper
from core.modulator import Demodulator, Modulator
from core.pulse_shaper import MatchedFilter, PulseShaper
from utils.constants import (
    CARRIER_FREQ_NORMALIZED, DEFAULT_SAMPLES_PER_SYM, filter_span_for_alpha,
)
from utils.validators import validate_roll_off, validate_sps


@dataclass
class LinkOutput:
    """Señales intermedias de una transmisión (solo la región útil)."""
    bits_tx: np.ndarray = field(default_factory=lambda: np.array([]))
    bits_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    symbols_tx: np.ndarray = field(default_factory=lambda: np.array([]))
    symbols_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    baseband_tx: np.ndarray = field(default_factory=lambda: np.array([]))
    passband_tx: np.ndarray = field(default_factory=lambda: np.array([]))
    passband_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    baseband_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_output: np.ndarray = field(default_factory=lambda: np.array([]))
    n_errors: int = 0
    n_bits: int = 0


class CommLink:
    """Cadena completa TX → canal → RX para un esquema, orden y roll-off dados."""

    def __init__(self, scheme: str, m: int, alpha: float,
                 sps: int = DEFAULT_SAMPLES_PER_SYM, span: int | None = None,
                 fc_norm: float = CARRIER_FREQ_NORMALIZED):
        ok, _, msg = validate_roll_off(alpha)
        if not ok:
            raise ValueError(msg)
        ok, _, msg = validate_sps(sps)
        if not ok:
            raise ValueError(msg)

        self.mapper = Mapper(scheme, m)
        self.scheme, self.m, self.alpha = scheme, self.mapper.m, alpha
        if span is None:
            span = filter_span_for_alpha(alpha)
        self.sps, self.span = sps, span
        self.k = self.mapper.get_bits_per_symbol()

        self.shaper = PulseShaper(alpha, span, sps)
        self.matched = MatchedFilter(alpha, span, sps)
        self.modulator = Modulator(fc_norm)
        self.demodulator = Demodulator(fc_norm, alpha)
        self.decisor = Decisor(self.mapper.get_constellation(), self.k)
        self.pad = span // 2 + 6


    def transmit(self, bits: np.ndarray) -> LinkOutput:
        """Mapeo → conformación → modulación (sin canal)."""
        out = LinkOutput(bits_tx=np.asarray(bits))
        symbols = self.mapper.map_bits(bits)
        padded = np.concatenate([np.zeros(self.pad, complex), symbols,
                                 np.zeros(self.pad, complex)])
        baseband = self.shaper.shape(padded)
        passband = self.modulator.modulate(baseband)

        out.symbols_tx = symbols
        out.baseband_tx = baseband
        out.passband_tx = passband
        return out

    def _core(self, n_symbols: int) -> slice:
        return slice(self.pad * self.sps, (self.pad + n_symbols) * self.sps)


    def receive(self, tx: LinkOutput, eb_n0_db: float, noise_seed,
                full_rate: bool = True) -> LinkOutput:
        """Canal AWGN → demodulación → filtro casado → muestreo → decisor."""
        n_sym = len(tx.symbols_tx)
        core = self._core(n_sym)

        es_theory = float(np.mean(np.abs(self.mapper.get_constellation()) ** 2))
        p_pb = es_theory / (2.0 * self.sps)
        channel = AWGNChannel(noise_seed)
        passband_rx = channel.transmit(tx.passband_tx, eb_n0_db, self.k, self.sps,
                                       signal_power=p_pb)
        baseband_rx = self.demodulator.demodulate(passband_rx, self.sps)
        if full_rate:
            mf_full = self.matched.filter(baseband_rx)
            symbols_rx = mf_full[core][:: self.sps]
            mf_core = mf_full[core]
        else:
            symbols_rx = self.matched.sample_symbols(baseband_rx, core.start, n_sym)
            mf_core = np.array([])
        bits_rx = self.decisor.decide_and_demap(symbols_rx)

        out = LinkOutput(
            bits_tx=tx.bits_tx, bits_rx=bits_rx,
            symbols_tx=tx.symbols_tx, symbols_rx=symbols_rx,
            baseband_tx=tx.baseband_tx[core], passband_tx=tx.passband_tx[core],
            passband_rx=passband_rx[core], baseband_rx=baseband_rx[core],
            mf_output=mf_core,
        )
        n_err = int(np.sum(out.bits_tx != out.bits_rx))
        out.n_errors, out.n_bits = n_err, len(out.bits_tx)
        return out

    def run(self, bits: np.ndarray, eb_n0_db: float, noise_seed) -> LinkOutput:
        """Transmisión completa de `bits` con ruido a `eb_n0_db`."""
        return self.receive(self.transmit(bits), eb_n0_db, noise_seed)

    def count_errors(self, bits: np.ndarray, eb_n0_db: float, noise_seed) -> tuple[int, int]:
        """Versión ligera de run(): solo (errores, bits)."""
        res = self.receive(self.transmit(bits), eb_n0_db, noise_seed, full_rate=False)
        return res.n_errors, res.n_bits
