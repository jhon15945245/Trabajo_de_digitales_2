"""
test_system.py
==============
Pruebas automáticas del simulador (solo biblioteca estándar: unittest).

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -v

Cada prueba verifica una propiedad teórica del sistema de comunicaciones:
mapeo Gray, energía unitaria del SRRC, ausencia de ISI, calibración del
ruido (Eb/N0), concordancia BER simulada vs teórica, métricas y validaciones.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from analysis.comparison import (                      # noqa: E402
    ebn0_for_ber, interpolate_ebn0, theoretical_table,
)
from core.ber_calculator import BERCalculator, q_function   # noqa: E402
from core.link import CommLink                         # noqa: E402
from core.mapper import Mapper, generate_random_bits, gray_code   # noqa: E402
from core.pulse_shaper import PulseShaper              # noqa: E402
from core.simulation_runner import SimulationParams, SimulationRunner   # noqa: E402
from metrics.signal_power import compute_signal_power  # noqa: E402
from metrics.spectral_efficiency import compute_spectral_efficiency   # noqa: E402
from plots.eye_diagram import eye_segments, eye_time_axis   # noqa: E402
from utils.constants import VALID_ORDERS               # noqa: E402
from utils.validators import (                         # noqa: E402
    validate_modulation_order, validate_roll_off, validate_simulation_params,
)

ALL_SCHEMES = [(s, m) for s, orders in VALID_ORDERS.items() for m in orders]


class TestMapper(unittest.TestCase):
    def test_gray_adjacency_all_orders(self):
        """Puntos vecinos a distancia mínima difieren en exactamente 1 bit."""
        for scheme, m in ALL_SCHEMES:
            c = Mapper(scheme, m).get_constellation()
            d = np.abs(c[:, None] - c[None, :])
            np.fill_diagonal(d, np.inf)
            dmin = d.min()
            for i in range(m):
                for j in range(m):
                    if abs(d[i, j] - dmin) < 1e-9:
                        self.assertEqual(bin(i ^ j).count("1"), 1, f"{scheme} M={m}: {i}-{j}")

    def test_unit_average_energy(self):
        for scheme, m in ALL_SCHEMES:
            c = Mapper(scheme, m).get_constellation()
            self.assertAlmostEqual(float(np.mean(np.abs(c) ** 2)), 1.0, places=12)

    def test_constellation_points_unique(self):
        for scheme, m in ALL_SCHEMES:
            c = Mapper(scheme, m).get_constellation()
            self.assertEqual(len(np.unique(np.round(c, 9))), m)

    def test_gray_code_sequence(self):
        self.assertEqual(gray_code(8), [0, 1, 3, 2, 6, 7, 5, 4])

    def test_invalid_order_rejected(self):
        for scheme, m in [("M-QAM", 8), ("M-QAM", 6), ("M-PSK", 6), ("M-PAM", 3)]:
            with self.assertRaises(ValueError):
                Mapper(scheme, m)


class TestPulseShaping(unittest.TestCase):
    def test_srrc_unit_energy(self):
        for a in (0.0, 0.2, 0.35, 0.5, 1.0):
            h = PulseShaper(a, 20, 8).get_coefficients()
            self.assertAlmostEqual(float(np.sum(h ** 2)), 1.0, places=12)

    def test_cascade_is_nyquist(self):
        """TX⊛RX = coseno alzado: ISI ≈ 0 en múltiplos de T (α ≥ 0.2)."""
        for a in (0.2, 0.35, 0.5, 1.0):
            h = PulseShaper(a, 20, 8).get_coefficients()
            rc = np.convolve(h, h)
            c = len(rc) // 2
            isi = [abs(rc[c + 8 * n] / rc[c]) for n in range(1, 8)]
            self.assertLess(max(isi), 5e-3, f"α={a}")

    def test_invalid_alpha_rejected(self):
        for a in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                PulseShaper(a, 20, 8)


class TestEndToEnd(unittest.TestCase):
    def test_noiseless_chain_has_no_errors(self):
        """Sin ruido (Eb/N0 muy alto) la BER es 0 para todo esquema, orden y roll-off."""
        for scheme, m in ALL_SCHEMES:
            for a in (0.2, 0.35, 1.0):
                link = CommLink(scheme, m, a)
                bits = generate_random_bits(1500, link.k, 7)
                out = link.run(bits, 80.0, 3)
                self.assertEqual(out.n_errors, 0, f"{scheme} M={m} α={a}")
                evm = np.sqrt(np.mean(np.abs(out.symbols_rx - out.symbols_tx) ** 2))
                self.assertLess(evm, 0.02, f"EVM {scheme} M={m} α={a}")

    def test_noise_calibration(self):
        """Varianza del ruido tras el filtro casado = N0/2 por componente (±5 %)."""
        link = CommLink("M-QAM", 4, 0.35)
        bits = generate_random_bits(40000, 2, 1)
        tx = link.transmit(bits)
        ref = link.receive(tx, 80.0, 3).symbols_rx
        for eb_n0_db in (4.0, 8.0):
            r = link.receive(tx, eb_n0_db, 3).symbols_rx - ref
            n0 = (1.0 / 2) / 10 ** (eb_n0_db / 10)
            self.assertAlmostEqual(r.real.var() / (n0 / 2), 1.0, delta=0.05)
            self.assertAlmostEqual(r.imag.var() / (n0 / 2), 1.0, delta=0.05)

    def test_ber_matches_theory(self):
        """BER simulada ≈ teórica (razón en [0.85, 1.15], ≥ 1000 errores)."""
        cases = [("M-QAM", 4, 0.35, 4.0), ("M-PAM", 4, 0.35, 6.0), ("M-PSK", 4, 0.35, 4.0),
                 ("M-QAM", 16, 0.35, 8.0), ("M-PSK", 8, 0.5, 6.0), ("M-PAM", 8, 0.2, 8.0),
                 ("M-PSK", 2, 0.35, 2.0), ("M-QAM", 64, 0.35, 10.0)]
        for scheme, m, a, eb_n0 in cases:
            link = CommLink(scheme, m, a)
            n_err = n_bits = 0
            for b in range(8):
                if n_err >= 1000:
                    break
                bits = generate_random_bits(20000, link.k, [11, b])
                e, n = link.count_errors(bits, eb_n0, [12, b])
                n_err += e
                n_bits += n
            ratio = (n_err / n_bits) / float(BERCalculator.ber_theoretical(scheme, m, eb_n0))
            self.assertTrue(0.85 < ratio < 1.15, f"{scheme} M={m}: razón {ratio:.2f} ({n_err} errores)")

    def test_alpha_is_used_in_curve(self):
        """La curva BER depende del roll-off elegido (antes estaba fijo en 0.35)."""
        pts = np.array([6.0])
        a, _, _ = SimulationRunner.simulate_ber_curve("M-QAM", 16, 0.0, pts, min_errors=500, max_symbols=60000)
        b, _, _ = SimulationRunner.simulate_ber_curve("M-QAM", 16, 1.0, pts, min_errors=500, max_symbols=60000)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        # α=0 (SRRC truncado) y α=1 dan ISI distinto → no pueden ser idénticos
        self.assertNotEqual(float(a[0]), float(b[0]))

    def test_reproducibility(self):
        p = SimulationParams(scheme="M-QAM", m=16, n_symbols=1500, eb_n0_db=8.0,
                             eb_n0_steps=4, max_symbols=10000, min_errors=50)
        r1, r2 = SimulationRunner().run(p), SimulationRunner().run(p)
        self.assertEqual(r1.ber_simulated_point, r2.ber_simulated_point)
        np.testing.assert_array_equal(r1.symbols_rx, r2.symbols_rx)
        np.testing.assert_array_equal(r1.ber_simulated, r2.ber_simulated)


class TestTheoryAndTable(unittest.TestCase):
    def test_q_function(self):
        self.assertAlmostEqual(q_function(0.0), 0.5, places=12)
        self.assertAlmostEqual(q_function(1.0), 0.15865525393145707, places=12)

    def test_bpsk_equals_2pam(self):
        x = np.array([0.0, 4.0, 8.0])
        np.testing.assert_allclose(BERCalculator.ber_theoretical("M-PSK", 2, x),
                                   BERCalculator.ber_theoretical("M-PAM", 2, x))
        self.assertAlmostEqual(float(BERCalculator.ber_theoretical("M-PSK", 2, 4.0)),
                               q_function(np.sqrt(2 * 10 ** 0.4)), places=12)

    def test_4psk_equals_4qam(self):
        x = np.linspace(0, 14, 15)
        np.testing.assert_allclose(BERCalculator.ber_theoretical("M-PSK", 4, x),
                                   BERCalculator.ber_theoretical("M-QAM", 4, x), rtol=1e-12)

    def test_ebn0_inversion(self):
        for sch, m in (("M-PAM", 4), ("M-QAM", 4), ("M-PSK", 4)):
            for t in (1e-2, 1e-7):
                x = ebn0_for_ber(sch, m, t)
                self.assertAlmostEqual(float(BERCalculator.ber_theoretical(sch, m, x)) / t, 1.0, places=6)

    def test_delivery_table(self):
        th = theoretical_table()
        d = th["diff"]
        for v in d["4PSK-4QAM"]:
            self.assertAlmostEqual(v, 0.0, places=6)
        limit = 10 * np.log10(2.5)
        prev = 0
        for v, w in zip(d["4PAM-4QAM"], d["4QAM-4PAM"]):
            self.assertAlmostEqual(v, -w, places=9)
            self.assertTrue(prev < v < limit)      # crece y tiende a 3.98 dB
            prev = v
        self.assertAlmostEqual(d["4PAM-4QAM"][0], 3.559, delta=0.005)

    def test_interpolation(self):
        x = np.array([0., 1., 2.])
        y = np.array([1e-1, 1e-2, 1e-3])
        e = np.array([1000, 1000, 1000])
        self.assertAlmostEqual(interpolate_ebn0(x, y, e, 1e-2), 1.0, places=9)
        self.assertAlmostEqual(interpolate_ebn0(x, y, e, 10 ** -2.5), 1.5, places=9)
        self.assertTrue(np.isnan(interpolate_ebn0(x, y, e, 1e-5)))


class TestMetrics(unittest.TestCase):
    def test_spectral_efficiency(self):
        self.assertAlmostEqual(compute_spectral_efficiency(4, 0.35), 2 / 1.35, places=12)
        self.assertAlmostEqual(compute_spectral_efficiency(16, 0.0), 4.0, places=12)

    def test_power_and_energy(self):
        for scheme, m in (("M-QAM", 4), ("M-PAM", 8), ("M-PSK", 16)):
            res = SimulationRunner().run_point(SimulationParams(scheme=scheme, m=m, n_symbols=4000))
            self.assertAlmostEqual(res.symbol_energy, 1.0, places=9)
            # Es medida en la señal = energía media de los símbolos realmente transmitidos
            # (la muestra aleatoria de símbolos puede desviarse unos % de Es = 1)
            es_sent = float(np.mean(np.abs(res.symbols_tx) ** 2))
            self.assertAlmostEqual(res.symbol_energy_measured / es_sent, 1.0, delta=0.005)
            self.assertAlmostEqual(res.bit_energy, 1.0 / np.log2(m), places=9)
            # P_pb = Es/2 (sin dividir dos veces por 2)
            self.assertAlmostEqual(res.signal_power / (0.5 * es_sent), 1.0, delta=0.01)
            self.assertAlmostEqual(res.signal_power_theory, 0.5, places=9)

    def test_compute_signal_power_has_no_hidden_factor(self):
        x = np.cos(2 * np.pi * 0.1 * np.arange(10000))
        self.assertAlmostEqual(compute_signal_power(x), 0.5, delta=1e-3)

    def test_bandwidth_metrics(self):
        res = SimulationRunner().run_point(SimulationParams(alpha=0.35, symbol_rate_baud=2000.0))
        self.assertAlmostEqual(res.bandwidth_norm, 1.35, places=12)
        self.assertAlmostEqual(res.bandwidth_hz, 2700.0, places=9)


class TestEye(unittest.TestCase):
    def test_time_axis_step_is_one_over_sps(self):
        t = eye_time_axis(8)
        self.assertEqual(len(t), 17)
        np.testing.assert_allclose(np.diff(t), 1 / 8)
        self.assertEqual(t[8], 0.0)          # instante de muestreo en el centro

    def test_eye_open_after_matched_filter(self):
        """Sin ruido, en t = 0 las trazas de 4-PAM se agrupan en 4 niveles."""
        res = SimulationRunner().run_point(SimulationParams(scheme="M-PAM", m=4, eb_n0_db=40.0, n_symbols=1500))
        segs = eye_segments(np.real(res.mf_output), 8, 300)
        centre = segs[:, 8]                         # t = 0 (instante de muestreo)
        nominal = np.array([-3, -1, 1, 3]) / np.sqrt(5)
        dist = np.min(np.abs(centre[:, None] - nominal[None, :]), axis=1)
        self.assertLess(dist.max(), 0.05)           # ojo abierto: 4 niveles nítidos


class TestValidation(unittest.TestCase):
    def test_roll_off(self):
        self.assertFalse(validate_roll_off(1.7)[0])
        self.assertFalse(validate_roll_off(-0.1)[0])
        self.assertFalse(validate_roll_off("abc")[0])
        self.assertFalse(validate_roll_off("")[0])
        self.assertFalse(validate_roll_off(float("nan"))[0])
        self.assertTrue(validate_roll_off("0,35")[0])        # coma decimal
        self.assertTrue(validate_roll_off(0)[0])
        self.assertTrue(validate_roll_off(1)[0])

    def test_modulation_order(self):
        self.assertFalse(validate_modulation_order("M-QAM", 8)[0])
        self.assertFalse(validate_modulation_order("M-QAM", "4.5")[0])
        self.assertFalse(validate_modulation_order("M-PSK", 0)[0])
        self.assertFalse(validate_modulation_order("M-PSK", -4)[0])
        self.assertTrue(validate_modulation_order("M-PSK", "16")[0])

    def test_params_collects_all_errors(self):
        values, errors = validate_simulation_params("M-QAM", 8, 1.7, 2000, 10, 0, 20, 10,
                                                    1000, 100, 10000, 8)
        self.assertIsNone(values)
        self.assertIn("m", errors)
        self.assertIn("alpha", errors)

    def test_runner_rejects_invalid_params(self):
        with self.assertRaises(ValueError):
            SimulationRunner().run_point(SimulationParams(scheme="M-QAM", m=8))
        with self.assertRaises(ValueError):
            SimulationRunner().run_point(SimulationParams(alpha=1.7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
