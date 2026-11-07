"""Pruebas automáticas del simulador (solo biblioteca estándar: unittest)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from analysis.comparison import (
    ebn0_for_ber, interpolate_ebn0, theoretical_table,
)
from core.ber_calculator import BERCalculator, q_function
from core.link import CommLink
from core.mapper import Mapper, generate_random_bits, gray_code
from core.pulse_shaper import PulseShaper
from core.simulation_runner import SimulationParams, SimulationRunner
from metrics.signal_power import compute_signal_power
from metrics.spectral_efficiency import compute_spectral_efficiency
from plots.eye_diagram import eye_segments, eye_time_axis
from utils.constants import VALID_ORDERS
from utils.validators import (
    validate_modulation_order, validate_roll_off, validate_simulation_params,
)

ALL_SCHEMES = [(s, m) for s, orders in VALID_ORDERS.items() for m in orders]


class TestMapper(unittest.TestCase):
    """Mapeo: código Gray, energía unitaria y órdenes válidos."""
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
    """Filtro SRRC: energía unitaria y cascada de Nyquist."""
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
    """Enlace completo: cadena sin ruido, calibración del ruido y BER frente a teoría."""
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
        self.assertNotEqual(float(a[0]), float(b[0]))

    def test_reproducibility(self):
        p = SimulationParams(scheme="M-QAM", m=16, n_symbols=1500, eb_n0_db=8.0,
                             eb_n0_steps=4, max_symbols=10000, min_errors=50)
        r1, r2 = SimulationRunner().run(p), SimulationRunner().run(p)
        self.assertEqual(r1.ber_simulated_point, r2.ber_simulated_point)
        np.testing.assert_array_equal(r1.symbols_rx, r2.symbols_rx)
        np.testing.assert_array_equal(r1.ber_simulated, r2.ber_simulated)


class TestTheoryAndTable(unittest.TestCase):
    """BER teórica, inversión de Eb/N0 y tabla de entrega."""
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
            self.assertTrue(prev < v < limit)
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
    """Métricas RF4: eficiencia espectral, energías, potencia y ancho de banda."""
    def test_spectral_efficiency(self):
        self.assertAlmostEqual(compute_spectral_efficiency(4, 0.35), 2 / 1.35, places=12)
        self.assertAlmostEqual(compute_spectral_efficiency(16, 0.0), 4.0, places=12)

    def test_power_and_energy(self):
        for scheme, m in (("M-QAM", 4), ("M-PAM", 8), ("M-PSK", 16)):
            res = SimulationRunner().run_point(SimulationParams(scheme=scheme, m=m, n_symbols=4000))
            self.assertAlmostEqual(res.symbol_energy, 1.0, places=9)
            es_sent = float(np.mean(np.abs(res.symbols_tx) ** 2))
            self.assertAlmostEqual(res.symbol_energy_measured / es_sent, 1.0, delta=0.005)
            self.assertAlmostEqual(res.bit_energy, 1.0 / np.log2(m), places=9)
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
    """Diagrama del ojo: eje de tiempo y apertura."""
    def test_time_axis_step_is_one_over_sps(self):
        t = eye_time_axis(8)
        self.assertEqual(len(t), 17)
        np.testing.assert_allclose(np.diff(t), 1 / 8)
        self.assertEqual(t[8], 0.0)

    def test_eye_open_after_matched_filter(self):
        """Sin ruido, en t = 0 las trazas de 4-PAM se agrupan en 4 niveles."""
        res = SimulationRunner().run_point(SimulationParams(scheme="M-PAM", m=4, eb_n0_db=40.0, n_symbols=1500))
        segs = eye_segments(np.real(res.mf_output), 8, 300)
        centre = segs[:, 8]
        nominal = np.array([-3, -1, 1, 3]) / np.sqrt(5)
        dist = np.min(np.abs(centre[:, None] - nominal[None, :]), axis=1)
        self.assertLess(dist.max(), 0.05)


class TestValidation(unittest.TestCase):
    """Validación de entradas del usuario (RNF1)."""
    def test_roll_off(self):
        self.assertFalse(validate_roll_off(1.7)[0])
        self.assertFalse(validate_roll_off(-0.1)[0])
        self.assertFalse(validate_roll_off("abc")[0])
        self.assertFalse(validate_roll_off("")[0])
        self.assertFalse(validate_roll_off(float("nan"))[0])
        self.assertTrue(validate_roll_off("0,35")[0])
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


def _ber_ratio(scheme, m, alpha, eb_n0, min_errors=1500, block=20000, max_blocks=12):
    """BER simulada / BER teórica acumulando bloques hasta `min_errors` errores."""
    link = CommLink(scheme, m, alpha)
    n_err = n_bits = 0
    for b in range(max_blocks):
        if n_err >= min_errors:
            break
        bits = generate_random_bits(block, link.k, [21, b])
        e, n = link.count_errors(bits, eb_n0, [22, b])
        n_err, n_bits = n_err + e, n_bits + n
    return (n_err / n_bits) / float(BERCalculator.ber_theoretical(scheme, m, eb_n0)), n_err


class TestSmallRollOff(unittest.TestCase):
    """Roll-off pequeño con SRRC truncado."""

    def test_span_grows_when_alpha_shrinks(self):
        from utils.constants import filter_span_for_alpha
        self.assertEqual(filter_span_for_alpha(0.35), 20)
        self.assertEqual(filter_span_for_alpha(0.10), 40)
        self.assertEqual(filter_span_for_alpha(0.0), 100)
        self.assertEqual(CommLink("M-QAM", 4, 0.0).span, 100)

    def test_noiseless_isi_small_for_alpha_ge_005(self):
        for a in (0.05, 0.10):
            link = CommLink("M-QAM", 16, a)
            out = link.run(generate_random_bits(2000, 4, 5), 80.0, 1)
            evm = np.sqrt(np.mean(np.abs(out.symbols_rx - out.symbols_tx) ** 2))
            self.assertLess(evm, 0.01, f"α={a}: EVM {evm:.4f}")
            self.assertEqual(out.n_errors, 0)

    def test_ber_matches_theory_small_alpha(self):
        for a in (0.05, 0.10):
            ratio, n = _ber_ratio("M-QAM", 16, a, 9.0)
            self.assertTrue(0.88 < ratio < 1.15, f"α={a}: razón {ratio:.2f} ({n} errores)")

    def test_alpha_zero_has_bounded_residual_isi(self):
        """α = 0: la ISI residual del filtro finito debe estar acotada."""
        link = CommLink("M-QAM", 16, 0.0)
        out = link.run(generate_random_bits(2000, 4, 5), 80.0, 1)
        evm = np.sqrt(np.mean(np.abs(out.symbols_rx - out.symbols_tx) ** 2))
        self.assertLess(evm, 0.06)
        ratio, _ = _ber_ratio("M-QAM", 16, 0.0, 9.0, min_errors=800)
        self.assertLess(ratio, 1.6)

    def test_advisory_message(self):
        from utils.validators import roll_off_advisory
        self.assertIsNotNone(roll_off_advisory("0"))
        self.assertIsNotNone(roll_off_advisory(0.02))
        self.assertIsNone(roll_off_advisory(0.35))
        self.assertIsNone(roll_off_advisory("abc"))


class TestNoiseCalibrationTheoreticalEs(unittest.TestCase):
    """Eb/N0 definido con la Es teórica."""

    def test_variance_with_pam_and_few_symbols(self):
        link = CommLink("M-PAM", 4, 0.35)
        tx = link.transmit(generate_random_bits(30000, 2, 9))
        ref = link.receive(tx, 80.0, 3).symbols_rx
        for eb_n0_db in (3.0, 6.0):
            r = link.receive(tx, eb_n0_db, 3).symbols_rx - ref
            n0 = (1.0 / 2) / 10 ** (eb_n0_db / 10)
            self.assertAlmostEqual(r.real.var() / (n0 / 2), 1.0, delta=0.05)


class TestOwnDSP(unittest.TestCase):
    """FFT y convolución propias frente a numpy como referencia."""

    def test_fft_matches_reference(self):
        from utils.dsp import fft, ifft
        rng = np.random.default_rng(0)
        for n in (1, 2, 8, 256, 4096):
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            np.testing.assert_allclose(fft(x), np.fft.fft(x), atol=1e-9)
            np.testing.assert_allclose(ifft(fft(x)), x, atol=1e-9)
        with self.assertRaises(ValueError):
            fft(np.zeros(6))

    def test_convolution_matches_reference(self):
        from utils.dsp import _convolve_direct, _convolve_fft
        rng = np.random.default_rng(1)
        for nx, nh in ((50, 7), (7, 50), (1000, 161)):
            x, h = rng.normal(size=nx), rng.normal(size=nh)
            xc = x + 1j * rng.normal(size=nx)
            np.testing.assert_allclose(_convolve_direct(x, h), np.convolve(x, h), atol=1e-9)
            np.testing.assert_allclose(_convolve_fft(x, h), np.convolve(x, h), atol=1e-9)
            np.testing.assert_allclose(_convolve_fft(xc, h), np.convolve(xc, h), atol=1e-9)

    def test_spectrum_helpers(self):
        from utils.dsp import fft_freqs, fft_shift
        np.testing.assert_allclose(fft_freqs(16), np.fft.fftfreq(16))
        np.testing.assert_allclose(fft_shift(np.arange(16)), np.fft.fftshift(np.arange(16)))


class TestEyeQBranch(unittest.TestCase):
    """El ojo de una constelación real no dibuja la rama Q."""

    def test_show_q_flag(self):
        from matplotlib.figure import Figure
        from plots.eye_diagram import plot_eye_diagram
        rng = np.random.default_rng(2)
        sig = rng.normal(size=800) + 1j * 0.5 * rng.normal(size=800)
        lines = {}
        for flag in (True, False):
            ax = Figure().add_subplot(111)
            plot_eye_diagram(ax, sig, 8, n_traces=20, show_q=flag)
            lines[flag] = len(ax.lines)
        self.assertGreater(lines[True], lines[False])


class TestRollOffAdvisory(unittest.TestCase):
    """Aviso de roll-off por esquema y orden, y punto de operación sin errores."""

    def test_advisory_depends_on_scheme_and_order(self):
        from utils.validators import roll_off_advisory
        self.assertIsNotNone(roll_off_advisory(0.0, "M-PAM", 4))
        self.assertIsNotNone(roll_off_advisory(0.015, "M-QAM", 64))
        self.assertIsNotNone(roll_off_advisory(0.0, "M-PSK", 8))
        self.assertIsNone(roll_off_advisory(0.02, "M-PAM", 4))
        self.assertIsNone(roll_off_advisory(0.35, "M-QAM", 256))
        self.assertIsNone(roll_off_advisory(0.015, "M-QAM", 4))
        soft = roll_off_advisory(0.0, "M-QAM", 4)
        strong = roll_off_advisory(0.0, "M-PAM", 4)
        self.assertIsNotNone(soft)
        self.assertNotEqual(soft, strong)
        self.assertIsNone(roll_off_advisory("abc", "M-PAM", 4))
        self.assertIsNone(roll_off_advisory(1.5, "M-PAM", 4))

    def test_ber_plot_zero_errors_operating_point(self):
        """Con 0 errores se dibuja un triángulo en 1/bits (antes no se dibujaba nada)."""
        from matplotlib.figure import Figure
        from plots.ber_curve import plot_ber_curve
        x = np.linspace(0, 10, 11)
        th = BERCalculator.ber_theoretical("M-QAM", 4, x)
        ax = Figure().add_subplot(111)
        plot_ber_curve(ax, x, th, operating_point=(10.0, 0.0, 4000))
        labels = [l.get_label() for l in ax.get_lines()]
        self.assertTrue(any("0 errores" in t for t in labels), labels)
        ys = [l.get_ydata()[0] for l in ax.get_lines() if "0 errores" in l.get_label()]
        self.assertAlmostEqual(ys[0], 1 / 4000)
        ax2 = Figure().add_subplot(111)
        plot_ber_curve(ax2, x, th, operating_point=(10.0, 2e-3, 4000))
        self.assertTrue(any(l.get_label() == "Punto de operación" for l in ax2.get_lines()))

    def test_alpha_small_pam_deviates_but_002_fits(self):
        """Documenta lo medido: 4-PAM con α=0 se separa de la teoría y con α=0.02 ajusta."""
        r0, _ = _ber_ratio("M-PAM", 4, 0.0, 13.0, min_errors=100)
        r2, _ = _ber_ratio("M-PAM", 4, 0.02, 13.0, min_errors=100)
        self.assertGreater(r0, 1.5)
        self.assertTrue(0.7 < r2 < 1.4, r2)

    def test_analysis_mentions_statistical_uncertainty(self):
        from analysis.comparison import generate_analysis
        fake_sim = {"diff": {"4PAM-4QAM": [3.5, 3.7, 3.9, float("nan"), float("nan"), float("nan")]}}
        text = generate_analysis(theoretical_table(), fake_sim)
        self.assertIn("Incertidumbre estadística", text)


try:
    import PyQt5
    _HAS_QT = True
except ImportError:
    _HAS_QT = False


@unittest.skipUnless(_HAS_QT, "PyQt5 no instalado: se omite la prueba de humo de la GUI")
class TestGuiSmoke(unittest.TestCase):
    """Prueba de humo de la GUI sin pantalla (offscreen)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import matplotlib
        matplotlib.use("Agg")
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_controls_validate_and_hint(self):
        from gui.controls_panel import ControlsPanel
        panel = ControlsPanel()
        panel.set_field_text("alpha", "1.7")
        self.assertIsNone(panel.collect(show_dialog=False))
        panel.set_field_text("alpha", "0.35")
        panel.set_scheme("M-QAM")
        panel.set_field_text("m", "8")
        self.assertIsNone(panel.collect(show_dialog=False))
        panel.set_field_text("m", "16")
        params = panel.collect(show_dialog=False)
        self.assertIsNotNone(params)
        self.assertEqual((params.scheme, params.m), ("M-QAM", 16))
        panel.set_scheme("M-PAM")
        panel.set_field_text("m", "4")
        panel.set_field_text("alpha", "0")
        self.assertIn("ISI", panel._alpha_hint.text())

    def test_main_window_point_plots(self):
        from core.simulation_runner import SimulationParams, SimulationRunner
        from gui.main_window import MainWindow, format_ber
        win = MainWindow()
        p = SimulationParams(scheme="M-QAM", m=4, alpha=0.35, eb_n0_steps=3,
                             eb_n0_min=0, eb_n0_max=10, max_symbols=2000, min_errors=5)
        res = SimulationRunner().run(p)
        win._plots.update_point_plots(res, p)
        win._controls.update_metrics(res, p)
        self.assertIn("BER", format_ber(res))
        win._pending = None
        win.close()

    def test_comparison_dialog_closes_with_running_thread(self):
        from PyQt5.QtCore import QCoreApplication
        from gui.comparison_dialog import ComparisonDialog
        dlg = ComparisonDialog(None, alpha=0.35)
        dlg._max_symbols.setText("1000")
        dlg._start_simulation()
        self.assertIsNotNone(dlg._thread)
        dlg.show()
        dlg.close()
        QCoreApplication.processEvents()
        self.assertIsNone(dlg._thread)


if __name__ == "__main__":
    unittest.main(verbosity=2)
