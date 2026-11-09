"""Panel de gráficas (lado derecho): cuadrícula 2×2 de matplotlib (RF3)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from plots.ber_curve import plot_ber_curve
from plots.constellation import plot_constellation
from plots.eye_diagram import plot_eye_diagram
from plots.spectrum import plot_spectrum
from utils.constants import CARRIER_FREQ_NORMALIZED


class PlotsPanel(QWidget):
    """Cuadrícula 2×2 con ojo, constelación, curva BER y espectro (RF3)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Cada espacio de la cuadrícula muestra una parte distinta del sistema.
        self._fig = Figure(figsize=(12, 8))
        self._ax_eye = self._fig.add_subplot(2, 2, 1)
        self._ax_const = self._fig.add_subplot(2, 2, 2)
        self._ax_ber = self._fig.add_subplot(2, 2, 3)
        self._ax_spec = self._fig.add_subplot(2, 2, 4)

        for ax, msg in [
            (self._ax_eye, "Diagrama del Ojo\n(presione Simular)"),
            (self._ax_const, "Constelación\n(presione Simular)"),
            (self._ax_ber, "Curva BER\n(presione Simular)"),
            (self._ax_spec, "Espectro\n(presione Simular)"),
        ]:
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    @property
    def figure(self) -> Figure:
        return self._fig

    def _redraw(self):
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def update_point_plots(self, results, params) -> None:
        """Ojo, constelación, espectro y curva teórica (punto de operación)."""
        # Muestra Q solo si la constelación usa la componente en cuadratura.
        has_q = bool(np.any(np.abs(results.ideal_constellation.imag) > 1e-9))
        plot_eye_diagram(self._ax_eye, results.mf_output, params.sps, n_traces=100,
                         show_q=has_q)
        plot_constellation(self._ax_const, results.symbols_rx, results.ideal_constellation,
                           evm_rms=results.evm_rms)
        plot_spectrum(self._ax_spec, results.baseband_shaped, results.passband_rx,
                      fc_normalized=CARRIER_FREQ_NORMALIZED, sps=params.sps, alpha=params.alpha)
        self.update_curve_plot(results, params)

    def update_curve_plot(self, results, params) -> None:
        plot_ber_curve(
            self._ax_ber, results.eb_n0_array, results.ber_theoretical,
            results.ber_simulated, scheme=params.scheme, m=params.m,
            errors=results.curve_errors, bits=results.curve_bits,
            operating_point=(params.eb_n0_db, results.ber_simulated_point, results.bits_point))
        self._redraw()

    def update_plots(self, results, params) -> None:
        """Actualiza las cuatro gráficas de una vez."""
        self.update_point_plots(results, params)
