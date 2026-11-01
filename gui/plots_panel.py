"""
plots_panel.py
==============
Panel de gráficas de la GUI (lado derecho).

Incrusta cuatro subplots de matplotlib dentro de la ventana PyQt5
usando FigureCanvasQTAgg (el puente oficial matplotlib↔Qt).

Las cuatro gráficas (RF3):
  1. Diagrama del ojo
  2. Diagrama de constelación
  3. Curva BER vs Eb/N0
  4. Espectro (banda base + pasabanda)

El layout es una cuadrícula 2×2 con la figura de matplotlib.
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from plots.eye_diagram  import plot_eye_diagram
from plots.constellation import plot_constellation
from plots.ber_curve     import plot_ber_curve
from plots.spectrum      import plot_spectrum
from utils.constants     import CARRIER_FREQ_NORMALIZED


class PlotsPanel(QWidget):
    """
    Panel que contiene las cuatro gráficas requeridas incrustadas en Qt.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Figura matplotlib con 4 subplots (2 filas × 2 columnas)
        self._fig = Figure(figsize=(12, 8), tight_layout=True)
        self._ax_eye    = self._fig.add_subplot(2, 2, 1)
        self._ax_const  = self._fig.add_subplot(2, 2, 2)
        self._ax_ber    = self._fig.add_subplot(2, 2, 3)
        self._ax_spec   = self._fig.add_subplot(2, 2, 4)

        # Mensajes de espera iniciales
        for ax, msg in [
            (self._ax_eye,   "Diagrama del Ojo\n(presione Simular)"),
            (self._ax_const, "Constelación\n(presione Simular)"),
            (self._ax_ber,   "Curva BER\n(presione Simular)"),
            (self._ax_spec,  "Espectro\n(presione Simular)"),
        ]:
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

        # Canvas Qt (puente matplotlib ↔ PyQt5)
        self._canvas  = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    # ── API pública ──────────────────────────────────────────────────────────

    def update_plots(self, results, params) -> None:
        """
        Actualiza las cuatro gráficas con los resultados de la simulación.

        Parámetros
        ----------
        results : SimulationResults — salida del SimulationRunner.
        params  : SimulationParams  — parámetros usados en la simulación.
        """
        # 1. Diagrama del ojo
        plot_eye_diagram(
            self._ax_eye,
            results.baseband_rx,
            params.sps,
            n_traces=80,
        )

        # 2. Diagrama de constelación
        plot_constellation(
            self._ax_const,
            results.symbols_rx,
            results.ideal_constellation,
        )

        # 3. Curva BER
        plot_ber_curve(
            self._ax_ber,
            results.eb_n0_array,
            results.ber_theoretical,
            results.ber_simulated,
            scheme=params.scheme,
            m=params.m,
        )

        # 4. Espectro
        plot_spectrum(
            self._ax_spec,
            results.baseband_shaped,
            results.passband_signal,
            fc_normalized=CARRIER_FREQ_NORMALIZED,
            sps=params.sps,
        )

        # Redibujar el canvas
        self._fig.tight_layout()
        self._canvas.draw()
