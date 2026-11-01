"""
main_window.py
==============
Ventana principal de la aplicación (QMainWindow).

Arquitectura de la GUI:
  MainWindow
    ├── ControlsPanel  (izquierda) — entrada de parámetros y métricas
    └── PlotsPanel     (derecha)   — cuatro gráficas matplotlib

La simulación corre en un QThread separado para no bloquear la interfaz
mientras se calculan los puntos de la curva BER (que puede tardar
varios segundos con muchos símbolos).

Manejo de errores:
  Cualquier excepción dentro del hilo de simulación se captura y
  se muestra en un QMessageBox sin crashear la aplicación (RNF1).
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QStatusBar, QMessageBox, QLabel,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

from gui.controls_panel     import ControlsPanel
from gui.plots_panel        import PlotsPanel
from core.simulation_runner import SimulationRunner, SimulationParams


# ── Worker de simulación (hilo separado) ─────────────────────────────────────

class SimulationWorker(QObject):
    """
    Ejecuta la simulación en un hilo secundario para no congelar la GUI.

    Señales
    -------
    finished  : emitida con (results, params) al completar.
    error     : emitida con un mensaje de error si algo falla.
    """
    finished = pyqtSignal(object, object)    # (SimulationResults, SimulationParams)
    error    = pyqtSignal(str)

    def __init__(self, params: SimulationParams):
        super().__init__()
        self._params = params

    def run(self):
        try:
            runner  = SimulationRunner()
            results = runner.run(self._params)
            self.finished.emit(results, self._params)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Ventana principal ─────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Ventana principal del simulador de comunicaciones digitales.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador de Sistema de Comunicación Digital")
        self.setMinimumSize(1200, 750)

        self._thread = None
        self._worker = None

        self._build_ui()

    def _build_ui(self):
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)

        # Layout principal horizontal con splitter redimensionable
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)

        # Panel izquierdo: controles
        self._controls = ControlsPanel()
        self._controls.setFixedWidth(280)
        self._controls.simulate_requested.connect(self._on_simulate_requested)

        # Panel derecho: gráficas
        self._plots = PlotsPanel()

        splitter.addWidget(self._controls)
        splitter.addWidget(self._plots)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        h_layout.addWidget(splitter)

        # Barra de estado
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(
            "Listo — Configure los parámetros y presione Simular"
        )

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_simulate_requested(self, params: SimulationParams):
        """Inicia la simulación en un hilo secundario."""
        self._status_bar.showMessage(
            f"Simulando {params.scheme} M={params.m} "
            f"α={params.alpha:.2f}  Eb/N0={params.eb_n0_db:.1f} dB ..."
        )

        # Crear hilo y worker
        self._thread = QThread()
        self._worker = SimulationWorker(params)
        self._worker.moveToThread(self._thread)

        # Conexiones
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_simulation_finished)
        self._worker.error.connect(self._on_simulation_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_simulation_finished(self, results, params):
        """Actualiza la GUI cuando la simulación termina exitosamente."""
        self._plots.update_plots(results, params)
        self._controls.update_metrics(results)

        self._status_bar.showMessage(
            f"✔ Simulación completada — "
            f"BER = {results.ber_simulated_point:.2e}  |  "
            f"η = {results.spectral_efficiency:.3f} bit/s/Hz  |  "
            f"BW = {results.bandwidth_norm:.3f}·Rs"
        )

    def _on_simulation_error(self, error_msg: str):
        """Muestra el error sin crashear la aplicación (RNF1)."""
        self._controls.update_metrics(
            type("R", (), {
                "ber_simulated_point": 0,
                "spectral_efficiency": 0,
                "symbol_energy": 0,
                "bandwidth_norm": 0,
                "signal_power": 0,
            })()
        )
        QMessageBox.critical(
            self,
            "Error en la simulación",
            f"Ocurrió un error durante la simulación:\n\n{error_msg}\n\n"
            "Revise los parámetros e intente nuevamente.",
        )
        self._status_bar.showMessage("Error en la simulación — revise los parámetros")
