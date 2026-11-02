

from __future__ import annotations

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAction, QMainWindow, QMessageBox, QProgressBar, QSplitter, QStatusBar,
    QWidget, QHBoxLayout,
)

from core.simulation_runner import SimulationParams, SimulationRunner
from gui.comparison_dialog import ComparisonDialog
from gui.controls_panel import ControlsPanel
from gui.plots_panel import PlotsPanel


class SimulationWorker(QObject):
    """Ejecuta la simulación en un hilo secundario."""

    point_ready = pyqtSignal(object, object)       # (results, params)
    progress = pyqtSignal(int, int)                # (puntos hechos, total)
    finished = pyqtSignal(object, object, bool)    # (results, params, completada)
    error = pyqtSignal(str)

    def __init__(self, params: SimulationParams):
        super().__init__()
        self._params = params
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            runner = SimulationRunner()
            results = runner.run_point(self._params)
            self.point_ready.emit(results, self._params)
            done = runner.run_curve(self._params, results,
                                    progress=self.progress.emit,
                                    cancelled=lambda: self._cancel)
            self.finished.emit(results, self._params, done)
        except Exception as exc:           # noqa: BLE001  (se informa al usuario)
            self.error.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador de Sistema de Comunicación Digital")
        self.setMinimumSize(1200, 750)

        self._thread: QThread | None = None
        self._worker: SimulationWorker | None = None
        self._pending: SimulationParams | None = None
        self._comparison_dialog: ComparisonDialog | None = None

        self._build_ui()
        self._build_menu()

    # ── Construcción ─────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        self._controls = ControlsPanel()
        self._controls.setMinimumWidth(340)
        self._controls.setMaximumWidth(420)
        self._controls.simulate_requested.connect(self._on_simulate_requested)
        self._controls.compare_requested.connect(self._open_comparison)

        self._plots = PlotsPanel()
        splitter.addWidget(self._controls)
        splitter.addWidget(self._plots)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        h_layout.addWidget(splitter)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.hide()
        self._status_bar.addPermanentWidget(self._progress)
        self._status_bar.showMessage("Listo — configure los parámetros y presione Simular")

    def _build_menu(self):
        tools = self.menuBar().addMenu("Herramientas")
        act = QAction("Comparar 4PAM / 4QAM / 4PSK (tabla de entrega)…", self)
        act.triggered.connect(self._open_comparison)
        tools.addAction(act)

    # ── Simulación ───────────────────────────────────────────────────────────

    def _on_simulate_requested(self, params: SimulationParams, automatic: bool):
        if self._thread is not None:               # hay una simulación en curso
            self._pending = params
            if self._worker is not None:
                self._worker.cancel()
            return
        self._start(params)

    def _start(self, params: SimulationParams):
        self._status_bar.showMessage(
            f"Simulando {params.scheme} M={params.m}  α={params.alpha:.2f}  "
            f"Eb/N0={params.eb_n0_db:.1f} dB …")
        self._controls.set_busy(True)

        self._thread = QThread()
        self._worker = SimulationWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point_ready.connect(self._on_point_ready)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_curve_finished)
        self._worker.error.connect(self._on_simulation_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _on_point_ready(self, results, params):
        if self._pending is not None:              # ya hay parámetros más nuevos
            return
        self._plots.update_point_plots(results, params)
        self._controls.update_metrics(results, params)
        self._progress.setRange(0, params.eb_n0_steps)
        self._progress.setValue(0)
        self._progress.show()
        self._status_bar.showMessage(
            f"Punto de operación listo — BER = {results.ber_simulated_point:.2e}  |  "
            f"η = {results.spectral_efficiency:.3f} bit/s/Hz  |  calculando curva BER…")

    def _on_progress(self, done: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_curve_finished(self, results, params, completed: bool):
        if completed and self._pending is None:
            self._plots.update_curve_plot(results, params)
            self._status_bar.showMessage(
                f"✔ Simulación completada — BER = {results.ber_simulated_point:.2e}  |  "
                f"η = {results.spectral_efficiency:.3f} bit/s/Hz  |  "
                f"B = {results.bandwidth_norm:.3f}·Rs")

    def _on_simulation_error(self, message: str):
        QMessageBox.critical(
            self, "Error en la simulación",
            f"Ocurrió un error durante la simulación:\n\n{message}\n\n"
            "Revise los parámetros e intente nuevamente.")
        self._status_bar.showMessage("Error en la simulación — revise los parámetros")

    def _on_thread_done(self):
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._progress.hide()
        self._controls.set_busy(False)
        if self._pending is not None:
            params, self._pending = self._pending, None
            self._start(params)

    # ── Comparación (tabla de entrega) ───────────────────────────────────────

    def _open_comparison(self):
        alpha = self._controls.current_alpha()
        self._comparison_dialog = ComparisonDialog(self, alpha=alpha)
        self._comparison_dialog.show()

    def closeEvent(self, event):                   # noqa: N802 (nombre de Qt)
        """Cancela y espera el hilo antes de cerrar para evitar cierres bruscos."""
        self._pending = None
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(event)
