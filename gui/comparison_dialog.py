"""Ventana de la tabla de entrega de diferencias de Eb/N0."""

from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from analysis.comparison import (
    generate_analysis, simulate_comparison, table_rows, theoretical_table,
)
from analysis.export import export_results
from plots.comparison import plot_ber_comparison, plot_delta_bars, plot_required_ebn0
from utils.validators import validate_max_symbols


class _SimWorker(QObject):
    """Ejecuta en un hilo las simulaciones Monte Carlo de la tabla de entrega."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, alpha: float, max_symbols: int):
        super().__init__()
        self._alpha, self._max_symbols = alpha, max_symbols
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            sim = simulate_comparison(alpha=self._alpha, max_symbols=self._max_symbols,
                                      progress=self.progress.emit,
                                      cancelled=lambda: self._cancel)
            self.finished.emit(sim)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class ComparisonDialog(QDialog):
    """Ventana de la tabla de entrega: ΔEb/N0 entre 4PAM, 4QAM y 4PSK, gráficas y análisis."""
    def __init__(self, parent=None, alpha: float = 0.35):
        super().__init__(parent)
        self.setWindowTitle("Comparación 4PAM / 4QAM / 4PSK — tabla de entrega")
        self.resize(1000, 700)
        self._alpha = alpha
        self._theory = theoretical_table()
        self._sim: dict | None = None
        self._thread: QThread | None = None
        self._worker: _SimWorker | None = None
        self._build_ui()
        self._refresh()

    @property
    def alpha(self) -> float:
        return self._alpha

    def done(self, result: int) -> None:
        """Se ejecuta al cerrar con Esc, Aceptar/Cancelar o close()."""
        self._stop_thread()
        super().done(result)

    def _stop_thread(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            if self._thread.wait(15000):
                if self._worker is not None:
                    self._worker.deleteLater()
                self._thread.deleteLater()
        self._worker = self._thread = None

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(f"Roll-off α = {self._alpha:g}   |   Máx. símbolos por punto:"))
        self._max_symbols = QLineEdit("400000")
        self._max_symbols.setMaximumWidth(100)
        top.addWidget(self._max_symbols)
        self._sim_btn = QPushButton("Verificar por simulación")
        self._sim_btn.clicked.connect(self._start_simulation)
        top.addWidget(self._sim_btn)
        self._export_btn = QPushButton("Exportar (CSV, MD, PNG, análisis)…")
        self._export_btn.clicked.connect(self._export)
        top.addWidget(self._export_btn)
        top.addStretch()
        layout.addLayout(top)

        self._progress = QProgressBar()
        self._progress.hide()
        layout.addWidget(self._progress)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        table_tab = QWidget()
        tl = QVBoxLayout(table_tab)
        self._table = QTableWidget()
        tl.addWidget(self._table, 2)
        self._analysis = QPlainTextEdit()
        self._analysis.setReadOnly(True)
        tl.addWidget(self._analysis, 3)
        tabs.addTab(table_tab, "Tabla y análisis")

        plots_tab = QWidget()
        pl = QVBoxLayout(plots_tab)
        self._fig = Figure(figsize=(10, 7))
        gs = self._fig.add_gridspec(2, 2)
        self._ax_ber = self._fig.add_subplot(gs[0, :])
        self._ax_bars = self._fig.add_subplot(gs[1, 0])
        self._ax_req = self._fig.add_subplot(gs[1, 1])
        self._canvas = FigureCanvasQTAgg(self._fig)
        pl.addWidget(NavigationToolbar2QT(self._canvas, self))
        pl.addWidget(self._canvas)
        tabs.addTab(plots_tab, "Gráficas")


    def _refresh(self):
        # Actualiza juntos la tabla, el análisis y las gráficas de comparación.
        rows = table_rows(self._theory, self._sim)
        self._table.setRowCount(len(rows) - 1)
        self._table.setColumnCount(len(rows[0]))
        self._table.setHorizontalHeaderLabels(rows[0])
        for i, row in enumerate(rows[1:]):
            for j, text in enumerate(row):
                self._table.setItem(i, j, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
        self._analysis.setPlainText(generate_analysis(self._theory, self._sim))

        plot_ber_comparison(self._ax_ber, sim=self._sim)
        plot_delta_bars(self._ax_bars, self._theory, self._sim)
        plot_required_ebn0(self._ax_req, self._theory)
        self._fig.tight_layout()
        self._canvas.draw_idle()


    def _start_simulation(self):
        if self._thread is not None:
            return
        ok, max_symbols, msg = validate_max_symbols(self._max_symbols.text())
        if not ok:
            QMessageBox.warning(self, "Valor no permitido", msg)
            return
        # La verificación Monte Carlo corre en segundo plano y muestra su avance.
        self._sim_btn.setEnabled(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.show()

        self._thread = QThread()
        self._worker = _SimWorker(self._alpha, max_symbols)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_sim_finished)
        self._worker.error.connect(self._on_sim_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _on_progress(self, done, total):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_sim_finished(self, sim):
        if sim is not None and self.isVisible():
            self._sim = sim
            self._refresh()

    def _on_sim_error(self, message):
        QMessageBox.critical(self, "Error en la simulación", message)

    def _on_thread_done(self):
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = self._thread = None
        self._progress.hide()
        self._sim_btn.setEnabled(True)

    def _export(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de destino")
        if not folder:
            return
        try:
            paths = export_results(folder, self._theory, self._sim)
        except OSError as exc:
            QMessageBox.critical(self, "No se pudo exportar", str(exc))
            return
        QMessageBox.information(self, "Exportación completa",
                                "Archivos creados:\n" + "\n".join(paths))

    def closeEvent(self, event):
        self._stop_thread()
        super().closeEvent(event)
