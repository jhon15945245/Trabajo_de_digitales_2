"""
controls_panel.py
=================
Panel de controles de la GUI (lado izquierdo).

Contiene todos los widgets de entrada:
  - Selector de esquema de modulación (QComboBox)
  - Selector de orden M (QSpinBox con valores válidos)
  - Slider/SpinBox de factor roll-off α
  - Slider/SpinBox de Eb/N0 (punto de operación)
  - Número de símbolos
  - Botón "Simular"
  - Panel de métricas (solo lectura)

Valida cada entrada en tiempo real usando validators.py
y muestra mensajes de error sin crashear la app (RNF1).
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox,
    QFormLayout, QMessageBox, QSlider, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from utils.constants import (
    MODULATION_SCHEMES, VALID_ORDERS,
    DEFAULT_M, DEFAULT_ROLL_OFF,
    DEFAULT_NUM_SYMBOLS,
)
from utils.validators import (
    validate_modulation_order, validate_roll_off, validate_num_symbols,
)
from core.simulation_runner import SimulationParams


class ControlsPanel(QWidget):
    """
    Panel izquierdo con todos los controles de parámetros.

    Señales
    -------
    simulate_requested : emitida cuando el usuario presiona "Simular",
                         lleva un SimulationParams válido.
    """

    simulate_requested = pyqtSignal(object)   # SimulationParams

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # ── Construcción de la interfaz ──────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # ── Título ────────────────────────────────────────────────────────
        title = QLabel("Parámetros del Sistema")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(self._make_separator())

        # ── Grupo: Modulación ─────────────────────────────────────────────
        mod_group = QGroupBox("Modulación")
        mod_form  = QFormLayout(mod_group)

        self._scheme_combo = QComboBox()
        self._scheme_combo.addItems(MODULATION_SCHEMES)
        self._scheme_combo.setCurrentText("M-QAM")
        self._scheme_combo.currentTextChanged.connect(self._on_scheme_changed)
        mod_form.addRow("Esquema:", self._scheme_combo)

        self._m_spin = QSpinBox()
        self._m_spin.setMinimum(2)
        self._m_spin.setMaximum(256)
        self._m_spin.setValue(DEFAULT_M)
        self._m_spin.setToolTip("Orden de la modulación (número de símbolos)")
        self._m_spin.editingFinished.connect(self._validate_m)
        mod_form.addRow("Orden M:", self._m_spin)

        self._m_hint = QLabel("Valores válidos: 4, 16, 64, 256")
        self._m_hint.setStyleSheet("color: gray; font-size: 10px;")
        mod_form.addRow("", self._m_hint)

        layout.addWidget(mod_group)

        # ── Grupo: Canal ──────────────────────────────────────────────────
        ch_group = QGroupBox("Canal y Filtro")
        ch_form  = QFormLayout(ch_group)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setValue(DEFAULT_ROLL_OFF)
        self._alpha_spin.setDecimals(2)
        self._alpha_spin.setToolTip("Factor roll-off del filtro coseno alzado (0 a 1)")
        ch_form.addRow("Roll-off α:", self._alpha_spin)

        self._eb_n0_spin = QDoubleSpinBox()
        self._eb_n0_spin.setRange(-10.0, 40.0)
        self._eb_n0_spin.setSingleStep(1.0)
        self._eb_n0_spin.setValue(10.0)
        self._eb_n0_spin.setDecimals(1)
        self._eb_n0_spin.setSuffix(" dB")
        self._eb_n0_spin.setToolTip("Relación Eb/N0 para el punto de operación")
        ch_form.addRow("Eb/N0:", self._eb_n0_spin)

        layout.addWidget(ch_group)

        # ── Grupo: Simulación ─────────────────────────────────────────────
        sim_group = QGroupBox("Simulación")
        sim_form  = QFormLayout(sim_group)

        self._n_sym_spin = QSpinBox()
        self._n_sym_spin.setRange(100, 100_000)
        self._n_sym_spin.setSingleStep(500)
        self._n_sym_spin.setValue(DEFAULT_NUM_SYMBOLS)
        self._n_sym_spin.setToolTip("Mayor número = BER más precisa, pero más lento")
        sim_form.addRow("Nº símbolos:", self._n_sym_spin)

        layout.addWidget(sim_group)

        # ── Botón Simular ─────────────────────────────────────────────────
        self._simulate_btn = QPushButton("▶  Simular")
        self._simulate_btn.setFixedHeight(40)
        self._simulate_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-size: 13px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:pressed { background-color: #0D47A1; }"
            "QPushButton:disabled { background-color: #90CAF9; }"
        )
        self._simulate_btn.clicked.connect(self._on_simulate_clicked)
        layout.addWidget(self._simulate_btn)

        layout.addWidget(self._make_separator())

        # ── Panel de métricas ─────────────────────────────────────────────
        metrics_group = QGroupBox("Métricas del Sistema")
        metrics_form  = QFormLayout(metrics_group)

        self._lbl_ber      = self._make_metric_label()
        self._lbl_spec_eff = self._make_metric_label()
        self._lbl_sym_e    = self._make_metric_label()
        self._lbl_bw       = self._make_metric_label()
        self._lbl_power    = self._make_metric_label()

        metrics_form.addRow("BER simulada:", self._lbl_ber)
        metrics_form.addRow("Ef. espectral:", self._lbl_spec_eff)
        metrics_form.addRow("Energía símbolo:", self._lbl_sym_e)
        metrics_form.addRow("Ancho de banda:", self._lbl_bw)
        metrics_form.addRow("Potencia señal:", self._lbl_power)

        layout.addWidget(metrics_group)
        layout.addStretch()

        # Actualizar hint al inicio
        self._on_scheme_changed("M-QAM")

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_scheme_changed(self, scheme: str):
        """Actualiza el hint de órdenes válidos cuando cambia el esquema."""
        valid = VALID_ORDERS.get(scheme, [])
        self._m_hint.setText(f"Valores válidos: {', '.join(str(v) for v in valid)}")
        # Ajustar M al primero válido si el actual no es válido
        current_m = self._m_spin.value()
        if current_m not in valid and valid:
            self._m_spin.setValue(valid[0])

    def _validate_m(self):
        """Valida el orden M al terminar de editar."""
        scheme = self._scheme_combo.currentText()
        m = self._m_spin.value()
        ok, msg = validate_modulation_order(scheme, m)
        if not ok:
            QMessageBox.warning(self, "Valor inválido", msg)
            valid = VALID_ORDERS.get(scheme, [4])
            self._m_spin.setValue(valid[0])

    def _on_simulate_clicked(self):
        """
        Valida todos los parámetros y emite simulate_requested.
        Si hay error de validación, muestra un QMessageBox (no crashea).
        """
        scheme = self._scheme_combo.currentText()
        m      = self._m_spin.value()
        alpha  = self._alpha_spin.value()
        n_sym  = self._n_sym_spin.value()
        eb_n0  = self._eb_n0_spin.value()

        # Validaciones
        ok, msg = validate_modulation_order(scheme, m)
        if not ok:
            QMessageBox.warning(self, "Error de parámetro", msg)
            return

        ok, msg = validate_roll_off(alpha)
        if not ok:
            QMessageBox.warning(self, "Error de parámetro", msg)
            return

        ok, msg = validate_num_symbols(n_sym)
        if not ok:
            QMessageBox.warning(self, "Error de parámetro", msg)
            return

        params = SimulationParams(
            scheme=scheme, m=m, alpha=alpha,
            n_symbols=n_sym, eb_n0_db=eb_n0,
        )

        self._simulate_btn.setEnabled(False)
        self._simulate_btn.setText("⏳  Simulando...")
        self.simulate_requested.emit(params)

    # ── API pública ──────────────────────────────────────────────────────────

    def update_metrics(self, results) -> None:
        """
        Actualiza el panel de métricas con los resultados de la simulación.
        Llamado por MainWindow cuando termina la simulación.
        """
        self._lbl_ber.setText(f"{results.ber_simulated_point:.2e}")
        self._lbl_spec_eff.setText(f"{results.spectral_efficiency:.3f} bit/s/Hz")
        self._lbl_sym_e.setText(f"{results.symbol_energy:.4f}")
        self._lbl_bw.setText(f"{results.bandwidth_norm:.3f} × Rs")
        self._lbl_power.setText(f"{results.signal_power:.4f}")

        self._simulate_btn.setEnabled(True)
        self._simulate_btn.setText("▶  Simular")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_metric_label() -> QLabel:
        lbl = QLabel("—")
        lbl.setStyleSheet("font-weight: bold; color: #1565C0;")
        return lbl

    @staticmethod
    def _make_separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep
