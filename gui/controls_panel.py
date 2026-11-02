"""
controls_panel.py
=================
Panel de controles (lado izquierdo).

Todos los parámetros numéricos se ingresan en cajas de texto y se validan con
utils/validators.py ANTES de simular.  Si hay un valor no permitido (por
ejemplo roll-off 1.7, M=8 en M-QAM, texto no numérico) se muestra un mensaje
bajo el campo y, al pulsar "Simular", un cuadro de diálogo; la aplicación
nunca se detiene ni se cae (RNF1).

Actualización inmediata (RF1): con la casilla "Actualizar automáticamente"
la simulación se relanza ~0.6 s después del último cambio válido.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from core.simulation_runner import SimulationParams
from utils.constants import (
    CURVE_MAX_SYMBOLS, CURVE_MIN_ERRORS, DEFAULT_EB_N0_DB, DEFAULT_EB_N0_MAX,
    DEFAULT_EB_N0_MIN, DEFAULT_EB_N0_STEPS, DEFAULT_M, DEFAULT_NUM_SYMBOLS,
    DEFAULT_ROLL_OFF, DEFAULT_SAMPLES_PER_SYM, DEFAULT_SYMBOL_RATE_BAUD,
    MODULATION_SCHEMES, VALID_ORDERS,
)
from utils.validators import (
    roll_off_advisory, validate_modulation_order, validate_simulation_params,
)

AUTO_UPDATE_DELAY_MS = 600
ERROR_STYLE = "color: #C62828; font-size: 10px;"
HINT_STYLE = "color: gray; font-size: 10px;"

# clave de error del validador → campo donde se muestra
_ERROR_FIELD = {"eb_n0_range": "eb_n0_min"}


class ControlsPanel(QWidget):
    """
    Señales
    -------
    simulate_requested(params, automatic) : parámetros válidos; `automatic`
        indica que vino de la actualización automática.
    compare_requested() : abrir la comparación 4PAM/4QAM/4PSK (tabla de entrega).
    """

    simulate_requested = pyqtSignal(object, bool)
    compare_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edits: dict[str, QLineEdit] = {}
        self._errors: dict[str, QLabel] = {}
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(AUTO_UPDATE_DELAY_MS)
        self._auto_timer.timeout.connect(self._on_auto_timeout)
        self._build_ui()

    # ── Construcción ─────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Parámetros del Sistema")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self._separator())

        # Modulación
        mod_group = QGroupBox("Modulación")
        mod_form = QFormLayout(mod_group)
        self._scheme_combo = QComboBox()
        self._scheme_combo.addItems(MODULATION_SCHEMES)
        self._scheme_combo.setCurrentText("M-QAM")
        self._scheme_combo.currentTextChanged.connect(self._on_scheme_changed)
        mod_form.addRow("Esquema:", self._scheme_combo)
        self._add_field(mod_form, "m", "Orden M:", str(DEFAULT_M),
                        "Número de símbolos de la constelación")
        self._m_hint = QLabel()
        self._m_hint.setStyleSheet(HINT_STYLE)
        mod_form.addRow("", self._m_hint)
        self._add_field(mod_form, "alpha", "Roll-off α:", f"{DEFAULT_ROLL_OFF:g}",
                        "Factor de roll-off del coseno alzado: 0 ≤ α ≤ 1")
        self._alpha_hint = QLabel()
        self._alpha_hint.setStyleSheet(HINT_STYLE)
        self._alpha_hint.setWordWrap(True)
        mod_form.addRow("", self._alpha_hint)
        self._edits["alpha"].textChanged.connect(self._update_alpha_hint)
        layout.addWidget(mod_group)

        # Canal / punto de operación
        ch_group = QGroupBox("Canal y punto de operación")
        ch_form = QFormLayout(ch_group)
        self._add_field(ch_form, "eb_n0_db", "Eb/N0 [dB]:", f"{DEFAULT_EB_N0_DB:g}",
                        "Relación Eb/N0 del punto de operación (-10 a 40 dB)")
        self._add_field(ch_form, "n_symbols", "Nº símbolos:", str(DEFAULT_NUM_SYMBOLS),
                        "Símbolos para constelación, ojo, espectro y BER puntual")
        self._add_field(ch_form, "symbol_rate_baud", "Rs [baudios]:",
                        f"{DEFAULT_SYMBOL_RATE_BAUD:g}",
                        "Tasa de símbolo (para ancho de banda en Hz y tasa de bit)")
        layout.addWidget(ch_group)

        # Curva BER
        curve_group = QGroupBox("Curva BER (Monte Carlo)")
        curve_form = QFormLayout(curve_group)
        self._add_field(curve_form, "eb_n0_min", "Eb/N0 mín [dB]:", f"{DEFAULT_EB_N0_MIN:g}", "")
        self._add_field(curve_form, "eb_n0_max", "Eb/N0 máx [dB]:", f"{DEFAULT_EB_N0_MAX:g}", "")
        self._add_field(curve_form, "eb_n0_steps", "Nº de puntos:", str(DEFAULT_EB_N0_STEPS), "")
        self._add_field(curve_form, "min_errors", "Errores mínimos:", str(CURVE_MIN_ERRORS),
                        "Se simula cada punto hasta acumular estos errores")
        self._add_field(curve_form, "max_symbols", "Máx. símbolos/punto:", str(CURVE_MAX_SYMBOLS),
                        "Tope de símbolos por punto (BER mínima medible ≈ 1/(bits))")
        layout.addWidget(curve_group)

        # Botones
        self._auto_chk = QCheckBox("Actualizar automáticamente")
        self._auto_chk.setChecked(True)
        self._auto_chk.setToolTip("Relanza la simulación al cambiar un parámetro válido")
        layout.addWidget(self._auto_chk)

        self._simulate_btn = QPushButton("▶  Simular")
        self._simulate_btn.setFixedHeight(38)
        self._simulate_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #1976D2; }"
            "QPushButton:disabled { background-color: #90CAF9; }")
        self._simulate_btn.clicked.connect(self._on_simulate_clicked)
        layout.addWidget(self._simulate_btn)

        self._compare_btn = QPushButton("Comparar 4PAM / 4QAM / 4PSK (tabla de entrega)")
        self._compare_btn.clicked.connect(self.compare_requested.emit)
        layout.addWidget(self._compare_btn)
        layout.addWidget(self._separator())

        # Métricas
        metrics = QGroupBox("Métricas del Sistema")
        mform = QFormLayout(metrics)
        self._metric_labels: dict[str, QLabel] = {}
        for key, text in (
            ("ber", "BER simulada:"), ("ber_th", "BER teórica:"), ("evm", "EVM rms:"),
            ("eta", "Ef. espectral:"), ("rb", "Tasa de bit Rb:"),
            ("es", "Energía de símbolo Es:"), ("eb", "Energía de bit Eb:"),
            ("bw", "Ancho de banda B:"), ("power", "Potencia señal modulada:"),
        ):
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-weight: bold; color: #1565C0;")
            self._metric_labels[key] = lbl
            mform.addRow(text, lbl)
        layout.addWidget(metrics)
        layout.addStretch()

        self._on_scheme_changed(self._scheme_combo.currentText())

    def _add_field(self, form: QFormLayout, key: str, label: str, default: str, tooltip: str):
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        edit = QLineEdit(default)
        edit.setToolTip(tooltip)
        err = QLabel("")
        err.setStyleSheet(ERROR_STYLE)
        err.setWordWrap(True)
        err.hide()
        v.addWidget(edit)
        v.addWidget(err)
        edit.textChanged.connect(self._schedule_auto)
        edit.editingFinished.connect(self._validate_inline)
        form.addRow(label, box)
        self._edits[key] = edit
        self._errors[key] = err

    @staticmethod
    def _separator() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep

    # ── Lectura y validación de parámetros ───────────────────────────────────

    def set_field_text(self, key: str, text: str) -> None:
        self._edits[key].setText(text)

    def set_scheme(self, scheme: str) -> None:
        self._scheme_combo.setCurrentText(scheme)

    def collect(self, show_dialog: bool = False):
        """
        Valida todos los campos.  Retorna SimulationParams o None.
        Los mensajes se muestran bajo cada campo; con show_dialog=True además
        se abre un QMessageBox con la lista de problemas.
        """
        t = {k: e.text() for k, e in self._edits.items()}
        values, errors = validate_simulation_params(
            self._scheme_combo.currentText(), t["m"], t["alpha"], t["n_symbols"],
            t["eb_n0_db"], t["eb_n0_min"], t["eb_n0_max"], t["eb_n0_steps"],
            t["symbol_rate_baud"], t["min_errors"], t["max_symbols"],
            DEFAULT_SAMPLES_PER_SYM)
        self._show_errors(errors)
        if errors:
            if show_dialog:
                QMessageBox.warning(self, "Valores no permitidos",
                                    "Corrija los siguientes parámetros:\n\n• "
                                    + "\n• ".join(errors.values()))
            return None
        return SimulationParams(**values)

    def _show_errors(self, errors: dict) -> None:
        for lbl in self._errors.values():
            lbl.hide()
            lbl.setText("")
        for key, msg in errors.items():
            target = _ERROR_FIELD.get(key, key)
            if target in self._errors:
                self._errors[target].setText(msg)
                self._errors[target].show()

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_scheme_changed(self, scheme: str):
        valid = VALID_ORDERS.get(scheme, [])
        self._m_hint.setText("Valores válidos: " + ", ".join(str(v) for v in valid))
        ok, _, _ = validate_modulation_order(scheme, self._edits["m"].text())
        if not ok and valid:
            self._edits["m"].setText(str(DEFAULT_M if DEFAULT_M in valid else valid[0]))
        self._schedule_auto()

    def _update_alpha_hint(self, text: str):
        self._alpha_hint.setText(roll_off_advisory(text) or "")

    def _validate_inline(self):
        self.collect(show_dialog=False)

    def _schedule_auto(self, *_):
        if hasattr(self, "_auto_chk") and self._auto_chk.isChecked():
            self._auto_timer.start()

    def _on_auto_timeout(self):
        params = self.collect(show_dialog=False)
        if params is not None:
            self.simulate_requested.emit(params, True)

    def _on_simulate_clicked(self):
        params = self.collect(show_dialog=True)
        if params is not None:
            self.simulate_requested.emit(params, False)

    # ── API pública ──────────────────────────────────────────────────────────

    def set_busy(self, busy: bool) -> None:
        self._simulate_btn.setEnabled(not busy)
        self._simulate_btn.setText("⏳  Simulando…" if busy else "▶  Simular")

    def current_alpha(self, default: float = DEFAULT_ROLL_OFF) -> float:
        params = self.collect(show_dialog=False)
        return params.alpha if params else default

    def update_metrics(self, results, params) -> None:
        m = self._metric_labels
        if results.errors_point == 0:
            bound = 1.0 / max(results.bits_point, 1)
            m["ber"].setText(f"< {bound:.1e}  (0 errores / {results.bits_point} bits)")
        else:
            m["ber"].setText(f"{results.ber_simulated_point:.3e}  "
                             f"({results.errors_point} errores / {results.bits_point} bits)")
        m["ber_th"].setText(f"{results.ber_theoretical_point:.3e}")
        m["evm"].setText(f"{results.evm_rms:.4f}")
        m["eta"].setText(f"{results.spectral_efficiency:.3f} bit/s/Hz")
        m["rb"].setText(f"{results.bit_rate_bps:,.0f} bit/s")
        m["es"].setText(f"{results.symbol_energy:.4f} u.e. (medida {results.symbol_energy_measured:.4f})")
        m["eb"].setText(f"{results.bit_energy:.4f} u.e.")
        m["bw"].setText(f"{results.bandwidth_norm:.3f}·Rs = {results.bandwidth_hz:,.1f} Hz")
        m["power"].setText(f"{results.signal_power:.4f} Es/Ts (teórica {results.signal_power_theory:.4f}) "
                           f"= {results.signal_power_db:.2f} dB;  {results.signal_power_per_s:,.1f} u.e./s")
