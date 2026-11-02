

# ── Esquemas de modulación soportados ────────────────────────────────────────
MODULATION_SCHEMES = ["M-PAM", "M-QAM", "M-PSK"]

# Órdenes válidos por esquema (M-QAM solo cuadrada: M = 4, 16, 64, 256)
VALID_ORDERS = {
    "M-PAM": [2, 4, 8, 16],
    "M-QAM": [4, 16, 64, 256],
    "M-PSK": [2, 4, 8, 16],
}

# ── Parámetros de simulación por defecto ─────────────────────────────────────
DEFAULT_M               = 4        # Orden de modulación por defecto
DEFAULT_ROLL_OFF        = 0.35     # Factor roll-off (alpha) del filtro coseno alzado
DEFAULT_SAMPLES_PER_SYM = 8        # Muestras por símbolo (sobremuestreo, sps)
DEFAULT_NUM_SYMBOLS     = 2000     # Símbolos para constelación / ojo / espectro
DEFAULT_EB_N0_DB        = 10.0     # Punto de operación [dB]
DEFAULT_EB_N0_MIN       = -2.0     # Eb/N0 mínimo [dB] para la curva BER
DEFAULT_EB_N0_MAX       = 20.0     # Eb/N0 máximo [dB] para la curva BER
DEFAULT_EB_N0_STEPS     = 23       # Puntos de la curva BER (paso de 1 dB)
DEFAULT_SYMBOL_RATE_BAUD = 1000.0  # Tasa de símbolo Rs [baudios] (para BW en Hz)

# ── Monte Carlo de la curva BER ──────────────────────────────────────────────
CURVE_MIN_ERRORS     = 100         # Errores mínimos por punto (confianza ~ ±20 %)
CURVE_MAX_SYMBOLS    = 100_000     # Tope de símbolos simulados por punto
CURVE_BLOCK_SYMBOLS  = 10_000      # Tamaño de cada bloque Monte Carlo
MIN_RELIABLE_ERRORS  = 30          # Errores mínimos para usar un punto en interpolaciones

# ── Semilla para reproducibilidad (RNF3) ─────────────────────────────────────
RANDOM_SEED = 42

# ── Límites de validación de interfaz ────────────────────────────────────────
ROLL_OFF_MIN = 0.0
ROLL_OFF_MAX = 1.0
SPS_MIN, SPS_MAX = 6, 32
NUM_SYMBOLS_MIN, NUM_SYMBOLS_MAX = 100, 100_000
EB_N0_LIMIT_MIN, EB_N0_LIMIT_MAX = -10.0, 40.0
EB_N0_STEPS_MIN, EB_N0_STEPS_MAX = 2, 200
SYMBOL_RATE_MIN, SYMBOL_RATE_MAX = 1.0, 1e12
MIN_ERRORS_MIN, MIN_ERRORS_MAX = 1, 10_000
MAX_SYMBOLS_MIN, MAX_SYMBOLS_MAX = 1_000, 5_000_000

# ── Filtro conformador ───────────────────────────────────────────────────────
FILTER_SPAN = 20   # Duración del SRRC en símbolos (±10 símbolos)

# ── Portadora y demodulador ──────────────────────────────────────────────────
CARRIER_FREQ_NORMALIZED = 0.25   # fc / fs   (fc = fs/4)
LPF_MARGIN_RS = 0.5              # Margen [Rs] sobre (1+α)/2·Rs para el corte del LPF
LPF_TAPS_PER_SYMBOL = 8          # Longitud del LPF: 8·sps + 1 coeficientes

# ── Comparación de esquemas (tabla de entrega) ───────────────────────────────
BER_TARGETS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
COMPARISON_SCHEMES = (("4PAM", "M-PAM", 4), ("4QAM", "M-QAM", 4), ("4PSK", "M-PSK", 4))
