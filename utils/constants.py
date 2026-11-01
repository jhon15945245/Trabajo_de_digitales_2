"""
constants.py
============
Constantes globales del simulador de comunicaciones digitales.
Centralizar aquí evita "magic numbers" en el resto del código.
"""

# ── Esquemas de modulación soportados ────────────────────────────────────────
MODULATION_SCHEMES = ["M-PAM", "M-QAM", "M-PSK"]

# Órdenes válidos por esquema
VALID_ORDERS = {
    "M-PAM": [2, 4, 8, 16],
    "M-QAM": [4, 16, 64, 256],
    "M-PSK": [2, 4, 8, 16],
}

# ── Parámetros de simulación por defecto ─────────────────────────────────────
DEFAULT_M              = 4          # Orden de modulación por defecto
DEFAULT_ROLL_OFF       = 0.35       # Factor roll-off (alpha) del filtro coseno alzado
DEFAULT_SAMPLES_PER_SYM = 8        # Muestras por símbolo (oversampling)
DEFAULT_NUM_SYMBOLS    = 2000       # Símbolos a simular
DEFAULT_EB_N0_MIN      = -2.0       # Eb/N0 mínimo [dB] para curva BER
DEFAULT_EB_N0_MAX      = 20.0       # Eb/N0 máximo [dB] para curva BER
DEFAULT_EB_N0_STEPS    = 30         # Número de puntos en la curva BER

# ── Semilla para reproducibilidad (RNF3) ─────────────────────────────────────
RANDOM_SEED = 42

# ── Límites de validación de interfaz ────────────────────────────────────────
ROLL_OFF_MIN = 0.0
ROLL_OFF_MAX = 1.0
FILTER_SPAN  = 10   # Duración del filtro en número de símbolos (span)

# ── Frecuencia portadora normalizada ─────────────────────────────────────────
CARRIER_FREQ_NORMALIZED = 0.25   # fc normalizada respecto a la tasa de muestreo
