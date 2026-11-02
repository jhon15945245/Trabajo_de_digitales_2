"""
validators.py
=============
Validaciones de entrada del usuario (RNF1).

Todas las funciones retornan (ok, valor, mensaje):
    ok      : True si el dato es válido.
    valor   : dato convertido (None si no es válido).
    mensaje : texto para el usuario (None si es válido).

Así la GUI puede mostrar el mensaje sin lanzar excepciones.  El núcleo
también las reutiliza (validate_simulation_params) para no depender de que
la entrada provenga de la interfaz.
"""

from __future__ import annotations

import math

from utils.constants import (
    VALID_ORDERS, MODULATION_SCHEMES,
    ROLL_OFF_MIN, ROLL_OFF_MAX, ROLL_OFF_ADVISORY, SPS_MIN, SPS_MAX,
    NUM_SYMBOLS_MIN, NUM_SYMBOLS_MAX,
    EB_N0_LIMIT_MIN, EB_N0_LIMIT_MAX, EB_N0_STEPS_MIN, EB_N0_STEPS_MAX,
    SYMBOL_RATE_MIN, SYMBOL_RATE_MAX,
    MIN_ERRORS_MIN, MIN_ERRORS_MAX, MAX_SYMBOLS_MIN, MAX_SYMBOLS_MAX,
)


# ── Conversión de texto ──────────────────────────────────────────────────────

def parse_float(text, name: str = "El valor"):
    """Convierte texto a float aceptando coma decimal; rechaza NaN/inf."""
    if isinstance(text, bool):
        return False, None, f"{name} no es un número válido."
    if isinstance(text, str):
        text = text.strip().replace(",", ".")
        if text == "":
            return False, None, f"{name} está vacío. Ingrese un número."
    try:
        value = float(text)
    except (TypeError, ValueError):
        return False, None, f"{name} debe ser un número (se recibió '{text}')."
    if not math.isfinite(value):
        return False, None, f"{name} debe ser un número finito."
    return True, value, None


def parse_int(text, name: str = "El valor"):
    """Convierte texto a entero; rechaza decimales como 4.5."""
    ok, value, msg = parse_float(text, name)
    if not ok:
        return False, None, msg
    if value != int(value):
        return False, None, f"{name} debe ser un número entero (se recibió {text})."
    return True, int(value), None


def _check_range(value, low, high, name, unit=""):
    if not (low <= value <= high):
        return False, None, f"{name} debe estar en [{low:g}, {high:g}]{unit}. Valor recibido: {value:g}"
    return True, value, None


# ── Validadores de parámetros ────────────────────────────────────────────────

def validate_scheme(scheme):
    if scheme not in MODULATION_SCHEMES:
        return False, None, f"Esquema '{scheme}' no reconocido. Use: {MODULATION_SCHEMES}"
    return True, scheme, None


def validate_modulation_order(scheme: str, m):
    """Verifica que el orden M sea válido para el esquema (p. ej. M-QAM: 4, 16, 64, 256)."""
    ok, _, msg = validate_scheme(scheme)
    if not ok:
        return False, None, msg
    ok, m_int, msg = parse_int(m, "El orden M")
    if not ok:
        return False, None, msg
    valid = VALID_ORDERS[scheme]
    if m_int not in valid:
        return False, None, (
            f"Orden M={m_int} no válido para {scheme}. "
            f"Valores permitidos: {', '.join(str(v) for v in valid)}."
        )
    return True, m_int, None


def validate_roll_off(alpha):
    """El factor de roll-off del coseno alzado debe cumplir 0 ≤ α ≤ 1."""
    ok, value, msg = parse_float(alpha, "El factor roll-off")
    if not ok:
        return False, None, msg
    if not (ROLL_OFF_MIN <= value <= ROLL_OFF_MAX):
        return False, None, (
            f"Factor roll-off debe estar en [{ROLL_OFF_MIN:g}, {ROLL_OFF_MAX:g}]. "
            f"Valor recibido: {value:g}"
        )
    return True, value, None


def roll_off_advisory(alpha):
    """
    Aviso NO bloqueante para α válido pero muy pequeño.  Un filtro finito no puede
    ser Nyquist exacto con α→0 (la sinc decae como 1/t): queda algo de ISI y la BER
    simulada queda ligeramente sobre la teórica.  Retorna texto o None.
    """
    ok, value, _ = validate_roll_off(alpha)
    if ok and value < ROLL_OFF_ADVISORY:
        return (f"Aviso: con α < {ROLL_OFF_ADVISORY:g} el filtro finito deja ISI residual; "
                "la BER simulada puede quedar algo por encima de la teórica.")
    return None


def validate_num_symbols(n):
    ok, value, msg = parse_int(n, "El número de símbolos")
    if not ok:
        return False, None, msg
    return _check_range(value, NUM_SYMBOLS_MIN, NUM_SYMBOLS_MAX, "El número de símbolos")


def validate_sps(sps):
    ok, value, msg = parse_int(sps, "Las muestras por símbolo")
    if not ok:
        return False, None, msg
    return _check_range(value, SPS_MIN, SPS_MAX, "Las muestras por símbolo")


def validate_eb_n0(eb_n0_db):
    ok, value, msg = parse_float(eb_n0_db, "Eb/N0")
    if not ok:
        return False, None, msg
    return _check_range(value, EB_N0_LIMIT_MIN, EB_N0_LIMIT_MAX, "Eb/N0", " dB")


def validate_eb_n0_range(eb_n0_min, eb_n0_max):
    """Verifica que el rango Eb/N0 sea coherente (min < max, dentro de límites)."""
    ok, lo, msg = validate_eb_n0(eb_n0_min)
    if not ok:
        return False, None, "Eb/N0 mínimo: " + msg
    ok, hi, msg = validate_eb_n0(eb_n0_max)
    if not ok:
        return False, None, "Eb/N0 máximo: " + msg
    if lo >= hi:
        return False, None, "Eb/N0 mínimo debe ser menor que el máximo."
    return True, (lo, hi), None


def validate_eb_n0_steps(steps):
    ok, value, msg = parse_int(steps, "El número de puntos de la curva")
    if not ok:
        return False, None, msg
    return _check_range(value, EB_N0_STEPS_MIN, EB_N0_STEPS_MAX, "El número de puntos de la curva")


def validate_symbol_rate(rs):
    ok, value, msg = parse_float(rs, "La tasa de símbolo Rs")
    if not ok:
        return False, None, msg
    return _check_range(value, SYMBOL_RATE_MIN, SYMBOL_RATE_MAX, "La tasa de símbolo Rs", " baudios")


def validate_min_errors(n):
    ok, value, msg = parse_int(n, "Los errores mínimos")
    if not ok:
        return False, None, msg
    return _check_range(value, MIN_ERRORS_MIN, MIN_ERRORS_MAX, "Los errores mínimos")


def validate_max_symbols(n):
    ok, value, msg = parse_int(n, "El máximo de símbolos por punto")
    if not ok:
        return False, None, msg
    return _check_range(value, MAX_SYMBOLS_MIN, MAX_SYMBOLS_MAX, "El máximo de símbolos por punto")


def validate_simulation_params(scheme, m, alpha, n_symbols, eb_n0_db, eb_n0_min,
                               eb_n0_max, eb_n0_steps, symbol_rate_baud,
                               min_errors, max_symbols, sps):
    """
    Valida todos los parámetros de una simulación.

    Retorna (valores, errores):
        valores : dict con los datos convertidos (o None si hay errores)
        errores : dict {campo: mensaje}; vacío si todo es válido
    """
    errors: dict[str, str] = {}
    values: dict = {}

    ok, scheme_v, msg = validate_scheme(scheme)
    if not ok:
        errors["scheme"] = msg
    else:
        values["scheme"] = scheme_v
        ok, m_v, msg = validate_modulation_order(scheme_v, m)
        errors.update({} if ok else {"m": msg})
        values["m"] = m_v

    for key, fn, raw in (
        ("alpha", validate_roll_off, alpha),
        ("n_symbols", validate_num_symbols, n_symbols),
        ("eb_n0_db", validate_eb_n0, eb_n0_db),
        ("eb_n0_steps", validate_eb_n0_steps, eb_n0_steps),
        ("symbol_rate_baud", validate_symbol_rate, symbol_rate_baud),
        ("min_errors", validate_min_errors, min_errors),
        ("max_symbols", validate_max_symbols, max_symbols),
        ("sps", validate_sps, sps),
    ):
        ok, v, msg = fn(raw)
        if ok:
            values[key] = v
        else:
            errors[key] = msg

    ok, rng, msg = validate_eb_n0_range(eb_n0_min, eb_n0_max)
    if ok:
        values["eb_n0_min"], values["eb_n0_max"] = rng
    else:
        errors["eb_n0_range"] = msg

    return (None if errors else values), errors
