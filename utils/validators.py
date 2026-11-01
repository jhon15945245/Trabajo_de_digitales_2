"""
validators.py
=============
Validaciones de entrada del usuario.
Retornan (True, None) si el valor es válido, o (False, mensaje_error) si no.
Esto permite que la GUI muestre mensajes sin crashear (RNF1).
"""

from utils.constants import VALID_ORDERS, ROLL_OFF_MIN, ROLL_OFF_MAX, MODULATION_SCHEMES


def validate_modulation_order(scheme: str, m: int) -> tuple[bool, str | None]:
    """
    Verifica que el orden M sea válido para el esquema seleccionado.
    Por ejemplo, M-QAM solo acepta potencias de 2 con raíz entera (4, 16, 64, 256).
    """
    if scheme not in MODULATION_SCHEMES:
        return False, f"Esquema '{scheme}' no reconocido. Use: {MODULATION_SCHEMES}"

    valid = VALID_ORDERS.get(scheme, [])
    if m not in valid:
        return False, (
            f"Orden M={m} no válido para {scheme}. "
            f"Valores permitidos: {valid}"
        )
    return True, None


def validate_roll_off(alpha: float) -> tuple[bool, str | None]:
    """
    Verifica que el factor roll-off esté en el rango [0, 1].
    Un valor > 1 es físicamente imposible y el profe lo menciona explícitamente.
    """
    if not (ROLL_OFF_MIN <= alpha <= ROLL_OFF_MAX):
        return False, (
            f"Factor roll-off debe estar en [{ROLL_OFF_MIN}, {ROLL_OFF_MAX}]. "
            f"Valor recibido: {alpha}"
        )
    return True, None


def validate_num_symbols(n: int) -> tuple[bool, str | None]:
    """Verifica que el número de símbolos sea positivo y razonable."""
    if n < 100:
        return False, "Se requieren al menos 100 símbolos para una simulación confiable."
    if n > 100_000:
        return False, "Máximo 100 000 símbolos para evitar tiempos de cómputo excesivos."
    return True, None


def validate_eb_n0_range(eb_n0_min: float, eb_n0_max: float) -> tuple[bool, str | None]:
    """Verifica que el rango Eb/N0 sea coherente."""
    if eb_n0_min >= eb_n0_max:
        return False, "Eb/N0 mínimo debe ser menor que el máximo."
    if eb_n0_min < -10 or eb_n0_max > 40:
        return False, "Rango Eb/N0 debe estar entre -10 dB y 40 dB."
    return True, None
