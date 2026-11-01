"""
ber_curve.py
============
RF3c — Curva de desempeño BER vs Eb/N0.

Muestra dos curvas:
  1. BER teórica  — fórmula analítica (línea continua).
  2. BER simulada — medición punto a punto de la simulación (marcadores).

El eje Y es logarítmico (escala de décadas) para visualizar BER desde
10⁻¹ hasta 10⁻⁷.

Una buena simulación produce puntos que siguen de cerca la curva teórica.
Las discrepancias a BER muy bajas se deben al número limitado de símbolos
(pocos errores = estimación estadística poco confiable).
"""

import numpy as np
from matplotlib.axes import Axes


def plot_ber_curve(ax: Axes,
                   eb_n0_db: np.ndarray,
                   ber_theoretical: np.ndarray,
                   ber_simulated: np.ndarray | None = None,
                   scheme: str = "",
                   m: int = 0) -> None:
    """
    Dibuja la curva BER vs Eb/N0 en escala semilog-Y.

    Parámetros
    ----------
    ax              : Axes    — eje de matplotlib.
    eb_n0_db        : ndarray — valores Eb/N0 en dB.
    ber_theoretical : ndarray — BER teórica calculada analíticamente.
    ber_simulated   : ndarray | None — BER simulada (None = no mostrar).
    scheme          : str     — nombre del esquema (para la leyenda).
    m               : int     — orden de la modulación.
    """
    ax.clear()

    label_theory = f"{scheme} M={m} — Teórica" if scheme else "Teórica"
    label_sim    = f"{scheme} M={m} — Simulada" if scheme else "Simulada"

    # Curva teórica
    ax.semilogy(eb_n0_db, ber_theoretical,
                "b-", linewidth=2, label=label_theory)

    # Curva simulada (si se provee)
    if ber_simulated is not None:
        # Reemplazar ceros para no romper escala log
        ber_sim_plot = np.where(ber_simulated > 0, ber_simulated, np.nan)
        ax.semilogy(eb_n0_db, ber_sim_plot,
                    "ro--", markersize=5, linewidth=1.2, label=label_sim)

    ax.set_title("Curva de desempeño BER vs Eb/N0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Eb/N0 [dB]", fontsize=9)
    ax.set_ylabel("BER", fontsize=9)
    ax.set_ylim(1e-7, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def plot_ber_comparison(ax: Axes,
                        eb_n0_db: np.ndarray,
                        curves: list[dict]) -> None:
    """
    Dibuja múltiples curvas BER en un solo eje (para la tabla comparativa).

    Parámetros
    ----------
    ax       : Axes — eje de matplotlib.
    eb_n0_db : ndarray — eje x común.
    curves   : list de dicts con claves:
                 'ber'    : ndarray — BER a graficar.
                 'label'  : str     — leyenda.
                 'color'  : str     — color de la línea.
                 'linestyle': str   — estilo de la línea.
    """
    ax.clear()
    for c in curves:
        ber_plot = np.where(c["ber"] > 0, c["ber"], np.nan)
        ax.semilogy(eb_n0_db, ber_plot,
                    color=c.get("color", "blue"),
                    linestyle=c.get("linestyle", "-"),
                    linewidth=2,
                    label=c.get("label", ""))

    ax.set_title("Comparación BER — 4PAM vs 4QAM vs 4PSK", fontsize=11, fontweight="bold")
    ax.set_xlabel("Eb/N0 [dB]", fontsize=9)
    ax.set_ylabel("BER", fontsize=9)
    ax.set_ylim(1e-7, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
