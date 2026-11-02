"""
ber_curve.py
============
RF3c — Curva de desempeño BER vs Eb/N0.

  • Línea azul continua : BER teórica (fórmula analítica).
  • Círculos rojos      : BER simulada (Monte Carlo, mismo sistema completo).
  • Triángulos huecos ▼ : puntos SIN errores; la BER real es menor que
                          1/(bits simulados) (cota superior, no BER = 0).
  • Estrella verde      : punto de operación de las demás gráficas.

La simulada debe seguir a la teórica (salvo el ruido estadístico cuando hay
pocos errores).
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def plot_ber_curve(ax: Axes, eb_n0_db: np.ndarray, ber_theoretical: np.ndarray,
                   ber_simulated: np.ndarray | None = None,
                   scheme: str = "", m: int = 0,
                   errors: np.ndarray | None = None,
                   bits: np.ndarray | None = None,
                   operating_point: tuple[float, float] | None = None) -> None:
    ax.clear()
    name = f"{scheme} M={m}" if scheme else ""
    ax.semilogy(eb_n0_db, ber_theoretical, "b-", linewidth=2,
                label=f"{name} — Teórica".strip(" —"))

    if ber_simulated is not None and len(ber_simulated):
        ber_simulated = np.asarray(ber_simulated, float)
        has_err = np.isfinite(ber_simulated) & (ber_simulated > 0)
        if np.any(has_err):
            ax.semilogy(np.asarray(eb_n0_db)[has_err], ber_simulated[has_err], "ro--",
                        markersize=5, linewidth=1.0, label=f"{name} — Simulada".strip(" —"))
        if errors is not None and bits is not None:
            zero = (np.asarray(errors) == 0) & (np.asarray(bits) > 0)
            if np.any(zero):
                ax.semilogy(np.asarray(eb_n0_db)[zero], 1.0 / np.asarray(bits)[zero], "v",
                            mfc="none", mec="red", markersize=7, label="Sin errores (BER < 1/bits)")

    if operating_point is not None and operating_point[1] > 0:
        ax.semilogy([operating_point[0]], [operating_point[1]], "g*", markersize=12,
                    label="Punto de operación")

    ax.set_title("Curva de desempeño BER vs Eb/N0", fontsize=11, fontweight="bold")
    ax.set_xlabel("Eb/N0 [dB]", fontsize=9)
    ax.set_ylabel("BER", fontsize=9)
    ax.set_ylim(1e-7, 1.0)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
