"""
constellation.py
================
RF3b — Diagrama de constelación en recepción.

Muestra los símbolos recibidos (después del filtro casado y antes del
decisor) en el plano IQ, junto con los puntos ideales de la constelación.

La dispersión de los puntos alrededor de los centros ideales es
proporcional al ruido del canal: a mayor Eb/N0, los puntos estarán
más agrupados.

Elementos del diagrama:
  - Puntos azules: símbolos recibidos (con ruido).
  - Cruces rojas:  puntos ideales de la constelación.
  - Círculo gris:  radio unitario (referencia para M-PSK).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_constellation(ax: Axes, received_symbols: np.ndarray,
                       ideal_constellation: np.ndarray,
                       title: str = "Constelación en Recepción") -> None:
    """
    Dibuja el diagrama de constelación sobre un Axes de matplotlib.

    Parámetros
    ----------
    ax                  : Axes    — eje donde graficar.
    received_symbols    : ndarray complejo — símbolos recibidos (post-MF).
    ideal_constellation : ndarray complejo — constelación ideal del TX.
    title               : str     — título de la gráfica.
    """
    ax.clear()

    # Limitar número de puntos para no saturar la visualización
    max_points = 1000
    rx = received_symbols[:max_points]

    # Puntos recibidos (con ruido)
    ax.scatter(np.real(rx), np.imag(rx),
               color="#1f77b4", alpha=0.4, s=8, label="Recibidos")

    # Puntos ideales de la constelación
    ax.scatter(np.real(ideal_constellation), np.imag(ideal_constellation),
               color="red", marker="x", s=80, linewidths=2,
               zorder=5, label="Constelación ideal")

    # Líneas de cuadrícula del plano IQ
    ax.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("I (en fase)", fontsize=9)
    ax.set_ylabel("Q (cuadratura)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
