"""
eye_diagram.py
==============
RF3a — Diagrama del ojo en recepción.

El diagrama del ojo se construye superponiendo múltiples segmentos
de la señal recibida (después del filtro casado, antes del decisor)
de longitud igual a 2 períodos de símbolo.

Un ojo bien abierto indica:
  - Poca ISI (Intersymbol Interference).
  - Buena relación señal-ruido.
  - El instante óptimo de muestreo es en el centro del ojo.

Parámetros de interés visibles:
  - Apertura vertical: margen de ruido antes del error.
  - Apertura horizontal: sensibilidad al jitter de sincronismo.
  - Cruce cero: instante de transición entre símbolos.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_eye_diagram(ax: Axes, signal: np.ndarray, sps: int,
                     n_traces: int = 100, title: str = "Diagrama del Ojo") -> None:
    """
    Dibuja el diagrama del ojo sobre un Axes de matplotlib.

    Proceso:
      1. Selecciona n_traces segmentos de 2·sps muestras.
      2. Superpone todos los segmentos en el mismo eje.
      3. Muestra la componente real (señal I después del filtro casado).

    Parámetros
    ----------
    ax       : Axes  — eje de matplotlib donde graficar.
    signal   : ndarray — señal en banda base después del filtro casado
                         (componente real).
    sps      : int   — muestras por símbolo.
    n_traces : int   — número de segmentos a superponer.
    title    : str   — título de la gráfica.
    """
    ax.clear()
    segment_len = 2 * sps     # ventana de 2 períodos de símbolo

    # Eje de tiempo normalizado en [0, 2T]
    t_norm = np.linspace(0, 2, segment_len)

    # Componente real de la señal recibida
    signal_real = np.real(signal)

    # Número máximo de segmentos disponibles
    n_available = (len(signal_real) - segment_len) // sps
    n_traces = min(n_traces, n_available)

    for i in range(n_traces):
        start = i * sps
        segment = signal_real[start: start + segment_len]
        ax.plot(t_norm, segment, color="#1f77b4", alpha=0.15, linewidth=0.8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Tiempo normalizado [T]", fontsize=9)
    ax.set_ylabel("Amplitud", fontsize=9)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.2,
               label="Instante de muestreo")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
