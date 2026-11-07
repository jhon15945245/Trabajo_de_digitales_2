"""RF3a — Diagrama del ojo en recepción."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def eye_time_axis(sps: int) -> np.ndarray:
    """Eje de tiempo en unidades de T: -1 … +1 con paso exacto 1/sps (2·sps+1 puntos)."""
    return (np.arange(2 * sps + 1) - sps) / sps


def eye_segments(signal: np.ndarray, sps: int, n_traces: int = 100) -> np.ndarray:
    """Construye las trazas del ojo: matriz (n_traces, 2·sps+1)."""
    n_symbols = len(signal) // sps
    n_traces = max(0, min(n_traces, n_symbols - 2))
    return np.array([signal[(j - 1) * sps: (j + 1) * sps + 1] for j in range(1, n_traces + 1)])


def plot_eye_diagram(ax: Axes, signal: np.ndarray, sps: int, n_traces: int = 100,
                     title: str = "Diagrama del Ojo", show_q: bool | None = None) -> None:
    """Dibuja el diagrama del ojo de la salida del filtro casado (rama I y, si aplica, Q)."""
    ax.clear()
    t = eye_time_axis(sps)
    segs_i = eye_segments(np.real(signal), sps, n_traces)
    for seg in segs_i:
        ax.plot(t, seg, color="#1f77b4", alpha=0.18, linewidth=0.8)

    handles = [Line2D([0], [0], color="#1f77b4", label="I")]
    if show_q is None:
        show_q = bool(np.max(np.abs(np.imag(signal))) > 1e-9
                      and np.std(np.imag(signal)) > 0.05 * np.std(np.real(signal)))
    if show_q:
        for seg in eye_segments(np.imag(signal), sps, n_traces):
            ax.plot(t, seg, color="#ff7f0e", alpha=0.18, linewidth=0.8)
        handles.append(Line2D([0], [0], color="#ff7f0e", label="Q"))

    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
    handles.append(Line2D([0], [0], color="red", linestyle="--", label="Instante de muestreo"))
    ax.set_title(title + " (salida del filtro casado)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Tiempo [T]  (T = período de símbolo)", fontsize=9)
    ax.set_ylabel("Amplitud [normalizada, Es = 1]", fontsize=9)
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
