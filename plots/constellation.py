"""RF3b — Diagrama de constelación en recepción."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def plot_constellation(ax: Axes, received_symbols: np.ndarray,
                       ideal_constellation: np.ndarray,
                       title: str = "Constelación en Recepción",
                       evm_rms: float | None = None) -> None:
    ax.clear()
    rx = received_symbols[:2000]
    ax.scatter(np.real(rx), np.imag(rx), color="#1f77b4", alpha=0.4, s=8,
               label="Recibidos (post filtro casado)")
    ax.scatter(np.real(ideal_constellation), np.imag(ideal_constellation),
               color="red", marker="x", s=80, linewidths=2, zorder=5,
               label="Constelación ideal")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)

    suffix = f"  (EVM rms = {evm_rms:.3f})" if evm_rms is not None else ""
    ax.set_title(title + suffix, fontsize=11, fontweight="bold")
    ax.set_xlabel("I (en fase)", fontsize=9)
    ax.set_ylabel("Q (cuadratura)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
