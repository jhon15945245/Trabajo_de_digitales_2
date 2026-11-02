"""
comparison.py (plots)
=====================
Gráficas de la tabla de entrega: curvas BER comparadas y diferencias de Eb/N0.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from analysis.comparison import PAIRS, format_target, pair_label
from core.ber_calculator import BERCalculator
from utils.constants import COMPARISON_SCHEMES

_COLORS = {"4PAM": "#d62728", "4QAM": "#1f77b4", "4PSK": "#2ca02c"}
_STYLES = {"4PAM": "-", "4QAM": "-", "4PSK": "--"}


def plot_ber_comparison(ax: Axes, eb_n0_db: np.ndarray | None = None,
                        sim: dict | None = None) -> None:
    """Curvas BER teóricas de 4PAM, 4QAM y 4PSK (+ puntos simulados si se dan)."""
    ax.clear()
    x = np.linspace(0, 16, 321) if eb_n0_db is None else np.asarray(eb_n0_db)
    for name, sch, m in COMPARISON_SCHEMES:
        ax.semilogy(x, BERCalculator.ber_theoretical(sch, m, x), _STYLES[name],
                    color=_COLORS[name], linewidth=2, label=f"{name} (teórica)")
        if sim is not None:
            c = sim["curves"][name]
            ok = (np.asarray(c["errors"]) >= 30) & (np.asarray(c["ber"]) > 0)
            ax.semilogy(sim["eb_n0"][ok], np.asarray(c["ber"])[ok], "o", color=_COLORS[name],
                        markersize=5, mfc="none", label=f"{name} (simulada)")
    ax.set_title("Comparación BER — 4PAM vs 4QAM vs 4PSK", fontsize=11, fontweight="bold")
    ax.set_xlabel("Eb/N0 [dB]", fontsize=9)
    ax.set_ylabel("BER", fontsize=9)
    ax.set_ylim(1e-7, 1.0)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def plot_delta_bars(ax: Axes, theory: dict, sim: dict | None = None) -> None:
    """Barras de ΔEb/N0 [dB] por BER objetivo para cada par."""
    ax.clear()
    targets = theory["targets"]
    idx = np.arange(len(targets))
    width = 0.8 / len(PAIRS)
    colors = ["#d62728", "#2ca02c", "#9467bd"]
    for j, (a, b) in enumerate(PAIRS):
        vals = theory["diff"][pair_label(a, b)]
        bars = ax.bar(idx + (j - 1) * width, vals, width, color=colors[j],
                      label=f"{pair_label(a, b)} (teórico)")
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:+.2f}", (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=6)
    if sim is not None:
        for j, (a, b) in enumerate(PAIRS[:2]):
            vals = np.array(sim["diff"][pair_label(a, b)], float)
            ok = ~np.isnan(vals)
            ax.plot((idx + (j - 1) * width)[ok], vals[ok], "k^", markersize=6,
                    label="simulado" if j == 0 else None)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(10 * np.log10(2.5), color="gray", linestyle=":", linewidth=1,
               label="límite 10·log10(2.5) = 3.98 dB")
    ax.set_xticks(idx)
    ax.set_xticklabels([format_target(t) for t in targets])
    ax.set_xlabel("BER objetivo", fontsize=9)
    ax.set_ylabel("Diferencia de Eb/N0 [dB]", fontsize=9)
    ax.set_title("Diferencia de Eb/N0 entre esquemas", fontsize=11, fontweight="bold")
    ax.set_ylim(-4.8, 6.6)                        # espacio libre arriba para la leyenda
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)


def plot_required_ebn0(ax: Axes, theory: dict) -> None:
    """Eb/N0 [dB] requerido por esquema en función de la BER objetivo."""
    ax.clear()
    t = np.array(theory["targets"])
    for name, _, _ in COMPARISON_SCHEMES:
        ax.semilogx(t, theory["ebn0"][name], "o" + _STYLES[name], color=_COLORS[name], label=name)
    ax.invert_xaxis()
    ax.set_xlabel("BER objetivo", fontsize=9)
    ax.set_ylabel("Eb/N0 requerido [dB]", fontsize=9)
    ax.set_title("Eb/N0 necesario para cada BER", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
