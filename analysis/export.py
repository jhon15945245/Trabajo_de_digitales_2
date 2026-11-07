"""Exporta la tabla de entrega: CSV, Markdown, análisis de texto y gráficas PNG."""

from __future__ import annotations

import csv
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analysis.comparison import (
    ebn0_rows, generate_analysis, table_rows, to_markdown,
)
from plots.comparison import plot_ber_comparison, plot_delta_bars, plot_required_ebn0


def _save_fig(fig: Figure, path: str) -> None:
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def export_results(folder: str, theory: dict, sim: dict | None = None) -> list[str]:
    """Escribe todos los archivos en `folder`; retorna las rutas creadas."""
    os.makedirs(folder, exist_ok=True)
    paths = []

    rows = table_rows(theory, sim)
    csv_path = os.path.join(folder, "tabla_diferencias_ebn0.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    paths.append(csv_path)

    md_path = os.path.join(folder, "tabla_diferencias_ebn0.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Diferencia de Eb/N0 [dB] entre esquemas (BER objetivo)\n\n")
        fh.write(to_markdown(rows) + "\n\n")
        fh.write("## Eb/N0 requerido por esquema\n\n")
        fh.write(to_markdown(ebn0_rows(theory)) + "\n")
    paths.append(md_path)

    txt_path = os.path.join(folder, "analisis_resultados.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(generate_analysis(theory, sim) + "\n")
    paths.append(txt_path)

    for name, fn in (("comparacion_ber.png", lambda ax: plot_ber_comparison(ax, sim=sim)),
                     ("diferencias_ebn0.png", lambda ax: plot_delta_bars(ax, theory, sim)),
                     ("ebn0_requerido.png", lambda ax: plot_required_ebn0(ax, theory))):
        fig = Figure(figsize=(8, 5))
        fn(fig.add_subplot(111))
        path = os.path.join(folder, name)
        _save_fig(fig, path)
        paths.append(path)
    return paths
