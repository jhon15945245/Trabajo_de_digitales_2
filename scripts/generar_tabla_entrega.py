"""Genera por línea de comandos la tabla de entrega, el análisis y las gráficas."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")

from analysis.comparison import (
    ebn0_rows, generate_analysis, simulate_comparison, table_rows,
    theoretical_table, to_markdown,
)
from analysis.export import export_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simular", action="store_true", help="agrega verificación Monte Carlo")
    parser.add_argument("--max-simbolos", type=int, default=400_000,
                        help="símbolos máximos por punto Eb/N0 (default 400000)")
    parser.add_argument("--alpha", type=float, default=0.35, help="roll-off (default 0.35)")
    parser.add_argument("--salida", default="resultados", help="carpeta de salida")
    args = parser.parse_args()

    theory = theoretical_table()
    sim = None
    if args.simular:
        print("Simulando (puede tardar)…")
        sim = simulate_comparison(alpha=args.alpha, max_symbols=args.max_simbolos,
                                  progress=lambda i, n: print(f"  {i}/{n}", end="\r"))
        print()

    print(to_markdown(table_rows(theory, sim)))
    print()
    print(to_markdown(ebn0_rows(theory)))
    print()
    print(generate_analysis(theory, sim))
    paths = export_results(args.salida, theory, sim)
    print("\nArchivos generados:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
