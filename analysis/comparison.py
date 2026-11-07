"""Tabla de entrega del PDF: ΔEb/N0 teórica y simulada entre 4PAM, 4QAM y 4PSK para BER de 1e-2 a 1e-7."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from core.ber_calculator import BERCalculator
from core.simulation_runner import SimulationRunner
from utils.constants import (
    BER_TARGETS, COMPARISON_SCHEMES, MIN_RELIABLE_ERRORS, RANDOM_SEED,
)

PAIRS = (("4PAM", "4QAM"), ("4PSK", "4QAM"), ("4QAM", "4PAM"))


def pair_label(a: str, b: str) -> str:
    return f"{a}-{b}"


def ebn0_for_ber(scheme: str, m: int, target_ber: float,
                 low_db: float = -10.0, high_db: float = 40.0) -> float:
    """Eb/N0 [dB] tal que BER_teórica = target (bisección; BER decrece con Eb/N0)."""
    def ber(x):
        return float(BERCalculator.ber_theoretical(scheme, m, x))

    if not (ber(high_db) <= target_ber <= ber(low_db)):
        raise ValueError(f"BER objetivo {target_ber:g} fuera del rango [{low_db}, {high_db}] dB")
    lo, hi = low_db, high_db
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ber(mid) > target_ber:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def theoretical_table(targets=BER_TARGETS) -> dict:
    """Diferencia de Eb/N0 teórica entre cada par de esquemas para los BER objetivo."""
    ebn0 = {name: [ebn0_for_ber(sch, m, t) for t in targets]
            for name, sch, m in COMPARISON_SCHEMES}
    diff = {pair_label(a, b): [x - y for x, y in zip(ebn0[a], ebn0[b])]
            for a, b in PAIRS}
    return {"targets": list(targets), "ebn0": ebn0, "diff": diff}


def interpolate_ebn0(eb_n0_db: np.ndarray, ber: np.ndarray, errors: np.ndarray,
                     target: float, min_errors: int = MIN_RELIABLE_ERRORS) -> float:
    """Eb/N0 [dB] donde la BER simulada cruza `target` (interpolación lineal en log10(BER))."""
    x = np.asarray(eb_n0_db, float)
    y = np.asarray(ber, float)
    ok = (np.asarray(errors) >= min_errors) & (y > 0)
    x, y = x[ok], y[ok]
    for i in range(len(x) - 1):
        if y[i] >= target >= y[i + 1]:
            ly0, ly1, lt = np.log10(y[i]), np.log10(y[i + 1]), np.log10(target)
            if ly0 == ly1:
                return float(x[i])
            return float(x[i] + (lt - ly0) / (ly1 - ly0) * (x[i + 1] - x[i]))
    return float("nan")


def simulate_comparison(alpha: float = 0.35, max_symbols: int = 400_000,
                        min_errors: int = 200, eb_n0_points: Optional[np.ndarray] = None,
                        seed: int = RANDOM_SEED,
                        progress: Optional[Callable[[int, int], None]] = None,
                        cancelled: Optional[Callable[[], bool]] = None) -> Optional[dict]:
    """Simula la curva BER de 4PAM, 4QAM y 4PSK y calcula el Eb/N0 simulado en cada BER objetivo."""
    if eb_n0_points is None:
        eb_n0_points = np.arange(0.0, 15.0, 1.0)
    curves = {}
    total = len(COMPARISON_SCHEMES) * len(eb_n0_points)
    done = 0
    for name, sch, m in COMPARISON_SCHEMES:
        def _p(i, n, base=done):
            if progress:
                progress(base + i, total)
        ber, err, bits = SimulationRunner.simulate_ber_curve(
            sch, m, alpha, eb_n0_points, seed=seed, min_errors=min_errors,
            max_symbols=max_symbols, progress=_p, cancelled=cancelled)
        if ber is None:
            return None
        curves[name] = {"ber": ber, "errors": err, "bits": bits}
        done += len(eb_n0_points)

    sim_ebn0 = {name: [interpolate_ebn0(eb_n0_points, c["ber"], c["errors"], t)
                       for t in BER_TARGETS] for name, c in curves.items()}
    sim_diff = {pair_label(a, b): [x - y for x, y in zip(sim_ebn0[a], sim_ebn0[b])]
                for a, b in PAIRS}
    return {"eb_n0": np.asarray(eb_n0_points), "curves": curves,
            "ebn0": sim_ebn0, "diff": sim_diff, "alpha": alpha}


def format_target(t: float) -> str:
    return f"1e-{int(round(-np.log10(t)))}"


def table_rows(theory: dict, sim: Optional[dict] = None) -> list[list[str]]:
    """Filas de texto de la tabla (encabezado incluido) para GUI/CSV/Markdown."""
    header = ["BER"] + [pair_label(a, b) + " [dB]" for a, b in PAIRS]
    if sim is not None:
        header += [pair_label(a, b) + " sim. [dB]" for a, b in PAIRS[:2]]
    rows = [header]
    for i, t in enumerate(theory["targets"]):
        row = [format_target(t)]
        row += [f"{theory['diff'][pair_label(a, b)][i]:+.3f}" for a, b in PAIRS]
        if sim is not None:
            for a, b in PAIRS[:2]:
                v = sim["diff"][pair_label(a, b)][i]
                row.append("n/m" if np.isnan(v) else f"{v:+.3f}")
        rows.append(row)
    return rows


def ebn0_rows(theory: dict) -> list[list[str]]:
    """Tabla auxiliar: Eb/N0 [dB] necesario por esquema y su equivalente lineal."""
    header = ["BER"] + [f"{n} [dB]" for n, _, _ in COMPARISON_SCHEMES] \
        + [f"{n} [lineal]" for n, _, _ in COMPARISON_SCHEMES]
    rows = [header]
    for i, t in enumerate(theory["targets"]):
        db = [theory["ebn0"][n][i] for n, _, _ in COMPARISON_SCHEMES]
        rows.append([format_target(t)] + [f"{v:.3f}" for v in db]
                    + [f"{10 ** (v / 10):.3f}" for v in db])
    return rows


def to_markdown(rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def generate_analysis(theory: dict, sim: Optional[dict] = None) -> str:
    """Análisis en español con los números reales calculados."""
    d1 = theory["diff"]["4PAM-4QAM"]
    d2 = theory["diff"]["4PSK-4QAM"]
    asym_db = 10 * np.log10(2.5)
    lin = [10 ** (v / 10) for v in d1]
    lines = [
        "ANÁLISIS DE RESULTADOS — 4PAM vs 4QAM vs 4PSK (canal AWGN, filtro SRRC)",
        "",
        "1) 4PSK frente a 4QAM: la diferencia es "
        f"{max(abs(v) for v in d2):.3f} dB en todos los BER. Con M=4 ambas son la misma "
        "constelación (cuatro puntos equidistantes y de igual energía) girada 45°. "
        "Tienen la misma distancia mínima d_min = √(2·Es), el mismo mapeo Gray y, por tanto, "
        "la misma Pb = Q(√(2·Eb/N0)). Las curvas se superponen.",
        "",
        "2) 4PAM frente a 4QAM: 4PAM necesita entre "
        f"{d1[0]:.2f} dB (BER=1e-2) y {d1[-1]:.2f} dB (BER=1e-7) más de Eb/N0. "
        f"En escala lineal eso es un factor de {lin[0]:.2f} a {lin[-1]:.2f} en energía por bit. "
        "La razón es que 4PAM usa solo una dimensión (eje I): para la misma Es los 4 niveles "
        "{±1, ±3}/√5 quedan más juntos (d_min = 2/√5·√Es ≈ 0.894·√Es) que en 4QAM "
        "(d_min = √2·√Es ≈ 1.414·√Es). El argumento de Q es √(0.8·Eb/N0) en 4PAM frente a "
        "√(2·Eb/N0) en 4QAM: razón de argumentos² = 2.5 → 10·log10(2.5) = "
        f"{asym_db:.2f} dB.",
        "",
        f"3) Tendencia: la diferencia crece con la exigencia de BER ({d1[0]:.3f} → {d1[-1]:.3f} dB) "
        f"y tiende asintóticamente a {asym_db:.2f} dB. A BER altos el coeficiente "
        "delante de Q (0.75 en 4PAM frente a 1 en 4QAM) todavía influye; a BER bajos domina "
        "la exponencial del argumento de Q.",
        "",
        "4) La columna 4QAM-4PAM es la anterior con el signo cambiado (solo indica quién "
        "necesita más energía): 4QAM necesita menos Eb/N0 que 4PAM.",
        "",
        "5) Compromiso: a igual Rs y roll-off, 4PAM, 4QAM y 4PSK modulados en pasabanda ocupan "
        "el mismo ancho de banda (1+α)·Rs, pero 4PAM transmite log2(4)=2 bits/símbolo con peor "
        "protección ante el ruido que 4QAM/4PSK; 4PSK además mantiene envolvente constante.",
    ]
    if sim is not None:
        lines += ["", "6) Verificación por simulación (Monte Carlo, sistema completo con filtro "
                      "SRRC, modulación y demodulación IQ): "]
        for i, t in enumerate(theory["targets"]):
            v = sim["diff"]["4PAM-4QAM"][i]
            if not np.isnan(v):
                lines.append(f"   BER {format_target(t)}: 4PAM-4QAM simulado = {v:+.3f} dB "
                             f"(teórico {d1[i]:+.3f} dB, error {v - d1[i]:+.3f} dB).")
        lines.append("   Incertidumbre estadística: con N errores por punto la BER medida tiene error "
                     "relativo ≈ 1/√N (≈ 11 % con ~80 errores, ≈ 6 % con ~300). Como la curva baja "
                     "~1 década cada 1.6 dB cerca de 1e-4, eso equivale a ≈ ±0.07 dB por curva y "
                     "≈ ±0.1 dB en la diferencia entre dos esquemas: las desviaciones simulado-teórico "
                     "de ese orden NO indican un error del sistema. Para reducirlas hay que simular "
                     "más símbolos (--max-simbolos), no afinar la rejilla de Eb/N0.")
        lines.append("   Los BER marcados 'n/m' no alcanzan el mínimo de errores fiables con el tamaño "
                     "Monte Carlo elegido; BER = 1e-6 y 1e-7 exigirían ≈ 1e8–1e9 bits, por lo que "
                     "allí se usa la expresión teórica (validada donde sí es medible).")
    return "\n".join(lines)
