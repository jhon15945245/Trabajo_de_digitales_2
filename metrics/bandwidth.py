"""Métrica RF4c — Ancho de banda."""

from __future__ import annotations


def compute_bandwidth_normalized(alpha: float) -> float:
    """B/Rs = 1+α."""
    return 1.0 + alpha


def compute_bandwidth_hz(alpha: float, symbol_rate_baud: float) -> float:
    """B [Hz] = Rs·(1+α)."""
    return symbol_rate_baud * (1.0 + alpha)
