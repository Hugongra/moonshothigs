"""Approximate Median Significance (AMS) — competition scoring."""

from __future__ import annotations

import numpy as np


def ams(s: float, b: float, br: float = 10.0) -> float:
    """
    AMS = sqrt(2 * ((s+b+br)*ln(1 + s/(b+br)) - s))

    `s` and `b` are weighted sums of signal-like vs background-like selections.
    """
    if s <= 0:
        return 0.0
    rad = 2.0 * ((s + b + br) * np.log1p(s / (b + br)) - s)
    return float(np.sqrt(max(rad, 0.0)))


def best_threshold_ams(
    y_true: np.ndarray,
    proba_signal: np.ndarray,
    weights: np.ndarray,
    *,
    n_grid: int = 200,
    br: float = 10.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Scan probability thresholds; return (best_ams, best_threshold, thresholds, ams_curve).

    Selection: predict signal if proba_signal >= threshold (high score = signal-like).
    """
    y_true = np.asarray(y_true).astype(bool)
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(proba_signal, dtype=np.float64)

    lo, hi = float(p.min()), float(p.max())
    if hi <= lo:
        return 0.0, 0.5, np.array([0.5]), np.array([0.0])

    thresholds = np.linspace(lo + 1e-9, hi - 1e-9, n_grid)
    ams_vals = []
    for t in thresholds:
        sel = p >= t
        s = w[sel & y_true].sum()
        b = w[sel & ~y_true].sum()
        ams_vals.append(ams(s, b, br))
    ams_vals = np.array(ams_vals)
    i = int(np.argmax(ams_vals))
    return float(ams_vals[i]), float(thresholds[i]), thresholds, ams_vals
