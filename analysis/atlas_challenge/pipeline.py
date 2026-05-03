"""
End-to-end baseline: weighted HistGradientBoosting + AMS on validation split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from .data import SENTINEL, X_y_weights, load_training_table
from .metrics import best_threshold_ams
from .paths import DEFAULT_CSV, OUTPUT_DIR
from .plots import (
    save_ams_curve,
    save_importance,
    save_roc,
    save_signal_background_mass_peak,
)


def run_atlas_baseline(
    csv_path: Path | None = None,
    *,
    max_train_rows: int | None = 150_000,
    test_size: float = 0.25,
    random_state: int = 42,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train a baseline gradient-boosted tree model with **weighted** events.

    Default ``max_train_rows`` keeps RAM/runtime reasonable; increase or set ``None``
    for full training table (large RAM).
    """
    csv_path = csv_path or DEFAULT_CSV
    out = (output_dir or OUTPUT_DIR).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def _v(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t0 = time.perf_counter()

    _v("")
    _v("[atlas] ──────────── step 1/10 · load training table ────────────")
    _v(f"[atlas]   CSV path: {csv_path.resolve()}")
    _v(f"[atlas]   max_train_rows: {max_train_rows}")
    t_load = time.perf_counter()
    df = load_training_table(csv_path, max_rows=max_train_rows, random_state=random_state)
    _v(f"[atlas]   rows after KaggleSet=='t' (+ subsample): {len(df):,} ({time.perf_counter() - t_load:.2f}s)")
    _v(f"[atlas]   output directory: {out}/")

    # Physics comparison: weighted signal vs background in collinear mass (peak vs continuum)
    _v("")
    _v("[atlas] ──────────── step 2/10 · signal vs background (m_MMC) ────────────")
    mass_col = "DER_mass_MMC"
    if mass_col not in df.columns:
        _v(f"[atlas]   skipped — column {mass_col!r} not in CSV")
    else:
        raw_mass = df[mass_col].replace(SENTINEL, np.nan)
        is_s = df["Label"].values == "s"
        is_b = df["Label"].values == "b"
        ms = raw_mass[is_s].to_numpy(dtype=np.float64)
        ws = df.loc[is_s, "Weight"].to_numpy(dtype=np.float64)
        mb = raw_mass[is_b].to_numpy(dtype=np.float64)
        wb = df.loc[is_b, "Weight"].to_numpy(dtype=np.float64)
        ok_s = np.isfinite(ms)
        ok_b = np.isfinite(mb)
        n_s_raw, n_b_raw = int(is_s.sum()), int(is_b.sum())
        ms, ws = ms[ok_s], ws[ok_s]
        mb, wb = mb[ok_b], wb[ok_b]
        _v(
            f"[atlas]   labels: signal={n_s_raw:,} background={n_b_raw:,} | "
            f"finite {mass_col}: signal={ms.size:,} background={mb.size:,}"
        )
        if ms.size and mb.size:
            sb_path = out / "signal_vs_background_mass_mmc"
            t_sb = time.perf_counter()
            save_signal_background_mass_peak(ms, ws, mb, wb, sb_path)
            _v(f"[atlas]   wrote ({time.perf_counter() - t_sb:.2f}s):")
            _v(f"[atlas]     {sb_path.resolve()}.pdf")
            _v(f"[atlas]     {sb_path.resolve()}.png")
        else:
            _v("[atlas]   skipped — no finite masses for both classes")

    _v("")
    _v("[atlas] ──────────── step 3/10 · feature matrix & weights ────────────")
    X, y, w = X_y_weights(df)
    feature_names = list(X.columns)
    _v(f"[atlas]   features: {len(feature_names)} columns (DER_*/PRI_*; -999 → NaN)")
    _v(f"[atlas]   label: positive rate P(signal) = {y.mean():.4f}")

    _v("")
    _v("[atlas] ──────────── step 4/10 · train / validation split ────────────")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X.values,
        y,
        w,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    _v(
        f"[atlas]   stratified split test_size={test_size}: "
        f"train={len(y_train):,} | validation={len(y_val):,}"
    )

    _v("")
    _v("[atlas] ──────────── step 5/10 · fit HistGradientBoostingClassifier ────────────")
    _v("[atlas]   model: max_depth=6, lr=0.06, max_iter=200, early_stopping=True")
    t_fit = time.perf_counter()
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=200,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    clf.fit(X_train, y_train, sample_weight=w_train)
    _v(f"[atlas]   fit finished ({time.perf_counter() - t_fit:.2f}s)")

    _v("")
    _v("[atlas] ──────────── step 6/10 · validation scores & AMS scan ────────────")
    proba_val = clf.predict_proba(X_val)[:, 1]
    ll = log_loss(y_val, clf.predict_proba(X_val), sample_weight=w_val)
    acc = accuracy_score(y_val, clf.predict(X_val), sample_weight=w_val)

    best_ams, best_t, thr_grid, ams_curve = best_threshold_ams(
        y_val.astype(bool), proba_val, w_val
    )

    _v(f"[atlas]   weighted log-loss (validation): {ll:.5f}")
    _v(f"[atlas]   weighted accuracy (validation): {acc:.5f}")
    _v(f"[atlas]   AMS @ scanned threshold: {best_ams:.5f} (best t ≈ {best_t:.5f})")

    _v("")
    _v("[atlas] ──────────── step 7/10 · ROC curve (validation) ────────────")
    roc_base = out / "roc_validation"
    t_plot = time.perf_counter()
    save_roc(y_val, proba_val, w_val, roc_base, "Weighted ROC — validation fold")
    _v(f"[atlas]   wrote ({time.perf_counter() - t_plot:.2f}s):")
    _v(f"[atlas]     {roc_base.resolve()}.pdf")

    _v("")
    _v("[atlas] ──────────── step 8/10 · AMS vs probability threshold ────────────")
    t_ams = time.perf_counter()
    ams_base = out / "ams_vs_threshold"
    save_ams_curve(thr_grid, ams_curve, best_t, ams_base)
    _v(f"[atlas]   wrote ({time.perf_counter() - t_ams:.2f}s):")
    _v(f"[atlas]     {ams_base.resolve()}.pdf")

    _v("")
    _v("[atlas] ──────────── step 9/10 · feature importances ────────────")
    imp = getattr(clf, "feature_importances_", None)
    if imp is not None and len(imp) == len(feature_names):
        imp_base = out / "feature_importance_top"
        t_imp = time.perf_counter()
        save_importance(feature_names, np.asarray(imp), imp_base)
        _v(f"[atlas]   wrote ({time.perf_counter() - t_imp:.2f}s):")
        _v(f"[atlas]     {imp_base.resolve()}.pdf")
    else:
        _v(
            "[atlas]   skipped — model has no feature_importances_ matching "
            f"n_features={len(feature_names)} (sklearn build/version)"
        )

    _v("")
    _v("[atlas] ──────────── step 10/10 · write metrics.json ────────────")
    metrics = {
        "csv": str(csv_path.resolve()),
        "n_train_rows_used": len(df),
        "validation_log_loss": float(ll),
        "validation_weighted_accuracy": float(acc),
        "validation_AMS": float(best_ams),
        "best_probability_threshold": float(best_t),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "model": "HistGradientBoostingClassifier",
        "note": "AMS computed on validation fold only; not leaderboard test set.",
    }
    mj = out / "metrics.json"
    mj.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _v(f"[atlas]   {mj.resolve()}")
    _v("")
    _v(f"[atlas] ========== done · wall time {metrics['elapsed_seconds']:.1f}s ==========")

    return metrics
