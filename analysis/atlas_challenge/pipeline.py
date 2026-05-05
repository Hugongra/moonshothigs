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
from sklearn.model_selection import StratifiedKFold

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
    cv_folds: int = 5,
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
    _v("[atlas] ──────────── step 4/10 · stratified K-fold CV (weighted) ────────────")
    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=random_state)
    X_np = X.values
    # ``X_y_weights`` returns ``y`` and ``w`` as NumPy arrays (not pandas Series).
    y_np = np.asarray(y, dtype=bool)
    w_np = np.asarray(w, dtype=np.float64)

    fold_ll: list[float] = []
    fold_acc: list[float] = []
    fold_ams: list[float] = []
    fold_thr: list[float] = []

    clf_template = dict(
        max_depth=6,
        learning_rate=0.06,
        max_iter=200,
        random_state=random_state,
        # K-fold provides an explicit validation fold; disable internal early stopping split.
        early_stopping=False,
    )

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_np, y_np), start=1):
        _v(f"[atlas]   fold {fold_idx}/{cv_folds}: train={tr_idx.size:,} | val={va_idx.size:,}")
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]
        w_tr, w_va = w_np[tr_idx], w_np[va_idx]

        t_fit = time.perf_counter()
        clf = HistGradientBoostingClassifier(**clf_template)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        _v(f"[atlas]     fit finished ({time.perf_counter() - t_fit:.2f}s)")

        proba_va = clf.predict_proba(X_va)[:, 1]
        ll = float(log_loss(y_va, clf.predict_proba(X_va), sample_weight=w_va))
        acc = float(accuracy_score(y_va, clf.predict(X_va), sample_weight=w_va))
        best_ams, best_t, thr_grid, ams_curve = best_threshold_ams(y_va, proba_va, w_va)

        fold_ll.append(ll)
        fold_acc.append(acc)
        fold_ams.append(float(best_ams))
        fold_thr.append(float(best_t))

        _v(f"[atlas]     weighted log-loss (val): {ll:.5f}")
        _v(f"[atlas]     weighted accuracy (val): {acc:.5f}")
        _v(f"[atlas]     AMS @ scanned threshold: {best_ams:.5f} (best t ≈ {best_t:.5f})")

    ll_mean, ll_std = float(np.mean(fold_ll)), float(np.std(fold_ll, ddof=1)) if len(fold_ll) > 1 else 0.0
    acc_mean, acc_std = float(np.mean(fold_acc)), float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0
    ams_mean, ams_std = float(np.mean(fold_ams)), float(np.std(fold_ams, ddof=1)) if len(fold_ams) > 1 else 0.0
    thr_mean, thr_std = float(np.mean(fold_thr)), float(np.std(fold_thr, ddof=1)) if len(fold_thr) > 1 else 0.0

    _v("")
    _v("[atlas] ──────────── step 5/10 · refit on full training table for plots ────────────")
    _v("[atlas]   model: max_depth=6, lr=0.06, max_iter=200, early_stopping=False (full-data refit)")
    t_full = time.perf_counter()
    clf_full = HistGradientBoostingClassifier(**clf_template)
    clf_full.fit(X_np, y_np, sample_weight=w_np)
    _v(f"[atlas]   full fit finished ({time.perf_counter() - t_full:.2f}s)")

    _v("")
    _v("[atlas] ──────────── step 6/10 · full-data scores (for ROC/AMS curves; not CV) ────────────")
    proba_all = clf_full.predict_proba(X_np)[:, 1]
    # NOTE: These are not a proper hold-out estimate; they are diagnostic plots on the training sample.
    best_ams_full, best_t_full, thr_grid_full, ams_curve_full = best_threshold_ams(y_np, proba_all, w_np)

    _v("")
    _v("[atlas] ──────────── step 7/10 · ROC curve (full-data scores) ────────────")
    roc_base = out / "roc_validation"
    t_plot = time.perf_counter()
    save_roc(y_np, proba_all, w_np, roc_base, "Weighted ROC — training sample scores (full refit)")
    _v(f"[atlas]   wrote ({time.perf_counter() - t_plot:.2f}s):")
    _v(f"[atlas]     {roc_base.resolve()}.pdf")

    _v("")
    _v("[atlas] ──────────── step 8/10 · AMS vs probability threshold (full-data scores) ────────────")
    t_ams = time.perf_counter()
    ams_base = out / "ams_vs_threshold"
    save_ams_curve(thr_grid_full, ams_curve_full, best_t_full, ams_base)
    _v(f"[atlas]   wrote ({time.perf_counter() - t_ams:.2f}s):")
    _v(f"[atlas]     {ams_base.resolve()}.pdf")

    _v("")
    _v("[atlas] ──────────── step 9/10 · feature importances ────────────")
    imp = getattr(clf_full, "feature_importances_", None)
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
        # Back-compat keys: now represent K-fold *validation* means (not a single holdout split).
        "validation_log_loss": float(ll_mean),
        "validation_weighted_accuracy": float(acc_mean),
        "validation_AMS": float(ams_mean),
        "best_probability_threshold": float(thr_mean),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "model": "HistGradientBoostingClassifier",
        "note": (
            "Metrics are stratified K-fold (K=5) out-of-fold estimates on the training table after "
            "KaggleSet=='t' filtering; AMS is the maximum over a scanned probability threshold within each fold. "
            "ROC/AMS curves are diagnostic plots from a full-data refit (not a hold-out)."
        ),
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits": int(cv_folds),
            "shuffle": True,
            "random_state": int(random_state),
            "legacy_holdout_test_size": float(test_size),
        },
        "folds": [
            {
                "fold": i + 1,
                "validation_log_loss": fold_ll[i],
                "validation_weighted_accuracy": fold_acc[i],
                "validation_AMS": fold_ams[i],
                "best_probability_threshold": fold_thr[i],
            }
            for i in range(len(fold_ll))
        ],
        "validation_log_loss_mean": float(ll_mean),
        "validation_log_loss_std": float(ll_std),
        "validation_weighted_accuracy_mean": float(acc_mean),
        "validation_weighted_accuracy_std": float(acc_std),
        "validation_AMS_mean": float(ams_mean),
        "validation_AMS_std": float(ams_std),
        "best_probability_threshold_mean": float(thr_mean),
        "best_probability_threshold_std": float(thr_std),
        "full_refit": {
            "AMS_train_sample_max": float(best_ams_full),
            "best_probability_threshold_train_sample": float(best_t_full),
        },
    }
    mj = out / "metrics.json"
    mj.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _v(f"[atlas]   {mj.resolve()}")
    _v("")
    _v(f"[atlas] ========== done · wall time {metrics['elapsed_seconds']:.1f}s ==========")

    return metrics
