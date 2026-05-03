from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def save_roc(y_true, proba, weights, outfile: Path, title: str) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    # weighted ROC via repeating weights approx: use sample weights in roc_curve if sklearn supports
    fpr, tpr, _ = roc_curve(y_true, proba, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#c44e52", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), dpi=150)
    plt.close(fig)
    return outfile.with_suffix(".pdf")


def save_ams_curve(thresholds: np.ndarray, ams_vals: np.ndarray, best_t: float, outfile: Path) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, ams_vals, color="#2ca02c", lw=1.5)
    ax.axvline(best_t, color="crimson", linestyle="--", label=f"max AMS @ t={best_t:.4f}")
    ax.set_xlabel(r"Cut on $\hat P(s)$")
    ax.set_ylabel("AMS")
    ax.set_title("Approximate Median Significance vs. probability threshold")
    ax.legend()
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), dpi=150)
    plt.close(fig)
    return outfile.with_suffix(".pdf")


def save_signal_background_mass_peak(
    mass_signal: np.ndarray,
    weight_signal: np.ndarray,
    mass_background: np.ndarray,
    weight_background: np.ndarray,
    outfile: Path,
    *,
    xlabel: str = r"$m_{\mathrm{MMC}}$ (GeV)",
    title: str = "Signal vs background — collinear mass (weighted density)",
) -> Path:
    """
    Overlay weighted normalized histograms so signal vs.\ background shape (the ``peak'')
    is visible; uses the same bin edges for both classes.
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    ms = np.asarray(mass_signal, dtype=np.float64)
    ws = np.asarray(weight_signal, dtype=np.float64)
    mb = np.asarray(mass_background, dtype=np.float64)
    wb = np.asarray(weight_background, dtype=np.float64)

    if ms.size == 0 or mb.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(0.5, 0.5, "Insufficient finite masses for plot", ha="center", va="center")
        ax.set_axis_off()
        for ext in (".png", ".pdf"):
            fig.savefig(outfile.with_suffix(ext), dpi=150)
        plt.close(fig)
        return outfile.with_suffix(".pdf")

    combined = np.concatenate([ms, mb])
    lo, hi = np.percentile(combined, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(np.min(combined)), float(np.max(combined))
    n_bins = min(80, max(25, int(np.sqrt(len(combined)))))
    bins = np.linspace(lo, hi, n_bins + 1)

    dens_s, _ = np.histogram(ms, bins=bins, weights=ws, density=True)
    dens_b, _ = np.histogram(mb, bins=bins, weights=wb, density=True)
    centers = (bins[:-1] + bins[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(
        centers,
        dens_s,
        where="mid",
        color="#c44e52",
        lw=2,
        label=r"Signal ($s$)", zorder=3,
    )
    ax.step(
        centers,
        dens_b,
        where="mid",
        color="#4c72b0",
        lw=2,
        label=r"Background ($b$)", zorder=2,
    )
    ax.axvline(125.0, color="0.35", ls=":", lw=1.2, label=r"$m_H \approx 125$ GeV")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Weighted density")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), dpi=150)
    plt.close(fig)
    return outfile.with_suffix(".pdf")


def save_importance(names: list[str], imp: np.ndarray, outfile: Path, top_k: int = 20) -> Path:
    idx = np.argsort(imp)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.2)))
    ax.barh(np.array(names)[idx][::-1], imp[idx][::-1], color="#4c72b0")
    ax.set_title(f"Top {top_k} feature importances (baseline model)")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), dpi=150)
    plt.close(fig)
    return outfile.with_suffix(".pdf")
