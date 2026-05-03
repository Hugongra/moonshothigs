from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_mass_histogram(
    masses: np.ndarray,
    title: str,
    xlabel: str,
    outfile: Path,
    bins: int = 500,
    mass_range: tuple[float, float] | None = None,
) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if mass_range is not None:
        lo, hi = mass_range
        m = masses[(masses >= lo) & (masses <= hi)]
        ax.hist(m, bins=bins, color="#1f77b4", edgecolor="none", alpha=0.85)
    else:
        ax.hist(masses, bins=bins, color="#1f77b4", edgecolor="none", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.set_title(title)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), dpi=150)
    plt.close(fig)
    return outfile.with_suffix(".pdf")
