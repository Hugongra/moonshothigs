"""
Paper-aligned *pedagogical* bump hunt on CSV invariant masses.

Maps loosely to CMS arXiv:1207.7235:
  - H → γγ: narrow peak in diphoton invariant mass (§5.1), search ~110–150 GeV.
  - H → ZZ → 4ℓ: excess in four-lepton invariant mass (high resolution channel).

This is NOT a reproduction of CMS likelihood fits, trigger, or full backgrounds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BumpResult:
    channel: str
    n_events: int
    mass_min: float
    mass_max: float
    sr_lo: float
    sr_hi: float
    n_signal_region: float
    n_left_sb: float
    n_right_sb: float
    b_expected_sr: float
    excess: float
    significance_naive: float | None
    caveat: str
    figure_stem: str = ""


def _sideband_background(
    m: np.ndarray,
    sr: tuple[float, float],
    left_sb: tuple[float, float],
    right_sb: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    """Average SB density (events / GeV) from left & right, extrapolated into SR width."""
    sr_lo, sr_hi = sr
    lb_lo, lb_hi = left_sb
    rb_lo, rb_hi = right_sb
    w_sr = sr_hi - sr_lo
    w_l = lb_hi - lb_lo
    w_r = rb_hi - rb_lo

    n_sr = float(np.sum((m >= sr_lo) & (m < sr_hi)))
    n_l = float(np.sum((m >= lb_lo) & (m < lb_hi)))
    n_r = float(np.sum((m >= rb_lo) & (m < rb_hi)))

    rho_l = n_l / w_l if w_l > 0 else np.nan
    rho_r = n_r / w_r if w_r > 0 else np.nan
    if np.isfinite(rho_l) and np.isfinite(rho_r):
        rho = (rho_l + rho_r) / 2.0
    elif np.isfinite(rho_l):
        rho = rho_l
    elif np.isfinite(rho_r):
        rho = rho_r
    else:
        rho = 0.0

    b_exp = rho * w_sr
    return n_sr, n_l, n_r, b_exp, rho


def naive_significance(observed: float, expected_b: float) -> float | None:
    """Simple excess / sqrt(B) when B > 0."""
    if expected_b <= 0:
        return None
    excess = observed - expected_b
    return float(excess / np.sqrt(expected_b))


def analyze_diphoton(
    csv_path: Path,
    *,
    search_lo: float = 110.0,
    search_hi: float = 150.0,
    sr_lo: float = 122.0,
    sr_hi: float = 128.0,
    left_sb: tuple[float, float] = (110.0, 118.0),
    right_sb: tuple[float, float] = (132.0, 150.0),
) -> tuple[np.ndarray, BumpResult]:
    """H → γγ style: diphoton mass M in GeV (precomputed in CSV)."""
    df = pd.read_csv(csv_path)
    m = df["M"].astype(float).values
    m = m[(m >= search_lo) & (m <= search_hi)]

    n_sr, n_l, n_r, b_exp, rho = _sideband_background(
        m, (sr_lo, sr_hi), left_sb, right_sb
    )
    sig = naive_significance(n_sr, b_exp)

    caveat_parts = []
    if len(m) < 30:
        caveat_parts.append(
            "Very small sample: sideband estimates and significance are not statistically meaningful."
        )
    if n_l == 0 and n_r == 0:
        caveat_parts.append(
            "No events in sidebands — cannot estimate smooth background under CMS-like selections."
        )
    if n_sr > 0 and b_exp == 0:
        caveat_parts.append(
            "Expected background in SR is ~0 from sidebands; naive significance is undefined."
        )

    br = BumpResult(
        channel=r"H $\to\gamma\gamma$ (diphoton CSV)",
        n_events=len(m),
        mass_min=float(np.min(m)) if len(m) else float("nan"),
        mass_max=float(np.max(m)) if len(m) else float("nan"),
        sr_lo=sr_lo,
        sr_hi=sr_hi,
        n_signal_region=n_sr,
        n_left_sb=n_l,
        n_right_sb=n_r,
        b_expected_sr=float(b_exp),
        excess=float(n_sr - b_exp),
        significance_naive=sig,
        caveat=" ".join(caveat_parts)
        or "Simplified sideband estimate only; compare to CMS §5.1 blind analysis and likelihood fits.",
    )
    return m, br


def analyze_four_lepton(
    csv_path: Path,
    *,
    search_lo: float = 70.0,
    search_hi: float = 140.0,
    sr_lo: float = 121.0,
    sr_hi: float = 129.0,
    left_sb: tuple[float, float] = (70.0, 115.0),
    right_sb: tuple[float, float] = (135.0, 140.0),
) -> tuple[np.ndarray, BumpResult]:
    """H → ZZ → 4ℓ style: four-lepton invariant mass."""
    df = pd.read_csv(csv_path)
    m = df["M"].astype(float).values
    m = m[(m >= search_lo) & (m <= search_hi)]

    n_sr, n_l, n_r, b_exp, _rho = _sideband_background(
        m, (sr_lo, sr_hi), left_sb, right_sb
    )
    sig = naive_significance(n_sr, b_exp)

    caveat_parts = []
    if len(m) < 10:
        caveat_parts.append(
            "Tiny 4ℓ sample — illustrative only (full analysis uses HiggsExample20112012 / AOD)."
        )

    br = BumpResult(
        channel=r"H $\to$ ZZ $\to$ 4$\ell$ (CSV)",
        n_events=len(m),
        mass_min=float(np.min(m)) if len(m) else float("nan"),
        mass_max=float(np.max(m)) if len(m) else float("nan"),
        sr_lo=sr_lo,
        sr_hi=sr_hi,
        n_signal_region=n_sr,
        n_left_sb=n_l,
        n_right_sb=n_r,
        b_expected_sr=float(b_exp),
        excess=float(n_sr - b_exp),
        significance_naive=sig,
        caveat=" ".join(caveat_parts)
        or "Sideband-only background toy; CMS uses simulation-driven shapes and normalization.",
    )
    return m, br


def plot_mass_search(
    masses: np.ndarray,
    outfile: Path,
    title: str,
    sr_lo: float,
    sr_hi: float,
    search_range: tuple[float, float],
    bins: int = 40,
) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = search_range
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        masses,
        bins=bins,
        range=(lo, hi),
        color="#2ca02c",
        alpha=0.85,
        edgecolor="white",
    )
    ax.axvspan(sr_lo, sr_hi, color="yellow", alpha=0.25, label=f"signal region [{sr_lo},{sr_hi}] GeV")
    ax.axvline(125.0, color="crimson", linestyle="--", linewidth=1.5, label=r"$m_H\approx 125$ GeV")
    ax.set_xlabel(r"Invariant mass [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile.with_suffix(ext), dpi=150)
    plt.close(fig)
    return outfile.with_suffix(".pdf")


def run_higgs_peak_analysis(
    out_dir: Path,
    *,
    verbose: bool = True,
) -> tuple[list[BumpResult], list[Path]]:
    """Load both CSVs, compute bump metrics, save figures. Returns results and figure paths."""
    from .higgs_paths import DIPHOTON_CSV, FOURL_CSV

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _v(msg: str) -> None:
        if verbose:
            print(msg)

    results: list[BumpResult] = []
    figures: list[Path] = []

    t0 = time.perf_counter()

    if not DIPHOTON_CSV.is_file():
        raise FileNotFoundError(f"Missing {DIPHOTON_CSV}")
    if not FOURL_CSV.is_file():
        raise FileNotFoundError(f"Missing {FOURL_CSV}")

    _v(f"[higgs] diphoton CSV: {DIPHOTON_CSV}")
    m_gg, br_gg = analyze_diphoton(DIPHOTON_CSV)
    fig_gg = out_dir / "diphoton_mass_higgs_search"
    plot_mass_search(
        m_gg,
        fig_gg,
        title=r"Diphoton invariant mass (CSV) — $H\to\gamma\gamma$-style search window",
        sr_lo=br_gg.sr_lo,
        sr_hi=br_gg.sr_hi,
        search_range=(110.0, 150.0),
        bins=40,
    )
    figures.append(fig_gg.with_suffix(".pdf"))
    br_gg.figure_stem = fig_gg.name
    results.append(br_gg)
    sig_str = (
        f"{br_gg.significance_naive:.2f}"
        if br_gg.significance_naive is not None
        else "n/a (B=0 or undefined)"
    )
    _v(
        f"         events={br_gg.n_events}  SR [{br_gg.sr_lo},{br_gg.sr_hi}] count={br_gg.n_signal_region:.0f}  "
        f"B_est={br_gg.b_expected_sr:.3f}  naive S/sqrt(B)={sig_str}"
    )

    _v(f"[higgs] 4-lepton CSV: {FOURL_CSV}")
    m_4l, br_4l = analyze_four_lepton(FOURL_CSV)
    fig_4l = out_dir / "fourlepton_mass_higgs_search"
    plot_mass_search(
        m_4l,
        fig_4l,
        title=r"Four-lepton invariant mass (CSV) — $H\to ZZ\to 4\ell$-style window",
        sr_lo=br_4l.sr_lo,
        sr_hi=br_4l.sr_hi,
        search_range=(70.0, 140.0),
        bins=35,
    )
    figures.append(fig_4l.with_suffix(".pdf"))
    br_4l.figure_stem = fig_4l.name
    results.append(br_4l)
    sig_str4 = (
        f"{br_4l.significance_naive:.2f}"
        if br_4l.significance_naive is not None
        else "n/a (B=0 or undefined)"
    )
    _v(
        f"         events={br_4l.n_events}  SR [{br_4l.sr_lo},{br_4l.sr_hi}] count={br_4l.n_signal_region:.0f}  "
        f"B_est={br_4l.b_expected_sr:.3f}  naive S/sqrt(B)={sig_str4}"
    )

    _v(f"[higgs] done in {time.perf_counter() - t0:.2f}s")

    return results, figures
