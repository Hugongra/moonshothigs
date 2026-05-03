from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, FIGURES_DIR
from .physics import invariant_mass_dimuon_pt_eta_phi
from .plotting import save_mass_histogram


@dataclass
class DatasetResult:
    key: str
    label: str
    n_events: int
    mean_mass: float
    std_mass: float
    min_mass: float
    max_mass: float
    figure_stem: str
    notes: str = ""


@dataclass
class AnalysisBundle:
    datasets: list[DatasetResult] = field(default_factory=list)
    summary_rows: list[dict[str, str | float | int]] = field(default_factory=list)


def _stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))


def _file_size(p: Path) -> str:
    if not p.is_file():
        return "?"
    n = p.stat().st_size
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.1f} MiB"


def _v(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def run_analysis(figures_dir: Path | None = None, verbose: bool = True) -> AnalysisBundle:
    figures_dir = figures_dir or FIGURES_DIR
    bundle = AnalysisBundle()
    t0 = time.perf_counter()

    _v(verbose, "")
    _v(verbose, "[analyze] CMS bundle Data directory:")
    _v(verbose, f"          {DATA_DIR.resolve()}")
    _v(verbose, f"          exists={DATA_DIR.is_dir()}")
    _v(verbose, "")

    # --- Z → μμ (Zmumu): compute mass from (pt, η, φ) ---
    zm_path = DATA_DIR / "Zmumu_Run2011A.csv"
    t_ds = time.perf_counter()
    _v(verbose, f"[analyze] zmumu — reading {zm_path.resolve()}")
    _v(verbose, f"          file size on disk: {_file_size(zm_path)}")
    df_z = pd.read_csv(zm_path)
    _v(verbose, f"          rows loaded: {len(df_z):,}  columns: {list(df_z.columns)}")
    m_z = invariant_mass_dimuon_pt_eta_phi(df_z).values
    bad = int(np.sum(~np.isfinite(m_z)))
    if bad:
        _v(verbose, f"          WARNING: {bad} non-finite masses excluded from stats and plot")
    m_z_plot = m_z[np.isfinite(m_z)] if bad else m_z
    mean, std, mn, mx = _stats(m_z_plot)
    _v(verbose, f"          invariant mass (computed): mean={mean:.4f} GeV  std={std:.4f}  "
        f"min={mn:.4f}  max={mx:.4f} GeV")
    stem = figures_dir / "zmumu_invariant_mass"
    pdf_out = save_mass_histogram(
        m_z_plot,
        title=r"Dimuon invariant mass (Zmumu\_Run2011A)",
        xlabel=r"Invariant mass [GeV]",
        outfile=stem,
        bins=500,
    )
    png_out = stem.with_suffix(".png")
    _v(verbose, f"          wrote {pdf_out.resolve()} ({_file_size(pdf_out)})")
    _v(verbose, f"          wrote {png_out.resolve()} ({_file_size(png_out)})")
    _v(verbose, f"          step time: {time.perf_counter() - t_ds:.2f}s")
    dr = DatasetResult(
        key="zmumu",
        label="Zmumu Run2011A (computed mass)",
        n_events=len(m_z_plot),
        mean_mass=mean,
        std_mass=std,
        min_mass=mn,
        max_mass=mx,
        figure_stem=str(stem.name),
        notes="Mass from $M=\\sqrt{2p_{T1}p_{T2}(\\cosh(\\Delta\\eta)-\\cos(\\Delta\\phi))}$.",
    )
    bundle.datasets.append(dr)
    bundle.summary_rows.append(
        {
            "dataset": dr.label,
            "n": dr.n_events,
            "mean_gev": round(mean, 4),
            "std_gev": round(std, 4),
        }
    )

    # --- ϒ → μμ sample: precomputed M ---
    ym_path = DATA_DIR / "Ymumu_Run2011A.csv"
    t_ds = time.perf_counter()
    _v(verbose, f"[analyze] ymumu — reading {ym_path.resolve()}")
    _v(verbose, f"          file size on disk: {_file_size(ym_path)}")
    df_y = pd.read_csv(ym_path)
    _v(verbose, f"          rows loaded: {len(df_y):,}  columns: {list(df_y.columns)}")
    m_y = df_y["M"].astype(float).values
    mean, std, mn, mx = _stats(m_y)
    _v(verbose, f"          invariant mass (column M): mean={mean:.4f} GeV  std={std:.4f}  "
        f"min={mn:.4f}  max={mx:.4f} GeV")
    stem = figures_dir / "ymumu_invariant_mass"
    save_mass_histogram(
        m_y,
        title=r"Invariant mass (Ymumu\_Run2011A)",
        xlabel=r"Invariant mass [GeV]",
        outfile=stem,
        bins=500,
    )
    pdf_out = stem.with_suffix(".pdf")
    png_out = stem.with_suffix(".png")
    _v(verbose, f"          wrote {pdf_out.resolve()} ({_file_size(pdf_out)})")
    _v(verbose, f"          wrote {png_out.resolve()} ({_file_size(png_out)})")
    _v(verbose, f"          step time: {time.perf_counter() - t_ds:.2f}s")
    dr = DatasetResult(
        key="ymumu",
        label="Ymumu Run2011A (column M)",
        n_events=len(m_y),
        mean_mass=mean,
        std_mass=std,
        min_mass=mn,
        max_mass=mx,
        figure_stem=str(stem.name),
        notes="Invariant mass provided in CSV.",
    )
    bundle.datasets.append(dr)
    bundle.summary_rows.append(
        {
            "dataset": dr.label,
            "n": dr.n_events,
            "mean_gev": round(mean, 4),
            "std_gev": round(std, 4),
        }
    )

    # --- J/ψ region: statistics exercise sample ---
    jp_path = DATA_DIR / "Jpsimumu_Run2011A.csv"
    t_ds = time.perf_counter()
    _v(verbose, f"[analyze] jpsi — reading {jp_path.resolve()}")
    _v(verbose, f"          file size on disk: {_file_size(jp_path)}")
    df_j = pd.read_csv(jp_path)
    _v(verbose, f"          rows loaded: {len(df_j):,}  columns: {list(df_j.columns)}")
    m_j = df_j["M"].astype(float).values
    mean, std, mn, mx = _stats(m_j)
    _v(verbose, f"          invariant mass (column M): mean={mean:.4f} GeV  std={std:.4f}  "
        f"min={mn:.4f}  max={mx:.4f} GeV")
    stem = figures_dir / "jpsi_invariant_mass"
    save_mass_histogram(
        m_j,
        title=r"Invariant mass (Jpsimumu\_Run2011A)",
        xlabel=r"Invariant mass [GeV]",
        outfile=stem,
        bins=200,
    )
    pdf_out = stem.with_suffix(".pdf")
    png_out = stem.with_suffix(".png")
    _v(verbose, f"          wrote {pdf_out.resolve()} ({_file_size(pdf_out)})")
    _v(verbose, f"          wrote {png_out.resolve()} ({_file_size(png_out)})")
    _v(verbose, f"          step time: {time.perf_counter() - t_ds:.2f}s")
    dr = DatasetResult(
        key="jpsi",
        label="Jpsimumu Run2011A",
        n_events=len(m_j),
        mean_mass=mean,
        std_mass=std,
        min_mass=mn,
        max_mass=mx,
        figure_stem=str(stem.name),
        notes=r"Pedagogical $J/\psi$ selection; not the Higgs search channel.",
    )
    bundle.datasets.append(dr)
    bundle.summary_rows.append(
        {
            "dataset": dr.label,
            "n": dr.n_events,
            "mean_gev": round(mean, 4),
            "std_gev": round(std, 4),
        }
    )

    _v(verbose, "")
    _v(verbose, f"[analyze] finished all datasets in {time.perf_counter() - t0:.2f}s total")
    _v(verbose, "")

    return bundle
