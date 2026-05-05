from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .analyze import AnalysisBundle
from .config import PROJECT_ROOT, PAPER_BUILD_DIR

# Stems written by ``analysis.atlas_challenge.pipeline`` / ``plots.py``
ATLAS_CHALLENGE_FIGURE_STEMS: tuple[str, ...] = (
    "signal_vs_background_mass_mmc",
    "roc_validation",
    "ams_vs_threshold",
    "feature_importance_top",
)


def _latex_escape_text(s: str) -> str:
    """Escape common characters for LaTeX body text (outside verbatim)."""
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("$", r"\$")
    )


def _atlas_metrics_tex(metrics: dict[str, Any]) -> dict[str, str]:
    """Render numeric metrics and escape strings for ``atlas_challenge_template.tex.j2``."""
    def pm(mean_key: str, std_key: str, legacy_key: str, *, digits: int = 6) -> str:
        m = float(metrics.get(mean_key, metrics.get(legacy_key, 0.0)))
        if std_key in metrics and metrics.get(std_key) is not None:
            s = float(metrics.get(std_key, 0.0))
            return f"{m:.{digits}f} \\pm {s:.{digits}f}"
        return f"{m:.{digits}f}"

    # Prefer explicit *_mean/*_std keys when present (K-fold outputs), else fall back to legacy keys.
    ll = pm("validation_log_loss_mean", "validation_log_loss_std", "validation_log_loss", digits=6)
    acc = pm(
        "validation_weighted_accuracy_mean",
        "validation_weighted_accuracy_std",
        "validation_weighted_accuracy",
        digits=6,
    )
    ams = pm("validation_AMS_mean", "validation_AMS_std", "validation_AMS", digits=6)
    thr = pm(
        "best_probability_threshold_mean",
        "best_probability_threshold_std",
        "best_probability_threshold",
        digits=6,
    )

    return {
        "n_train_rows": f"{int(metrics['n_train_rows_used']):,}",
        "validation_log_loss": ll,
        "validation_weighted_accuracy": acc,
        "validation_AMS": ams,
        "best_probability_threshold": thr,
        "elapsed_seconds": f"{float(metrics['elapsed_seconds']):.2f}",
        "model": _latex_escape_text(str(metrics.get("model", ""))),
        "csv_tex": _latex_escape_text(str(metrics.get("csv", ""))),
        "note_tex": _latex_escape_text(str(metrics.get("note", ""))),
    }


def _atlas_figure_entries(figures_src_dir: Path) -> list[dict[str, str]]:
    captions = {
        "signal_vs_background_mass_mmc": (
            "Weighted density of collinear mass "
            r"$m_{\mathrm{MMC}}$ for signal vs.\ background (training sample); "
            "vertical line at 125~GeV. "
        ),
        "roc_validation": (
            "Weighted receiver operating characteristic using in-sample scores from a "
            "model refit on the full training table after K-fold CV (diagnostic; not an "
            "additional hold-out). "
        ),
        "ams_vs_threshold": (
            "Approximate median significance (AMS) vs probability threshold for the same "
            "full-data refit as the ROC figure; marker at the in-sample optimum. "
        ),
        "feature_importance_top": (
            "Impurity-based feature importances from HistGradientBoosting "
            "(when available). "
        ),
    }
    entries: list[dict[str, str]] = []
    for stem in ATLAS_CHALLENGE_FIGURE_STEMS:
        if (figures_src_dir / f"{stem}.pdf").is_file():
            cap = captions.get(stem, stem.replace("_", r"\_"))
            entries.append({"stem": stem, "caption": cap})
    return entries


def _env() -> Environment:
    tpl_dir = PROJECT_ROOT / "paper"
    return Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(enabled_extensions=()),
        block_start_string="{%",
        block_end_string="%}",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )


def render_paper(
    bundle: AnalysisBundle,
    build_dir: Path | None = None,
    filename: str = "paper.tex",
    graphicspath: str = "{../figures/}",
) -> Path:
    """Write LaTeX. Default ``graphicspath`` is one LaTeX path entry for ``\\graphicspath``."""
    build_dir = build_dir or PAPER_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)

    template = _env().get_template("template.tex.j2")
    tex = template.render(
        today=date.today().isoformat(),
        datasets=bundle.datasets,
        summary_rows=bundle.summary_rows,
        graphicspath=graphicspath,
    )
    out = build_dir / filename
    out.write_text(tex, encoding="utf-8")
    return out


def _higgs_rows_and_figures(results: list) -> tuple[list[dict], list[dict]]:
    """Build template dicts from BumpResult list (avoid circular import at module level)."""
    rows: list[dict] = []
    figs: list[dict] = []
    for br in results:
        rows.append(
            {
                "stem": br.figure_stem.replace("_", r"\_"),
                "n_events": br.n_events,
                "n_sr": f"{br.n_signal_region:.2f}",
                "n_lb": f"{br.n_left_sb:.0f}",
                "n_rb": f"{br.n_right_sb:.0f}",
                "b_est": f"{br.b_expected_sr:.4f}",
                "sig": "---"
                if br.significance_naive is None
                else f"{br.significance_naive:.2f}",
            }
        )
        c = br.caveat.replace("&", r"\&").replace("%", r"\%")
        figs.append(
            {
                "stem_tex": br.figure_stem.replace("_", r"\_"),
                "figure_stem": br.figure_stem,
                "caveat": c,
            }
        )
    return rows, figs


def render_higgs_paper(
    results: list,
    build_dir: Path,
    *,
    filename: str = "paper.tex",
    graphicspath: str = "{../higgs_figures/}",
) -> Path:
    """LaTeX note for ``run_higgs_pipeline.py`` outputs."""
    build_dir.mkdir(parents=True, exist_ok=True)
    rows, fig_blocks = _higgs_rows_and_figures(results)
    template = _env().get_template("higgs_template.tex.j2")
    tex = template.render(
        today=date.today().isoformat(),
        rows=rows,
        results=fig_blocks,
        graphicspath=graphicspath,
    )
    out = build_dir / filename
    out.write_text(tex, encoding="utf-8")
    return out


def write_higgs_overleaf_bundle(
    results: list,
    figures_src_dir: Path,
    bundle_root: Path,
) -> Path:
    """``main.tex`` + ``figures/*.pdf`` for Overleaf upload."""
    bundle_root = bundle_root.resolve()
    fig_dest = bundle_root / "figures"
    fig_dest.mkdir(parents=True, exist_ok=True)

    stems = {br.figure_stem for br in results}
    for stem in stems:
        for ext in (".pdf", ".png"):
            src = figures_src_dir / f"{stem}{ext}"
            if src.is_file():
                shutil.copy2(src, fig_dest / src.name)

    return render_higgs_paper(
        results,
        build_dir=bundle_root,
        filename="main.tex",
        graphicspath="{figures/}",
    )


def write_overleaf_bundle(
    bundle: AnalysisBundle,
    figures_src_dir: Path,
    bundle_root: Path,
) -> Path:
    """
    Flat layout for Overleaf: ``bundle_root/main.tex`` + ``bundle_root/figures/*.pdf``.
    Upload the **folder contents** (or zip the folder) so paths match ``\\graphicspath{{figures/}}``.
    """
    bundle_root = bundle_root.resolve()
    fig_dest = bundle_root / "figures"
    fig_dest.mkdir(parents=True, exist_ok=True)

    for ds in bundle.datasets:
        for ext in (".pdf", ".png"):
            src = figures_src_dir / f"{ds.figure_stem}{ext}"
            if src.is_file():
                shutil.copy2(src, fig_dest / src.name)

    main_tex = render_paper(
        bundle,
        build_dir=bundle_root,
        filename="main.tex",
        graphicspath="{figures/}",
    )
    return main_tex


def render_atlas_challenge_paper(
    metrics: dict[str, Any],
    figures_src_dir: Path,
    build_dir: Path,
    *,
    filename: str = "paper.tex",
    graphicspath: str,
) -> Path:
    """LaTeX note for ``run_atlas_pipeline.py`` outputs."""
    build_dir.mkdir(parents=True, exist_ok=True)
    figures = _atlas_figure_entries(figures_src_dir)
    template = _env().get_template("atlas_challenge_template.tex.j2")
    tex = template.render(
        today=date.today().isoformat(),
        metrics=_atlas_metrics_tex(metrics),
        figures=figures,
        graphicspath=graphicspath,
    )
    out = build_dir / filename
    out.write_text(tex, encoding="utf-8")
    return out


def write_atlas_overleaf_bundle(
    metrics: dict[str, Any],
    figures_src_dir: Path,
    bundle_root: Path,
) -> Path:
    """``main.tex`` + ``figures/*.pdf`` for Overleaf upload."""
    bundle_root = bundle_root.resolve()
    fig_dest = bundle_root / "figures"
    fig_dest.mkdir(parents=True, exist_ok=True)

    for stem in ATLAS_CHALLENGE_FIGURE_STEMS:
        for ext in (".pdf", ".png"):
            src = figures_src_dir / f"{stem}{ext}"
            if src.is_file():
                shutil.copy2(src, fig_dest / src.name)

    return render_atlas_challenge_paper(
        metrics,
        fig_dest,
        build_dir=bundle_root,
        filename="main.tex",
        graphicspath="{figures/}",
    )


def render_neurips_rag_atlas_paper(
    context: dict[str, Any],
    figures_src_dir: Path,
    build_dir: Path,
    *,
    filename: str = "neurips_rag_atlas.tex",
    graphicspath: str,
) -> Path:
    """Draft ``paper/neurips_rag_atlas.tex.j2`` --- RAG eval methodology + ATLAS case study.

    Writes atomically (temp + rename) and validates that no template markers remain.
    """
    from .robust_io import atomic_write_text, validate_rendered_tex

    build_dir.mkdir(parents=True, exist_ok=True)
    figures = _atlas_figure_entries(figures_src_dir)
    template = _env().get_template("neurips_rag_atlas.tex.j2")
    tex = template.render(
        **context,
        today=date.today().isoformat(),
        graphicspath=graphicspath,
        figures=figures,
    )
    out = build_dir / filename
    atomic_write_text(out, tex)
    validate_rendered_tex(out)

    expected_eval_present = bool(context.get("eval_available"))
    if expected_eval_present and "Aggregate metrics" not in tex:
        from .robust_io import RenderValidationError

        raise RenderValidationError(
            f"{out} reports eval_available=True but lacks an 'Aggregate metrics' block; "
            "template and context drifted apart."
        )
    return out


def write_neurips_overleaf_bundle(
    context: dict[str, Any],
    atlas_figures_src: Path,
    bundle_root: Path,
) -> Path:
    """``main.tex`` + ``figures/*.pdf`` for Overleaf (NeurIPS draft).

    Atomic file copies + validation of the rendered ``main.tex``.
    """
    from .robust_io import atomic_copy

    bundle_root = bundle_root.resolve()
    fig_dest = bundle_root / "figures"
    fig_dest.mkdir(parents=True, exist_ok=True)

    for stem in ATLAS_CHALLENGE_FIGURE_STEMS:
        for ext in (".pdf", ".png"):
            src = atlas_figures_src / f"{stem}{ext}"
            if src.is_file():
                atomic_copy(src, fig_dest / src.name)

    return render_neurips_rag_atlas_paper(
        context,
        fig_dest,
        build_dir=bundle_root,
        filename="main.tex",
        graphicspath="{figures/}",
    )
