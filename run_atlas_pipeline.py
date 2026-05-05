#!/usr/bin/env python3
"""
One-shot ATLAS Higgs Challenge (2014) baseline pipeline.

  python run_atlas_pipeline.py
  python run_atlas_pipeline.py --max-rows 80000
  python run_atlas_pipeline.py --csv /path/to/atlas-higgs-challenge-2014-v2.csv

Requires: pandas, numpy, matplotlib, scikit-learn, jinja2 (see requirements-pipeline.txt).

See docs/atlas_higgs_challenge_scaffolding.md for methodology mapping.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="ATLAS Higgs Challenge baseline ML + AMS")
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to atlas-higgs-challenge-2014-v2.csv (default: data/ under repo)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=150_000,
        help="Max training rows after KaggleSet=='t' filter (default 150000; lower=faster)",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified K-fold splits for weighted CV metrics (default: 5). Set to 1 to disable K-fold (not recommended).",
    )
    p.add_argument(
        "--full-train",
        action="store_true",
        help="Use all training rows (large RAM; slow)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/atlas_challenge)",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Write LaTeX and Overleaf bundle but do not run pdflatex",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    from analysis.atlas_challenge.pipeline import run_atlas_baseline
    from analysis.latex_render import render_atlas_challenge_paper, write_atlas_overleaf_bundle
    from analysis.tex_compile import find_pdflatex, graphicspath_latex, run_pdflatex_twice

    csv_path = args.csv or (root / "data" / "atlas-higgs-challenge-2014-v2.csv")
    max_rows = None if args.full_train else args.max_rows

    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    verbose = not args.quiet

    def _p(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    _p("")
    _p("[atlas-pipeline] ══════════ configuration ══════════")
    _p("[atlas-pipeline]   methodology: docs/atlas_higgs_challenge_scaffolding.md")
    _p(f"[atlas-pipeline]   CSV: {csv_path.resolve()}")
    _p(
        f"[atlas-pipeline]   max training rows: "
        f"{'all (full table)' if max_rows is None else f'{max_rows:,}'}"
    )
    _p(f"[atlas-pipeline]   quiet mode: {args.quiet} (use -q to suppress step logs)")

    _p("")
    _p("[atlas-pipeline] ══════════ baseline ML + plots (numbered substeps) ══════════")
    cv_folds = int(args.cv_folds)
    if cv_folds < 1:
        print("--cv-folds must be >= 1", file=sys.stderr)
        return 2

    metrics = run_atlas_baseline(
        csv_path,
        max_train_rows=max_rows,
        cv_folds=cv_folds,
        output_dir=args.output_dir,
        verbose=verbose,
    )

    fig_dir = args.output_dir or (root / "output" / "atlas_challenge")
    fig_dir = fig_dir.resolve()
    paper_dir = fig_dir.parent / "atlas_paper"
    overleaf_root = fig_dir.parent / "atlas_overleaf_bundle"

    _p("")
    _p("[atlas-pipeline] ══════════ LaTeX note (local paper.tex) ══════════")
    paper_dir.mkdir(parents=True, exist_ok=True)
    gp = graphicspath_latex(paper_dir, fig_dir)
    _p(f"[atlas-pipeline]   paper dir: {paper_dir.resolve()}/")
    _p(f"[atlas-pipeline]   figures dir (graphicspath): {fig_dir.resolve()}/")
    t0 = time.perf_counter()
    tex_path = render_atlas_challenge_paper(
        metrics, fig_dir, paper_dir, filename="paper.tex", graphicspath=gp
    )
    _p(f"[atlas-pipeline]   wrote {tex_path.resolve()} ({tex_path.stat().st_size:,} bytes, {time.perf_counter() - t0:.2f}s)")

    _p("")
    _p("[atlas-pipeline] ══════════ Overleaf bundle (upload whole folder) ══════════")
    t1 = time.perf_counter()
    main_tex = write_atlas_overleaf_bundle(metrics, fig_dir, overleaf_root)
    _p(f"[atlas-pipeline]   root: {overleaf_root.resolve()}/")
    _p(f"[atlas-pipeline]   main: {main_tex.resolve()}")
    _p(f"[atlas-pipeline]   figures: {overleaf_root.resolve() / 'figures'}/")
    _p(f"[atlas-pipeline]   ({time.perf_counter() - t1:.2f}s)")

    _p("")
    _p("[atlas-pipeline] ══════════ artifact paths (quick reference) ══════════")
    _p(f"[atlas-pipeline]   figures & metrics.json → {fig_dir}/")
    _p(f"[atlas-pipeline]   LaTeX                 → {tex_path.resolve()}")
    _p(f"[atlas-pipeline]   Overleaf bundle       → {overleaf_root.resolve()}/")

    if args.no_compile:
        print(
            "\n[atlas-pipeline] PDF skipped (--no-compile). "
            "Figures and TeX are ready; use Overleaf or omit --no-compile if pdflatex is installed.\n",
            file=sys.stderr,
        )
        return 0

    _p("")
    _p("[atlas-pipeline] ══════════ PDF compilation (optional) ══════════")
    pdflatex = find_pdflatex()
    if not pdflatex:
        print(
            "\n[atlas-pipeline] pdflatex not on PATH — skipping local PDF.\n"
            f"  • Upload for compile: {overleaf_root.resolve()}/\n"
            "  • Or install MacTeX/BasicTeX and re-run without skipping.\n"
            "  • Or pass --no-compile to silence this message when you only want plots+TeX.\n",
            file=sys.stderr,
        )
        return 0

    _p(f"[atlas-pipeline]   pdflatex: {pdflatex}")

    _p("[atlas-pipeline]   compiling atlas_paper/paper.tex …")
    r = run_pdflatex_twice(tex_path.parent, tex_path.name)
    if r is None or r.returncode != 0:
        if r:
            print(r.stdout[-3500:] if r.stdout else "", file=sys.stderr)
        print(
            f"[atlas-pipeline] pdflatex failed on {tex_path.name} (cwd={tex_path.parent})",
            file=sys.stderr,
        )
        return r.returncode if r else 1

    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.is_file():
        _p(f"[atlas-pipeline]   PDF: {pdf_path.resolve()} ({pdf_path.stat().st_size:,} bytes)")

    _p("[atlas-pipeline]   compiling Overleaf layout main.tex …")
    r2 = run_pdflatex_twice(main_tex.parent, main_tex.name)
    if r2 and r2.returncode == 0:
        mpdf = main_tex.with_suffix(".pdf")
        if mpdf.is_file():
            _p(f"[atlas-pipeline]   PDF (Overleaf layout): {mpdf.resolve()}")

    _p("[atlas-pipeline] ══════════ finished ══════════")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
