#!/usr/bin/env python3
"""
Higgs-style peak search on data/diphoton.csv and data/hto4leptons.csv
aligned with CMS arXiv:1207.7235 (H→γγ and H→ZZ→4l concepts).

  python run_higgs_pipeline.py
  python run_higgs_pipeline.py --no-compile    # skip pdflatex
  python run_higgs_pipeline.py -q
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Higgs peak pipeline (γγ + 4ℓ CSV) — paper-inspired sideband toy"
    )
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figures directory (default: output/higgs_figures)",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Write LaTeX but do not run pdflatex",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    fig_dir = args.output_dir or (root / "output" / "higgs_figures")
    fig_dir = fig_dir.resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    paper_dir = fig_dir.parent / "higgs_paper"
    overleaf_root = fig_dir.parent / "higgs_overleaf_bundle"

    from analysis.higgs_peak_analysis import run_higgs_peak_analysis
    from analysis.latex_render import render_higgs_paper, write_higgs_overleaf_bundle
    from analysis.tex_compile import find_pdflatex, graphicspath_latex, run_pdflatex_twice

    verbose = not args.quiet

    print("[higgs-pipeline] Paper reference: CMS arXiv:1207.7235 (§5.1 γγ; ZZ→4ℓ high-resolution channels)")
    print("[higgs-pipeline] Method: sideband-averaged background density × SR width (educational).")
    print()

    results, _figures = run_higgs_peak_analysis(fig_dir, verbose=verbose)

    gp = graphicspath_latex(paper_dir, fig_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tex_path = render_higgs_paper(results, paper_dir, graphicspath=gp)
    if verbose:
        print(f"[higgs-pipeline] wrote {tex_path} ({tex_path.stat().st_size:,} bytes) in {time.perf_counter() - t0:.2f}s")

    main_tex = write_higgs_overleaf_bundle(results, fig_dir, overleaf_root)
    if verbose:
        print(f"[higgs-pipeline] Overleaf bundle: {overleaf_root}/ (main.tex + figures/)")

    print()
    for br in results:
        print(f"[{br.channel}]")
        print(f"  caveat: {br.caveat}")
        print()
    print(f"Figures: {fig_dir}")
    print(f"LaTeX:   {tex_path}")
    print("See docs/higgs_csv_methodology.md for mapping and caveats.")

    if args.no_compile:
        print(
            "\n  *** No PDF — used --no-compile. Omit it to run pdflatex (needs TeX installed). ***\n",
            file=sys.stderr,
        )
        return 0

    pdflatex = find_pdflatex()
    if not pdflatex:
        print(
            "\npdflatex not found. Install MacTeX/BasicTeX or use Overleaf with higgs_overleaf_bundle/\n",
            file=sys.stderr,
        )
        return 0

    if verbose:
        print(f"[higgs-pipeline] pdflatex: {pdflatex}")

    r = run_pdflatex_twice(tex_path.parent, tex_path.name)
    if r is None or r.returncode != 0:
        if r:
            print(r.stdout[-3500:] if r.stdout else "", file=sys.stderr)
        print("pdflatex failed on higgs paper.tex", file=sys.stderr)
        return r.returncode if r else 1

    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.is_file() and verbose:
        print(f"[higgs-pipeline] PDF: {pdf_path} ({pdf_path.stat().st_size:,} bytes)")

    r2 = run_pdflatex_twice(main_tex.parent, main_tex.name)
    if r2 and r2.returncode == 0:
        mpdf = main_tex.with_suffix(".pdf")
        if mpdf.is_file() and verbose:
            print(f"[higgs-pipeline] PDF (Overleaf layout): {mpdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
