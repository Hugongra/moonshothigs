#!/usr/bin/env python3
"""
Single-shot pipeline: analyze CMS education CSVs → figures → LaTeX note → optional PDF.

Usage:
  python run_pipeline.py
  python run_pipeline.py --no-compile    # skip pdflatex
  python run_pipeline.py --output-dir ./my_out
  python run_pipeline.py --quiet         # minimal logging
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="CMS open-data CSV → plots → LaTeX → PDF")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override base output directory (default: project output/)",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Write paper.tex and figures but do not run pdflatex",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (default is verbose: paths, row counts, stats, timings)",
    )
    args = p.parse_args()
    verbose = not args.quiet

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    from analysis.analyze import run_analysis
    from analysis.config import FIGURES_DIR, PAPER_BUILD_DIR, OUTPUT_DIR
    from analysis.latex_render import render_paper, write_overleaf_bundle
    from analysis.tex_compile import find_pdflatex

    if args.output_dir is not None:
        out = args.output_dir.resolve()
        fig_dir = out / "figures"
        paper_dir = out / "paper"
    else:
        out = OUTPUT_DIR
        fig_dir = FIGURES_DIR
        paper_dir = PAPER_BUILD_DIR

    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[pipeline] project root:", project_root.resolve())
        print("[pipeline] output dir:", out.resolve())
        print("[pipeline] figures dir:", fig_dir.resolve())
        print("[pipeline] paper dir:", paper_dir.resolve())
        print("[pipeline] starting analysis…")
    else:
        print("Running analysis…")

    t_analysis = time.perf_counter()
    bundle = run_analysis(figures_dir=fig_dir, verbose=verbose)
    if verbose:
        print("[pipeline] analysis wall time: {:.2f}s".format(time.perf_counter() - t_analysis))
        print("[pipeline] datasets processed:", len(bundle.datasets))
        print("[pipeline] summary (mean / std in GeV):")
        for ds in bundle.datasets:
            print(
                f"           • {ds.key}: N={ds.n_events:,}  "
                f"mean={ds.mean_mass:.4f}  std={ds.std_mass:.4f}"
            )
    else:
        print(f"  Datasets: {len(bundle.datasets)}")
        print(f"  Figures: {fig_dir}")

    if verbose:
        print("[pipeline] rendering LaTeX from paper/template.tex.j2 …")
    else:
        print("Rendering LaTeX…")
    t_tex = time.perf_counter()
    tex_path = render_paper(bundle, build_dir=paper_dir)
    if verbose:
        print(f"[pipeline] wrote {tex_path.resolve()} ({tex_path.stat().st_size:,} bytes) in {time.perf_counter() - t_tex:.2f}s")
    else:
        print(f"  {tex_path}")

    overleaf_root = out / "overleaf_bundle"
    main_tex = write_overleaf_bundle(bundle, fig_dir, overleaf_root)
    if verbose:
        print("[pipeline] Overleaf-ready bundle (upload this folder or zip it):")
        print(f"          {overleaf_root.resolve()}/")
        print(f"            main.tex   ← set as main document in Overleaf, or merge project")
        print(f"            figures/   ← PDF/PNG plots (required for \\includegraphics)")
        print(f"          ({main_tex.stat().st_size:,} bytes)")
    else:
        print(f"  Overleaf bundle: {overleaf_root}/main.tex + figures/")

    if args.no_compile:
        print(
            "\n"
            "  *** No PDF was produced — you used --no-compile ***\n"
            "  To build PDF locally, run from anywhere:\n"
            "    ~/Desktop/higgs/.venv/bin/python ~/Desktop/higgs/run_pipeline.py\n"
            "  (omit --no-compile). Requires pdflatex (MacTeX/BasicTeX), or use Overleaf with output/overleaf_bundle/\n",
            file=sys.stderr,
        )
        if verbose:
            print("[pipeline] skipping pdflatex (--no-compile).")
        else:
            print("Skipping pdflatex (--no-compile).")
        return 0

    pdflatex = find_pdflatex()
    if not pdflatex:
        print(
            "pdflatex not found. Install MacTeX/BasicTeX or add /Library/TeX/texbin to PATH.\n"
            "LaTeX sources are still in output/paper/ and output/overleaf_bundle/.",
            file=sys.stderr,
        )
        return 0

    if verbose:
        print(f"[pipeline] using pdflatex: {pdflatex}")

    print("Running pdflatex (two passes)…")
    for _ in range(2):
        r = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                str(tex_path.name),
            ],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout[-4000:] if r.stdout else "", file=sys.stderr)
            print(r.stderr[-2000:] if r.stderr else "", file=sys.stderr)
            print("pdflatex failed.", file=sys.stderr)
            return r.returncode

    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.is_file():
        sz = pdf_path.stat().st_size
        if verbose:
            print(f"[pipeline] PDF: {pdf_path.resolve()} ({sz:,} bytes)")
        else:
            print(f"Done: {pdf_path}")
        # Also compile Overleaf layout (same figures, main.tex) for a portable PDF
        main_pdf = main_tex.with_suffix(".pdf")
        if verbose:
            print("[pipeline] compiling Overleaf bundle (main.tex) for portable PDF…")
        for _ in range(2):
            r2 = subprocess.run(
                [
                    pdflatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "main.tex",
                ],
                cwd=overleaf_root,
                capture_output=True,
                text=True,
            )
            if r2.returncode != 0:
                print(r2.stdout[-4000:] if r2.stdout else "", file=sys.stderr)
                print("pdflatex on overleaf_bundle/main.tex failed.", file=sys.stderr)
                break
        else:
            if main_pdf.is_file() and verbose:
                print(f"[pipeline] PDF (Overleaf layout): {main_pdf.resolve()} ({main_pdf.stat().st_size:,} bytes)")
    else:
        print("pdflatex ran but PDF not found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
