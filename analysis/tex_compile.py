"""Locate pdflatex and compute LaTeX \\graphicspath entries."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_pdflatex() -> str | None:
    w = shutil.which("pdflatex")
    if w:
        return w
    for c in (
        Path("/Library/TeX/texbin/pdflatex"),
        Path("/usr/local/texlive/2025/bin/universal-darwin/pdflatex"),
        Path("/usr/local/texlive/2024/bin/universal-darwin/pdflatex"),
        Path("/opt/homebrew/bin/pdflatex"),
    ):
        if c.is_file():
            return str(c)
    return None


def graphicspath_latex(paper_dir: Path, figures_dir: Path) -> str:
    """One LaTeX path entry for ``\\graphicspath``, relative *from* paper dir *to* figures dir."""
    rel = os.path.relpath(figures_dir.resolve(), paper_dir.resolve())
    return "{" + rel.replace(os.sep, "/") + "/}"


def run_pdflatex_twice(cwd: Path, tex_name: str) -> subprocess.CompletedProcess | None:
    """Return last CompletedProcess; caller checks .returncode."""
    pdflatex = find_pdflatex()
    if not pdflatex:
        return None
    last = None
    for _ in range(2):
        last = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_name,
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if last.returncode != 0:
            return last
    return last
