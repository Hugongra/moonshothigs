"""Paths to Higgs-oriented CSV extracts (paper-style channels)."""

from pathlib import Path

from .config import PROJECT_ROOT

HIGGS_DATA_DIR = PROJECT_ROOT / "data"
DIPHOTON_CSV = HIGGS_DATA_DIR / "diphoton.csv"
FOURL_CSV = HIGGS_DATA_DIR / "hto4leptons.csv"
