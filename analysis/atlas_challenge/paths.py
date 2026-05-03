from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "atlas-higgs-challenge-2014-v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "atlas_challenge"
