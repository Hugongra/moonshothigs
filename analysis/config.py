from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE = PROJECT_ROOT / "references" / "cms-jupyter-materials-english-1.0"
DATA_DIR = BUNDLE / "Data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
PAPER_BUILD_DIR = OUTPUT_DIR / "paper"
