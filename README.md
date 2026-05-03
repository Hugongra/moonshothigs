# higgs

## Tier 2 — Jupyter + CSV track

Pedagogical workflow (invariant mass → histograms → statistics): see **`notebooks/README.md`**.

CSV inventory and bundle paths: **`data/PROVENANCE.md`**.

Quick start:

```bash
cd references/cms-jupyter-materials-english-1.0/Exercises-with-open-data
jupyter notebook
```

## One-shot Python pipeline → LaTeX/PDF

The `ragsci` conda env used for `rag-ai-scientist` does **not** include `pandas` until you install it there. Easiest fix: use a **project virtualenv** (already standard if you ran setup below):

```bash
cd ~/Desktop/higgs
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-pipeline.txt
MPLCONFIGDIR="$PWD/.mplcache" python run_pipeline.py --no-compile
```

Or call the venv Python explicitly (no `activate`):

```bash
~/Desktop/higgs/.venv/bin/python run_pipeline.py
```

By default the pipeline prints **verbose** progress: absolute paths to each CSV, row counts, column names, mass statistics, figure outputs with sizes, and timings. Use **`--quiet`** (`-q`) for the short summary only.

To silence matplotlib cache warnings:

```bash
export MPLCONFIGDIR="$PWD/.mplcache"
```

### Overleaf auto-sync (push every run)

The pipeline can push the rendered NeurIPS bundle to an Overleaf project automatically on every run via Overleaf's Git integration. Details: `docs/overleaf-mcp.md`.

One-time setup:

```bash
cp configs/overleaf.example.yaml configs/overleaf.local.yaml
# edit configs/overleaf.local.yaml and paste your Overleaf git_token
# (Overleaf → Account Settings → Git Integration → Create Token).
# project_id defaults to 69f6848ce638a310664f4c90; override via env
# OVERLEAF_PROJECT_ID or the YAML if you target a different project.
```

Then pick one:

```bash
# full pipeline + push
OVERLEAF_SYNC=1 scripts/build_rag_and_paper.sh

# or directly
.venv/bin/python run_neurips_pipeline.py \
  --reuse-eval-json evals/results/neurips_full.json \
  --no-compile --sync-overleaf

# kick the tires without committing
.venv/bin/python run_neurips_pipeline.py --sync-overleaf --overleaf-dry-run

# manual one-off
.venv/bin/python scripts/sync_overleaf.py
```

Commits land on Overleaf's `master` with messages like `auto: regenerate from pipeline (<sha10> @ <UTC-iso>)`. If push fails (network, auth), the pipeline logs and continues — the local bundle at `output/neurips_overleaf_bundle/` remains valid. Optional read-only inspection of the Overleaf project from Cursor is provided by the separate [OverleafMCP](https://github.com/mjyoo2/overleafmcp) Node server; see the doc above.

**Overleaf upload (manual, no sync):** use **`output/overleaf_bundle/`** (`main.tex` + **`figures/`** together). Copy-pasting only `.tex` omits plots.

**Local PDF:** run without `--no-compile` if `pdflatex` is installed (PATH or `/Library/TeX/texbin` on macOS).

## ATLAS Higgs Challenge (2014) — baseline ML pipeline

Weighted **HistGradientBoosting** + **AMS** metric on `data/atlas-higgs-challenge-2014-v2.csv`. See **`docs/atlas_higgs_challenge_scaffolding.md`**.

```bash
.venv/bin/python run_atlas_pipeline.py              # subsample for speed
.venv/bin/python run_atlas_pipeline.py --full-train # entire training table (heavy)
.venv/bin/python run_atlas_pipeline.py --no-compile # figures + LaTeX + Overleaf bundle; skip pdflatex
```

Writes **`output/atlas_challenge/`** (figures + **`metrics.json`**, including **signal vs background** in **`DER_mass_MMC`**, the collinear-mass “peak” view), **`output/atlas_paper/paper.tex`**, and **`output/atlas_overleaf_bundle/`** (`main.tex` + `figures/`). Same Overleaf rule as below: upload the entire **`atlas_overleaf_bundle`** folder.

## NeurIPS-style draft (RAG AI Scientist + evals + ATLAS replication)

Runs **retrieval evaluation** (when a Chroma DB is available), **replicates** the ATLAS challenge baseline, and writes a draft **`neurips_rag_atlas.tex`** that explains the eval methodology and embeds case-study tables or figures. Narrative stance: **RAG AI Scientist** is primary; ATLAS is supporting evidence. Methodology text also lives in **`docs/neurips_style_draft.md`**.

```bash
pip install -r evals/requirements-eval.txt    # needed for eval + full pipeline
MPLCONFIGDIR="$PWD/.mplcache" .venv/bin/python run_neurips_pipeline.py
.venv/bin/python run_neurips_pipeline.py --skip-eval --no-compile   # paper + ATLAS only (no Chroma)
```

Outputs **`output/neurips_paper/`** (`neurips_rag_atlas.tex`, `figures/`, `pipeline_manifest.json`), **`output/neurips_overleaf_bundle/`**, and **`evals/results/neurips_*.json`** when retrieval eval succeeds.

**Paper evaluation section includes:** run configuration (embedding model, chunk count, paths, timestamps), summary metrics, **per-difficulty** breakdown, **per-query** table with first-hit ranks, a **failure cases** table (gold-labeled queries with no relevant retrieved chunk), and qualitative snippets from the top retrieved chunks.

**Recommended workflow (uses the `rag-ai-scientist` package):**

```bash
scripts/build_rag_and_paper.sh
# or with explicit env paths:
RAG_CLI=/opt/homebrew/Caskroom/miniconda/base/envs/ragsci/bin/rag-ai-scientist \
PIPE_PY=$PWD/.venv/bin/python scripts/build_rag_and_paper.sh
```

The script (1) runs `rag-ai-scientist setup-rag --project-root . --collection-name higgs-rag --force` so ATLAS / CMS / methodology docs enter the collection, (2) runs the retrieval eval in the RAG env, (3) renders the NeurIPS paper via the pipeline env. Environment variables `SKIP_INDEX=1` / `SKIP_EVAL=1` skip individual steps; `OVERLEAF_SYNC=1` also pushes the rendered bundle to the configured Overleaf project.

**Manual cross-env recipe** (when RAG deps live outside `.venv`):

```bash
/path/to/ragsci/bin/rag-ai-scientist setup-rag \
  --project-root ~/Desktop/higgs --collection-name higgs-rag --force

/path/to/ragsci/bin/python evals/run_retrieval_eval.py \
  --rag-db ~/Desktop/higgs/.cursor/rag_db \
  --collection higgs-rag \
  --output evals/results/neurips_full.json

.venv/bin/python run_neurips_pipeline.py \
  --reuse-eval-json evals/results/neurips_full.json --no-compile
```

## Higgs-style CSV peak search (paper-inspired)

Uses `data/diphoton.csv` and `data/hto4leptons.csv` with a sideband background **toy** aligned with arXiv:1207.7235 (γγ and 4ℓ channels). See **`docs/higgs_csv_methodology.md`**.

```bash
.venv/bin/python run_higgs_pipeline.py           # figures + LaTeX + PDF if pdflatex exists
.venv/bin/python run_higgs_pipeline.py --no-compile   # skip PDF only
```

Writes **`output/higgs_paper/paper.tex`**, **`output/higgs_overleaf_bundle/`** (`main.tex` + `figures/`), and **`output/higgs_figures/`** plots. Same Overleaf rule as the main pipeline: upload the **whole** `higgs_overleaf_bundle` folder, not only the `.tex`.

This runs the CSV analysis, writes figures under `output/figures/`, generates `output/paper/paper.tex`, and runs `pdflatex` twice if it is on your `PATH`. Use `python run_pipeline.py --no-compile` if you only want the `.tex` and plots.

# moonshotHiggs
