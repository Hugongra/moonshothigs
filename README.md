# RAG AI Scientist — reviewer-facing README

This repository accompanies the manuscript **“RAG AI Scientist: Retrieval Grounding for Scientific Code Assistants, with an ATLAS Higgs-Challenge Replication Testbed.”** This document is written for **NeurIPS / ICML-style reviewers**: what is claimed, what is measured, how it is implemented, and how to reproduce it.

---

## One-sentence summary

We evaluate a **retrieval-augmented scientific assistant** along three axes: **(i)** transparent retrieval metrics with explicit failure reporting, **(ii)** an **executable replication** of the 2014 ATLAS Higgs ML challenge as a consistency check between indexed documents and code, and **(iii)** **OGTS**, an execution-grounded benchmark where generated Python is checked by deterministic oracles (complementary to retrieval scores).

---

## Contributions (mapped to the paper)

1. **Evaluation contract for scientific RAG** — Hand-authored queries with path-pattern gold labels, aggregate and stratified metrics (Recall@\(k\), MRR, nDCG@\(k\)), optional lexical checks or RAGAS-style LLM-judge scores, and **published failure cases** when gold documents never appear in the top-\(k\) retrieval list.

2. **Validation testbed (ATLAS Higgs Boson ML Challenge, 2014)** — Weighted training, sentinel missingness (\(-999 \rightarrow\) NaN), AMS with regulator \(b_r{=}10\), stratified K-fold reporting. Demonstrates that the **same references** the assistant indexes align with an executable pipeline (not a leaderboard claim on private test labels).

3. **OGTS (Oracle-Guided Tree Search)** — \(N{=}50\) micro-tasks (AMS closed-form, weighted log-loss, nDCG@\(k\), AMS threshold scan) with **deterministic oracles** and JSONL task definitions. Compares **linear retry** (pass@\(k\)) vs **OGTS** under controlled budgets.

4. **Reproducibility tooling** — Configuration-driven ingest (`configs/references.yaml`), scripts under `evals/` and `run_neurips_pipeline.py`, and LaTeX generation from `paper/neurips_rag_atlas.tex.j2`.

---

## Scope and non-claims

- **Retrieval metrics are corpus-specific.** Numbers transfer only together with the **embedding checkpoint**, **chunking settings**, and **indexed snapshot** recorded in each eval JSON.
- **Path-pattern relevance is a proxy** for “correct document family” when stable chunk IDs across re-ingests are unavailable.
- **ATLAS validation AMS** is **not** the private competition leaderboard score; we report **stratified K-fold** metrics on the public training table for reproducibility.
- **OGTS** targets **small scientific numerical utilities**, not full repository-scale software engineering.

---

## Repository map

| Path | Role |
|------|------|
| `paper/neurips_rag_atlas.tex.j2` | NeurIPS-style manuscript template (Jinja placeholders filled by `run_neurips_pipeline.py`) |
| `run_neurips_pipeline.py` | Orchestrates retrieval eval (optional), ATLAS replication, paper render |
| `run_atlas_pipeline.py` | ATLAS challenge baseline (weighted boosting, AMS, figures, metrics JSON) |
| `evals/run_retrieval_eval.py` | Main retrieval evaluation CLI |
| `evals/retrieval_eval_lib.py` | Metrics + aggregation + optional RAGAS hooks |
| `evals/judge_metrics.py` | Optional LLM-judge (OpenAI / OpenRouter paths documented in code) |
| `evals/ogts/run_ogts_eval.py` | OGTS harness CLI |
| `evals/ogts/strategies.py` | **Authoritative** implementation of linear retry vs OGTS |
| `evals/ogts/data/ogts_50_tasks.jsonl` | Frozen 50-task suite |
| `evals/README.md` | Retrieval harness details (RAGAS, embedding sweep, gold JSONL format) |
| `configs/references.yaml` | Corpus manifest for indexing |
| `croissant.json` | Dataset / artifact metadata (where applicable) |

Pedagogical Jupyter workflows and CSV provenance remain documented under **`notebooks/README.md`** and **`data/PROVENANCE.md`** (orthogonal to the paper’s core claims).

---

## Evaluation axis 1 — Retrieval (summary)

- **Inputs:** Chroma persist dir + embedding model id matching the index; gold queries JSONL (`evals/data/` — starter set or generated `rag_queries_500_neurips_mirror.jsonl`).
- **Outputs:** JSON under `evals/results/` with aggregates, per-difficulty breakdowns, per-query rows, and failure-case metadata when no gold path appears in top-\(k\).

**Full commands and RAGAS costs:** see **`evals/README.md`**.

Minimal reproduction (after indexing):

```bash
pip install -r evals/requirements-eval.txt
python evals/run_retrieval_eval.py \
  --rag-db ./.cursor/rag_db \
  --k-list 5 10 \
  --queries evals/data/rag_queries.jsonl \
  --output evals/results/reviewer_retrieval.json
```

---

## Evaluation axis 2 — ATLAS replication (summary)

```bash
pip install -r requirements-pipeline.txt   # project venv recommended
export MPLCONFIGDIR="$PWD/.mplcache"
python run_atlas_pipeline.py --no-compile   # fast dev default uses subsample; see script --help for --full-train
```

Writes figures and **`output/atlas_challenge/metrics.json`** (and LaTeX bundles under `output/atlas_*`). Metrics in the paper are **mean ± std over stratified K-fold** on the training table (`KaggleSet=t`), not private test AMS.

---

## Evaluation axis 3 — OGTS (what reviewers should verify)

### Task format

Each line in `evals/ogts/data/ogts_50_tasks.jsonl` defines:

- Natural-language **prompt**,
- Python **entrypoint** name,
- **Oracle kind** (`numeric_equal`, `numeric_close`, `json_equal`),
- **`oracle_payload`** with fixed test cases (arguments + expected outputs).

The oracle **imports the generated module**, calls the entrypoint on each case, and aggregates pass/fail + score (`evals/ogts/oracles.py`).

### Strategy A — Linear retry (pass@\(k\))

For each task:

1. Repeat up to \(k\) times: sample code from the LM using the **original prompt only** (no feedback between failures).
2. Stop at the **first** module that passes all oracle cases.
3. If none pass, report best score achieved.

So failures are “caught” **only** by executable tests; there is **no** prompt refinement from oracle output.

### Strategy B — OGTS (as implemented in this repo)

Parameters: **depth** \(d\), **branch** \(b\), sampling temperature \(T\).

For each depth level \(1 \ldots d\):

1. **Expand:** Sample \(b\) **independent** candidate modules from the **current prompt** `ctx` (parallel siblings at this depth).
2. **Evaluate:** Run the oracle on **every** candidate. **Count each oracle run** (`oracle_calls` in logs).
3. **Early exit:** If **any** candidate passes, stop immediately (success).
4. **If none pass:** Sort candidates by oracle **score**, keep **only the single highest-scoring** failure.
5. **Refine:** Set `ctx` to the **original task prompt** plus a short fixed suffix containing **`Status: <best_failure.status>`** (oracle status string only — not full tracebacks, not per-case diffs).
6. Proceed to the next depth with this new `ctx`.

**What is *not* done:** This is **not** beam search retaining multiple competing hypotheses across depths. Only **one** lineage survives refinement (the best-scoring failure at each depth). Alternative siblings are **discarded** for deeper search.

**Why this still matters:** Parallel width \(b\) explores diverse corrections at each step; oracle scores gate which failure message informs the next prompt. The paper’s **nDCG family** example illustrates **near-perfect numeric overlap with systematic misuse of rank indexing** — invisible to lexical grounding but exposed by execution.

### Running OGTS

```bash
pip install openai   # OpenAI SDK; used also for OpenRouter-compatible base_url
# Smoke (no API key):
python evals/ogts/run_ogts_eval.py --generator dummy --max-tasks 2

# Example OpenRouter (see evals/ogts/generators.py for env vars & model IDs):
export OPENROUTER_API_KEY='…'
python evals/ogts/run_ogts_eval.py \
  --generator openai \
  --model anthropic/claude-3.7-sonnet \
  --tasks evals/ogts/data/ogts_50_tasks.jsonl \
  --k 5 --depth 3 --branch 3 \
  --output evals/ogts/results/ogts_eval_run.json
```

---

## Full paper bundle (NeurIPS draft)

```bash
pip install -r evals/requirements-eval.txt
export MPLCONFIGDIR="$PWD/.mplcache"
python run_neurips_pipeline.py --no-compile
```

Options commonly used in split environments:

- `--skip-eval` — ATLAS + LaTeX only (no Chroma).
- `--reuse-eval-json path/to.json` — inject a frozen retrieval JSON into the manuscript.

Automated index + eval + paper:

```bash
scripts/build_rag_and_paper.sh
```

See comments in **`scripts/build_rag_and_paper.sh`** for `SKIP_INDEX`, `SKIP_EVAL`, and Overleaf sync.

---

## Limitations (explicit)

- **OGTS refinement signal is intentionally minimal** (status string). Richer feedback (per-case oracle diffs) would likely change search effectiveness and is left to future work.
- **50 tasks, four families** — generalization beyond these scientific micro-patterns is not claimed.
- **RAGAS / LLM judges** introduce cost, variance, and provider dependence when enabled.
- **Anonymized submission:** scrub absolute user paths and API keys from any JSON you bundle as supplementary material.

---

## Ethics / data

Public challenge CSV and open methodology references are documented in **`docs/atlas_higgs_challenge_scaffolding.md`** and **`croissant.json`** where applicable. Do not commit secrets; use environment variables only.

---

## Citation

Use the citation block from the camera-ready paper once available. Until DOI assignment, cite **repository + commit SHA + eval JSON timestamps** alongside embedding model ids for any numerical claim.

---

## Maintainer note

If this README and `paper/neurips_rag_atlas.tex.j2` diverge, treat **`evals/ogts/strategies.py`** and **`evals/retrieval_eval_lib.py`** as ground truth for algorithmic behavior.

---

## Additional materials (outside the paper’s core claims)

- **Tier-2 Jupyter + CSV pedagogy:** `notebooks/README.md`
- **CSV inventory / provenance:** `data/PROVENANCE.md`
- **Higgs-style CSV peak-search pipeline:** `docs/higgs_csv_methodology.md`, `run_higgs_pipeline.py`
- **Legacy combined ops notes** (Overleaf, venv quirks) remain recoverable from git history if needed; this README is intentionally **reviewer-first**.
