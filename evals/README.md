# RAG AI Scientist — evaluation harness

This directory implements **Section 6–style** protocols from `docs/neurips_style_draft.md`: retrieval quality on a **gold query set**, lightweight **lexical groundedness** (or **RAGAS** faithfulness / context relevance when `--ragas` is enabled), optional **human rubrics**, and reproducible **JSON artifacts** under `evals/results/`.

**Paper bundle:** `../run_neurips_pipeline.py` runs retrieval eval (unless `--skip-eval`), the ATLAS replication, and writes **`output/neurips_paper/neurips_rag_atlas.tex`** with methodology prose + metric tables.

## Prerequisites

1. **Indexed corpus** — same Chroma DB used by `rag-ai-scientist` MCP (`setup-rag` / ingest).  
2. **Eval dependencies** (same embedding model as the server):

```bash
cd ~/Desktop/higgs
pip install -r evals/requirements-eval.txt
# Optional LLM-judge scores (OpenAI-backed by default; see ``evals/judge_metrics.py``):
#   pip install -r evals/requirements-ragas.txt
```

3. **Tell the script where Chroma lives** (pick one):

| Layout | Path |
|--------|------|
| Default discovery | `../rag-ai-scientist/.cursor/rag_db` **or** `./.cursor/rag_db` |
| Explicit | `--rag-db /absolute/path/to/.cursor/rag_db` |

Collection name defaults to `configs/references.yaml` (`indexing.collection_name`), then `RAG_COLLECTION_NAME`, then `rag-ai-scientist`.

## Embedding-model benchmark (parallel Chroma DBs)

When you want **comparable** MRR / nDCG across embedding checkpoints, each model needs its **own** persisted index (same chunking, different vectors):

```bash
pip install -r evals/requirements-eval.txt
python evals/run_embedding_sweep.py \
  --queries evals/data/rag_queries_500_neurips_mirror.jsonl \
  --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
```

Writes one Chroma directory per model under `.cursor/rag_db__<slug>/`, per-model eval JSON under `evals/results/`, and a CSV summary (`--output-csv`, default `evals/results/embedding_sweep_summary.csv`). OpenAI `text-embedding-*` models work if `langchain-openai` is installed and credentials are configured.

## RAGAS (`--ragas`)

Requires `pip install -r evals/requirements-ragas.txt`, Chroma available (`--rag-db` or default `./.cursor/rag_db`), and **`OPENAI_API_KEY`** in the environment. If `RAGAS_JUDGE_PROVIDER` is unset and a key is present, the harness defaults to OpenAI (`evals/judge_metrics.py`).

- **Default:** only the **first 25 queries** get RAGAS scores (cost guard). Use **`--ragas-max-queries -1`** for all lines in the JSONL (e.g. 500).

Example (500-query mirror, faithfulness + context relevance for every query):

```bash
export OPENAI_API_KEY="sk-..."
# optional: export RAGAS_JUDGE_MODEL=gpt-4o-mini
python evals/run_retrieval_eval.py \
  --queries evals/data/rag_queries_500_neurips_mirror.jsonl \
  --k-list 5 10 \
  --ragas \
  --ragas-max-queries -1 \
  --output evals/results/run_mirror_500_ragas.json
```

Expect **many** LLM calls (per query × each `k` in `--k-list`); full 500×2 can take a long time and incur noticeable API cost.

## Quickstart

```bash
cd ~/Desktop/higgs
python evals/run_retrieval_eval.py --dry-run
python evals/run_retrieval_eval.py --k-list 5 10 --max-k 20
python evals/report_aggregate.py evals/results/run_<timestamp>.json
```

Outputs:

- **Console:** aggregate Recall@k, MRR, nDCG@k, lexical groundedness (mean over queries).  
- **JSON:** `evals/results/run_<UTC>.json` with per-query breakdown (`path_hit_rank`, top metadata previews).

## Gold data format

See `evals/data/README.md` and edit `evals/data/rag_queries.jsonl`. Paths are **substring** matches on chunk metadata (`file`, `doc_path`, `source`) — tune patterns after one dry retrieval pass.

## Optional baselines (paper narrative)

| Baseline | Role |
|----------|------|
| **No retrieval** | Same prompts answered by an LLM **without** project chunks (manual or separate script). |
| **BM25 / sparse** | Future: lexical index over same corpus for hybrid comparisons. |

Document seed, embedding model id, and `k` in any publication.

## Files

| Path | Purpose |
|------|---------|
| `data/rag_queries.jsonl` | Gold queries + patterns + optional `required_terms` |
| `metrics.py` | Recall@k, MRR, nDCG (binary relevance) |
| `rag_store.py` | Load Chroma + embeddings (HF MiniLM by default; OpenAI `text-embedding-*` optional) |
| `chroma_ingest.py` | Build Chroma from `configs/references.yaml` (for sweeps / CI without `rag-ai-scientist`) |
| `run_embedding_sweep.py` | Index + eval loop across `--models` |
| `run_retrieval_eval.py` | Main CLI |
| `report_aggregate.py` | Markdown-friendly table from a run JSON |
| `human_rubric.md` | Template for qualitative scoring |
