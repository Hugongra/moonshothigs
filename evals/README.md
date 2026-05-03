# RAG AI Scientist — evaluation harness

This directory implements **Section 6–style** protocols from `docs/neurips_style_draft.md`: retrieval quality on a **gold query set**, lightweight **lexical groundedness**, optional **human rubrics**, and reproducible **JSON artifacts** under `evals/results/`.

**Paper bundle:** `../run_neurips_pipeline.py` runs retrieval eval (unless `--skip-eval`), the ATLAS replication, and writes **`output/neurips_paper/neurips_rag_atlas.tex`** with methodology prose + metric tables.

## Prerequisites

1. **Indexed corpus** — same Chroma DB used by `rag-ai-scientist` MCP (`setup-rag` / ingest).  
2. **Eval dependencies** (same embedding model as the server):

```bash
cd ~/Desktop/higgs
pip install -r evals/requirements-eval.txt
```

3. **Tell the script where Chroma lives** (pick one):

| Layout | Path |
|--------|------|
| Default discovery | `../rag-ai-scientist/.cursor/rag_db` **or** `./.cursor/rag_db` |
| Explicit | `--rag-db /absolute/path/to/.cursor/rag_db` |

Collection name defaults to `RAG_COLLECTION_NAME` or `rag-ai-scientist`.

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
| `rag_store.py` | Load Chroma + MiniLM embeddings |
| `run_retrieval_eval.py` | Main CLI |
| `report_aggregate.py` | Markdown-friendly table from a run JSON |
| `human_rubric.md` | Template for qualitative scoring |
