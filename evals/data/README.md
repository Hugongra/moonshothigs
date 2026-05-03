# Evaluation gold data

## `rag_queries.jsonl` (one JSON object per line)

| Field | Meaning |
|--------|--------|
| `id` | Stable ID for tables and debugging. |
| `query` | Natural-language query mimicking a scientist or agent prompt. |
| `relevant_path_patterns` | Each **substring** is tested against retrieved chunk metadata `file` / `doc_path` / `source`. A chunk is **relevant** if **any** pattern matches **any** of those fields (case-insensitive). |
| `required_terms` | Optional: **all** terms must appear somewhere in the **concatenation** of top-`k` chunk texts (case-insensitive) for **lexical groundedness**. Use short distinctive tokens (e.g. `AMS`, `-999`). |
| `difficulty` | `easy` \| `medium` \| `hard` — for stratified reporting only. |
| `notes` | Human-readable intent (not used by scripts). |

Expand this file as your corpus grows; keep patterns short so minor path changes still match.

## Adding queries

1. Run retrieval manually (`retrieve_documents` MCP tool or `run_retrieval_eval.py --dry-run`).  
2. Inspect top metadata paths and chunk text.  
3. Choose `relevant_path_patterns` that identify **correct** documents without being too loose.  
4. Add `required_terms` only for facts you expect to appear verbatim (metric names, sentinel values).
