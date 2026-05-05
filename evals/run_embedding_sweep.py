#!/usr/bin/env python3
"""
Build parallel Chroma DBs (one per embedding model) and run the N-query retrieval benchmark.

Example:

  python evals/run_embedding_sweep.py \\
    --queries evals/data/rag_queries_500_neurips_mirror.jsonl \\
    --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5

Requires ingest deps (see ``evals/requirements-eval.txt``) plus optional ``langchain-openai``
for ``text-embedding-*`` models.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _slug(model_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip()).strip("_")
    return s[:96] if len(s) > 96 else s


def main() -> int:
    p = argparse.ArgumentParser(description="Embedding-model sweep: index + retrieval eval")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
        ],
        help="Embedding model ids (HF sentence-transformers or OpenAI text-embedding-*)",
    )
    p.add_argument(
        "--rag-db-parent",
        type=Path,
        default=None,
        help="Directory under which ``rag_db__<modelslug>/`` dirs are created (default: <root>/.cursor)",
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "evals" / "data" / "rag_queries_500_neurips_mirror.jsonl",
    )
    p.add_argument("--k-list", type=int, nargs="+", default=[5, 10])
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--skip-index", action="store_true", help="Only run eval; assume DBs already exist")
    p.add_argument("--force-index", action="store_true", help="Delete existing per-model DB before re-indexing")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "evals" / "results" / "embedding_sweep_summary.csv",
    )
    args = p.parse_args()

    project_root = args.project_root.resolve()
    parent = (args.rag_db_parent or (project_root / ".cursor")).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    from evals.chroma_ingest import build_chroma_from_references_yaml
    from evals.rag_store import default_collection_name
    from evals.retrieval_eval_lib import run_retrieval_evaluation

    collection = default_collection_name(project_root)
    rows_out: list[dict[str, object]] = []

    for model in args.models:
        slug = _slug(model)
        rag_db = parent / f"rag_db__{slug}"
        print(f"\n[embedding-sweep] model={model!r} -> {rag_db}", flush=True)

        if not args.skip_index:
            n_chunks = build_chroma_from_references_yaml(
                project_root,
                rag_db,
                embedding_model_id=model,
                collection_name=collection,
                force=bool(args.force_index),
            )
            print(f"[embedding-sweep] indexed chunks={n_chunks}", flush=True)

        aggregate, json_path = run_retrieval_evaluation(
            rag_db=rag_db,
            queries_path=args.queries,
            collection_name=collection,
            k_list=list(args.k_list),
            max_k=args.max_k,
            embedding_model_id=model,
        )
        row: dict[str, object] = {
            "embedding_model": model,
            "rag_db": str(rag_db),
            "eval_json": str(json_path),
            "mrr_mean": aggregate.get("mrr_mean"),
            "n_queries": aggregate.get("n_queries"),
            "rag_chunk_count": aggregate.get("rag_chunk_count"),
        }
        for k in args.k_list:
            row[f"recall_at_{k}"] = aggregate.get("recall_mean", {}).get(str(k))
            row[f"ndcg_at_{k}"] = aggregate.get("ndcg_mean", {}).get(str(k))
        rows_out.append(row)
        print(f"[embedding-sweep] wrote {json_path}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        fieldnames = list(rows_out[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
    print(f"\n[embedding-sweep] summary CSV: {args.output_csv}", flush=True)
    print(json.dumps(rows_out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
