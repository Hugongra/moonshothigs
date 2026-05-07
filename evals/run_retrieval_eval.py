#!/usr/bin/env python3
"""
Evaluate retrieval against ``evals/data/rag_queries.jsonl``.

  python evals/run_retrieval_eval.py \\
    --rag-db /path/to/rag-ai-scientist/.cursor/rag_db \\
    --k-list 5 10

Uses the same embedding model as the MCP server. Requires ``evals/requirements-eval.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.load_local_env import load_repo_dotenv
from evals.rag_store import default_collection_name, default_rag_db_path
from evals.retrieval_eval_lib import run_retrieval_evaluation


def main() -> int:
    load_repo_dotenv(ROOT)

    p = argparse.ArgumentParser(description="RAG retrieval eval for rag-ai-scientist-style corpora")
    p.add_argument(
        "--rag-db",
        type=Path,
        default=None,
        help="Chroma persist directory (default: ../rag-ai-scientist/.cursor/rag_db or ./.cursor/rag_db)",
    )
    p.add_argument("--collection", type=str, default=None, help="Chroma collection name")
    p.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model id (must match how the Chroma DB was built: HuggingFace sentence-transformers id "
        "or OpenAI text-embedding-* if langchain-openai is installed). "
        "Defaults to sentence-transformers/all-MiniLM-L6-v2 or EMBEDDING_MODEL_ID env var.",
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "evals" / "data" / "rag_queries.jsonl",
    )
    p.add_argument(
        "--k-list",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Report Recall / nDCG at these cutoffs",
    )
    p.add_argument("--max-k", type=int, default=20, help="Retrieve up to this many chunks per query")
    p.add_argument("--dry-run", action="store_true", help="Print paths only; do not load Chroma")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results here (default: evals/results/run_<timestamp>.json)",
    )
    p.add_argument(
        "--ragas",
        action="store_true",
        help="Enable optional RAGAS LLM-judge metrics (requires extra deps + OPENAI_API_KEY by default).",
    )
    p.add_argument(
        "--ragas-max-queries",
        type=int,
        default=25,
        help="When --ragas is set, only score the first N queries (default: 25). Use -1 for all queries (full-file RAGAS).",
    )
    args = p.parse_args()

    rag_db = args.rag_db or default_rag_db_path(ROOT)
    if rag_db is None:
        print(
            "[eval] No rag_db found. Pass --rag-db or index references under "
            "../rag-ai-scientist/.cursor/rag_db or .cursor/rag_db",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(f"[eval] would use rag_db={rag_db.resolve()}")
        print(f"[eval] queries={args.queries}")
        return 0

    if not args.queries.is_file():
        print(f"[eval] queries file not found: {args.queries}", file=sys.stderr)
        return 2

    collection = args.collection or default_collection_name(ROOT)
    try:
        aggregate, out_path = run_retrieval_evaluation(
            rag_db=rag_db,
            queries_path=args.queries,
            collection_name=collection,
            k_list=list(args.k_list),
            max_k=args.max_k,
            output_path=args.output,
            embedding_model_id=args.embedding_model,
            enable_ragas=bool(args.ragas),
            ragas_max_queries=(None if int(args.ragas_max_queries) < 0 else int(args.ragas_max_queries)),
        )
    except Exception as exc:
        print(f"[eval] failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({k: aggregate[k] for k in aggregate if k != "per_query"}, indent=2))
    print(f"\n[eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
