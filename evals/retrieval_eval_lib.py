"""Programmatic retrieval evaluation (shared by CLI and NeurIPS pipeline)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import mean, ndcg_at_k, reciprocal_rank, recall_at_k
from .rag_store import load_vectorstore
from .judge_metrics import cheap_answer_from_contexts, faithfulness_and_context_relevance, judge_config_from_env

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class MissingEvalDependencies(RuntimeError):
    """Raised when langchain-chroma / langchain-huggingface are absent in the current env."""


def _metadata_path(doc) -> str:
    md = doc.metadata or {}
    parts = []
    for key in ("file", "doc_path", "source", "path"):
        v = md.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts).lower()


def chunk_relevant(doc, patterns: list[str]) -> bool:
    if not patterns:
        return False
    hay = _metadata_path(doc)
    return any(p.lower() in hay for p in patterns)


def lexical_grounded(concat_text: str, required_terms: list[str]) -> bool:
    if not required_terms:
        return True
    low = concat_text.lower()
    return all(t.lower() in low for t in required_terms)


def run_retrieval_evaluation(
    *,
    rag_db: Path,
    queries_path: Path,
    collection_name: str | None = None,
    k_list: list[int] | None = None,
    max_k: int = 20,
    output_path: Path | None = None,
    results_dir: Path | None = None,
    embedding_model_id: str | None = None,
    enable_ragas: bool = False,
    ragas_max_queries: int | None = None,
) -> tuple[dict[str, Any], Path]:
    """
    Run gold-query retrieval evaluation; write JSON and return (aggregate, path).

    Raises FileNotFoundError if ``queries_path`` or ``rag_db`` is missing.
    """
    k_list = k_list or [5, 10]
    if not queries_path.is_file():
        raise FileNotFoundError(queries_path)
    if not rag_db.is_dir():
        raise FileNotFoundError(rag_db)

    rows: list[dict] = []
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    try:
        vs = load_vectorstore(
            rag_db,
            collection_name=collection_name,
            embedding_model_id=embedding_model_id,
        )
    except ImportError as exc:
        raise MissingEvalDependencies(
            "Eval dependencies missing in active Python environment. Install with:\n"
            "    pip install -r evals/requirements-eval.txt\n"
            f"(underlying error: {exc})"
        ) from exc

    try:  # pragma: no cover (Chroma internals vary)
        chunk_count = int(vs._collection.count())  # type: ignore[attr-defined]
    except Exception:
        chunk_count = -1

    ks = sorted(set(k_list))
    k_max = max(max_k, max(ks))

    per_query: list[dict] = []
    judge_cfg = judge_config_from_env()
    judge_status_global = "disabled"
    if enable_ragas:
        judge_status_global = f"enabled_provider={judge_cfg.provider}"

    for item in rows:
        patterns = item.get("relevant_path_patterns") or []
        req_terms = item.get("required_terms") or []

        docs = vs.similarity_search(item["query"], k=k_max)
        hits = [chunk_relevant(d, patterns) for d in docs]
        while len(hits) < k_max:
            hits.append(False)

        pq: dict[str, Any] = {
            "id": item["id"],
            "query": item["query"],
            "difficulty": item.get("difficulty", ""),
            "has_gold": bool(patterns),
            "recall": {str(k): recall_at_k(hits, k) for k in ks},
            "mrr": reciprocal_rank(hits),
            "ndcg": {str(k): ndcg_at_k(hits, k) for k in ks},
            "lexical_grounded_topk": {
                str(k): lexical_grounded("\n".join(d.page_content for d in docs[:k]), req_terms)
                for k in ks
            },
            "top_metadata_preview": [_metadata_path(d)[:160] for d in docs[:3]],
            "top_snippets": [d.page_content[:300].replace("\n", " ") for d in docs[:2]],
            "required_terms": req_terms,
            "relevant_path_patterns": patterns,
        }
        pq["path_hit_rank"] = (
            next((i + 1 for i, h in enumerate(hits) if h), None) if patterns else None
        )

        # Optional RAGAS judge metrics (expensive; off by default).
        pq["faithfulness_topk"] = {str(k): None for k in ks}
        pq["context_relevance_topk"] = {str(k): None for k in ks}
        pq["ragas_status"] = "disabled"

        if enable_ragas:
            # Bound cost: only evaluate first N queries unless unlimited.
            limit = ragas_max_queries
            if limit is None:
                limit = 25
            q_index = len(per_query)
            if q_index < int(limit):
                qtext = str(item["query"])
                for k in ks:
                    ctxs = [d.page_content for d in docs[:k]]
                    ans = cheap_answer_from_contexts(qtext, ctxs)
                    f, cr, st = faithfulness_and_context_relevance(
                        query=qtext, contexts=ctxs, answer=ans, cfg=judge_cfg
                    )
                    pq["faithfulness_topk"][str(k)] = f
                    pq["context_relevance_topk"][str(k)] = cr
                pq["ragas_status"] = st
                judge_status_global = st

        per_query.append(pq)

    difficulty_groups: dict[str, list[dict]] = defaultdict(list)
    for pq in per_query:
        difficulty_groups[pq.get("difficulty") or "unlabeled"].append(pq)

    def _mean_optional(xs: list[float | None]) -> float | None:
        ys = [float(x) for x in xs if x is not None]
        return mean(ys) if ys else None

    difficulty_breakdown = {}
    for d, group in difficulty_groups.items():
        difficulty_breakdown[d] = {
            "n": len(group),
            "recall_mean": {str(k): mean([p["recall"][str(k)] for p in group]) for k in ks},
            "mrr_mean": mean([p["mrr"] for p in group]),
            "ndcg_mean": {str(k): mean([p["ndcg"][str(k)] for p in group]) for k in ks},
            "lexical_grounded_mean": {
                str(k): mean([float(p["lexical_grounded_topk"][str(k)]) for p in group])
                for k in ks
            },
        }
        if enable_ragas:
            difficulty_breakdown[d]["faithfulness_mean"] = {
                str(k): _mean_optional([p["faithfulness_topk"][str(k)] for p in group]) for k in ks
            }
            difficulty_breakdown[d]["context_relevance_mean"] = {
                str(k): _mean_optional([p["context_relevance_topk"][str(k)] for p in group])
                for k in ks
            }

    with_gold = [p for p in per_query if p["has_gold"]]
    failures = [
        {
            "id": p["id"],
            "query": p["query"],
            "difficulty": p["difficulty"],
            "top_metadata_preview": p["top_metadata_preview"],
        }
        for p in with_gold
        if p.get("path_hit_rank") is None
    ]

    aggregate: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model_id or EMBEDDING_MODEL_ID,
        "rag_db": str(rag_db.resolve()),
        "rag_chunk_count": chunk_count,
        "queries_file": str(queries_path.resolve()),
        "n_queries": len(rows),
        "n_with_gold": len(with_gold),
        "n_failures": len(failures),
        "difficulty_counts": dict(Counter(p.get("difficulty") or "unlabeled" for p in per_query)),
        "k_values": ks,
        "recall_mean": {str(k): mean([p["recall"][str(k)] for p in per_query]) for k in ks},
        "mrr_mean": mean([p["mrr"] for p in per_query]),
        "ndcg_mean": {str(k): mean([p["ndcg"][str(k)] for p in per_query]) for k in ks},
        "lexical_grounded_mean": {
            str(k): mean([float(p["lexical_grounded_topk"][str(k)]) for p in per_query])
            for k in ks
        },
        "difficulty_breakdown": difficulty_breakdown,
        "failures": failures,
        "per_query": per_query,
    }

    aggregate["ragas"] = {
        "enabled": bool(enable_ragas),
        "max_queries": ragas_max_queries,
        "status": judge_status_global,
        "judge_model": judge_cfg.model,
        "judge_provider": judge_cfg.provider,
        "note": (
            "RAGAS metrics are optional and require extra dependencies + LLM credentials. "
            "When enabled, this harness uses a cheap extractive 'answer' built from the retrieved "
            "contexts so Faithfulness/ContextRelevance can run without a separate generator model."
        ),
    }

    if enable_ragas:
        aggregate["faithfulness_mean"] = {
            str(k): _mean_optional([p["faithfulness_topk"][str(k)] for p in per_query]) for k in ks
        }
        aggregate["context_relevance_mean"] = {
            str(k): _mean_optional([p["context_relevance_topk"][str(k)] for p in per_query])
            for k in ks
        }

    base = results_dir or (Path(__file__).resolve().parent / "results")
    base.mkdir(parents=True, exist_ok=True)
    out_path = output_path or base / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}.json"
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate, out_path.resolve()
