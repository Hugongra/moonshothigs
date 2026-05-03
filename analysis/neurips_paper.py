"""Build Jinja context for ``paper/neurips_rag_atlas.tex.j2``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .latex_render import ATLAS_CHALLENGE_FIGURE_STEMS, _atlas_metrics_tex, _latex_escape_text


def _fmt(value: float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "---"


def _summary_rows(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rm = agg.get("recall_mean", {})
    nm = agg.get("ndcg_mean", {})
    lm = agg.get("lexical_grounded_mean", {})
    for k in agg.get("k_values", []):
        rows.append(
            {
                "k": int(k),
                "recall": _fmt(rm.get(str(k), 0.0)),
                "ndcg": _fmt(nm.get(str(k), 0.0)),
                "lexical": _fmt(lm.get(str(k), 0.0)),
            }
        )
    return rows


def _difficulty_rows(agg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ks = agg.get("k_values", [])
    if not ks:
        return out
    k_hi = max(ks)
    for diff, block in sorted(agg.get("difficulty_breakdown", {}).items()):
        out.append(
            {
                "difficulty": _latex_escape_text(diff),
                "n": int(block.get("n", 0)),
                "recall_hi": _fmt(block.get("recall_mean", {}).get(str(k_hi), 0.0)),
                "ndcg_hi": _fmt(block.get("ndcg_mean", {}).get(str(k_hi), 0.0)),
                "lexical_hi": _fmt(block.get("lexical_grounded_mean", {}).get(str(k_hi), 0.0)),
                "mrr": _fmt(block.get("mrr_mean", 0.0)),
                "k_hi": int(k_hi),
            }
        )
    return out


def _per_query_rows(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ks = agg.get("k_values", [])
    if not ks:
        return rows
    k_hi = max(ks)
    for pq in agg.get("per_query", []):
        rank = pq.get("path_hit_rank")
        rank_s = "--" if rank is None else str(int(rank))
        rows.append(
            {
                "id": _latex_escape_text(pq.get("id", "")),
                "difficulty": _latex_escape_text(pq.get("difficulty") or "--"),
                "recall_hi": _fmt(pq.get("recall", {}).get(str(k_hi), 0.0), digits=2),
                "ndcg_hi": _fmt(pq.get("ndcg", {}).get(str(k_hi), 0.0), digits=2),
                "lexical_hi": _fmt(float(pq.get("lexical_grounded_topk", {}).get(str(k_hi), 0.0)), digits=2),
                "mrr": _fmt(pq.get("mrr", 0.0), digits=2),
                "first_hit_rank": rank_s,
                "has_gold": pq.get("has_gold", False),
                "k_hi": int(k_hi),
            }
        )
    return rows


def _failure_rows(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in agg.get("failures", []):
        rows.append(
            {
                "id": _latex_escape_text(f.get("id", "")),
                "query": _latex_escape_text(f.get("query", ""))[:90],
                "difficulty": _latex_escape_text(f.get("difficulty") or "--"),
                "top_meta": _latex_escape_text(
                    " | ".join(f.get("top_metadata_preview", [])[:2])
                )[:120],
            }
        )
    return rows


def _example_rows(agg: dict[str, Any], max_examples: int = 2) -> list[dict[str, Any]]:
    """Show first queries with snippets so reviewers can see grounded context."""
    out: list[dict[str, Any]] = []
    for pq in agg.get("per_query", [])[:max_examples]:
        snippets = pq.get("top_snippets") or []
        out.append(
            {
                "id": _latex_escape_text(pq.get("id", "")),
                "query": _latex_escape_text(pq.get("query", "")),
                "snippet": _latex_escape_text(snippets[0]) if snippets else "",
                "meta": _latex_escape_text(
                    (pq.get("top_metadata_preview") or [""])[0]
                ),
            }
        )
    return out


def build_neurips_context(
    *,
    atlas_metrics: dict[str, Any],
    eval_aggregate: dict[str, Any] | None,
    eval_skipped: bool,
    eval_skip_reason: str,
    eval_json_path: Path | None,
    rag_db_display: str,
    queries_file_display: str,
) -> dict[str, Any]:
    """Returns kwargs for ``render_neurips_rag_atlas_paper``."""
    atlas_tex = _atlas_metrics_tex(atlas_metrics)

    eval_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    mrr = ""
    n_queries = ""
    n_with_gold = ""
    n_failures = ""
    ts = ""
    embedding_model = ""
    chunk_count_display = ""

    if eval_aggregate and not eval_skipped:
        eval_rows = _summary_rows(eval_aggregate)
        difficulty_rows = _difficulty_rows(eval_aggregate)
        per_query_rows = _per_query_rows(eval_aggregate)
        failure_rows = _failure_rows(eval_aggregate)
        example_rows = _example_rows(eval_aggregate)
        ts = eval_aggregate.get("timestamp", "")
        n_queries = str(eval_aggregate.get("n_queries", ""))
        n_with_gold = str(eval_aggregate.get("n_with_gold", ""))
        n_failures = str(eval_aggregate.get("n_failures", ""))
        mrr = _fmt(eval_aggregate.get("mrr_mean", 0.0))
        embedding_model = _latex_escape_text(str(eval_aggregate.get("embedding_model", "")))
        cc = eval_aggregate.get("rag_chunk_count")
        if isinstance(cc, int) and cc >= 0:
            chunk_count_display = f"{cc:,}"
        else:
            chunk_count_display = "n/a"

    eval_json_esc = _latex_escape_text(str(eval_json_path)) if eval_json_path else "---"
    rag_esc = _latex_escape_text(rag_db_display)
    queries_esc = _latex_escape_text(queries_file_display)
    reason_esc = _latex_escape_text(eval_skip_reason)

    return {
        "atlas": atlas_tex,
        "eval_available": bool(eval_rows) and not eval_skipped,
        "eval_skipped": eval_skipped,
        "eval_skip_reason": reason_esc,
        "eval_rows": eval_rows,
        "difficulty_rows": difficulty_rows,
        "per_query_rows": per_query_rows,
        "failure_rows": failure_rows,
        "example_rows": example_rows,
        "eval_json_path": eval_json_esc,
        "rag_db_display": rag_esc,
        "queries_file_display": queries_esc,
        "eval_mrr": mrr,
        "eval_n_queries": n_queries,
        "eval_n_with_gold": n_with_gold,
        "eval_n_failures": n_failures,
        "eval_timestamp": _latex_escape_text(ts),
        "eval_embedding_model": embedding_model,
        "eval_chunk_count": chunk_count_display,
        "future_work_code_eval": _latex_escape_text(
            "Subsequent work will add automated evaluation of repository code quality "
            "(tests, linting, complexity, and reproducibility checks) complementary to "
            "retrieval metrics."
        ),
    }


def copy_atlas_figures_to_dir(figures_src: Path, dest_dir: Path) -> None:
    """Copy ATLAS pipeline PDFs into ``dest_dir`` for \\includegraphics (atomic)."""
    from .robust_io import atomic_copy

    dest_dir.mkdir(parents=True, exist_ok=True)
    for stem in ATLAS_CHALLENGE_FIGURE_STEMS:
        for ext in (".pdf", ".png"):
            src = figures_src / f"{stem}{ext}"
            if src.is_file():
                atomic_copy(src, dest_dir / src.name)
