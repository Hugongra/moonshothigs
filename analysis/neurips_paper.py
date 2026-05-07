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


def _fmt_optional(value: object, digits: int = 4) -> str:
    if value is None:
        return "---"
    try:
        return _fmt(float(value), digits=digits)
    except (TypeError, ValueError):
        return "---"


def _summary_rows(agg: dict[str, Any], *, ragas_enabled: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rm = agg.get("recall_mean", {})
    nm = agg.get("ndcg_mean", {})
    lm = agg.get("lexical_grounded_mean", {})
    fm = agg.get("faithfulness_mean", {}) if isinstance(agg.get("faithfulness_mean", {}), dict) else {}
    crm = agg.get("context_relevance_mean", {}) if isinstance(agg.get("context_relevance_mean", {}), dict) else {}
    for k in agg.get("k_values", []):
        rows.append(
            {
                "k": int(k),
                "recall": _fmt(rm.get(str(k), 0.0)),
                "ndcg": _fmt(nm.get(str(k), 0.0)),
                "lexical": _fmt(lm.get(str(k), 0.0)),
                "faithfulness": _fmt_optional(fm.get(str(k)), digits=4) if ragas_enabled else "",
                "context_relevance": _fmt_optional(crm.get(str(k)), digits=4) if ragas_enabled else "",
            }
        )
    return rows


def _difficulty_rows(agg: dict[str, Any], *, ragas_enabled: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ks = agg.get("k_values", [])
    if not ks:
        return out
    k_hi = max(ks)
    for diff, block in sorted(agg.get("difficulty_breakdown", {}).items()):
        f_hi = None
        cr_hi = None
        if isinstance(block, dict):
            fm = block.get("faithfulness_mean") or {}
            cr = block.get("context_relevance_mean") or {}
            if isinstance(fm, dict):
                f_hi = fm.get(str(k_hi))
            if isinstance(cr, dict):
                cr_hi = cr.get(str(k_hi))
        out.append(
            {
                "difficulty": _latex_escape_text(diff),
                "n": int(block.get("n", 0)),
                "recall_hi": _fmt(block.get("recall_mean", {}).get(str(k_hi), 0.0)),
                "ndcg_hi": _fmt(block.get("ndcg_mean", {}).get(str(k_hi), 0.0)),
                "lexical_hi": _fmt(block.get("lexical_grounded_mean", {}).get(str(k_hi), 0.0)),
                "faithfulness_hi": _fmt_optional(f_hi, digits=4) if ragas_enabled else "",
                "context_relevance_hi": _fmt_optional(cr_hi, digits=4) if ragas_enabled else "",
                "mrr": _fmt(block.get("mrr_mean", 0.0)),
                "k_hi": int(k_hi),
            }
        )
    return out


def _per_query_rows(agg: dict[str, Any], *, ragas_enabled: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ks = agg.get("k_values", [])
    if not ks:
        return rows
    k_hi = max(ks)
    for pq in agg.get("per_query", []):
        rank = pq.get("path_hit_rank")
        rank_s = "--" if rank is None else str(int(rank))
        ft = None
        cr = None
        ft_map = pq.get("faithfulness_topk") or {}
        cr_map = pq.get("context_relevance_topk") or {}
        if isinstance(ft_map, dict):
            ft = ft_map.get(str(k_hi))
        if isinstance(cr_map, dict):
            cr = cr_map.get(str(k_hi))
        rows.append(
            {
                "id": _latex_escape_text(pq.get("id", "")),
                "difficulty": _latex_escape_text(pq.get("difficulty") or "--"),
                "recall_hi": _fmt(pq.get("recall", {}).get(str(k_hi), 0.0), digits=2),
                "ndcg_hi": _fmt(pq.get("ndcg", {}).get(str(k_hi), 0.0), digits=2),
                "lexical_hi": _fmt(float(pq.get("lexical_grounded_topk", {}).get(str(k_hi), 0.0)), digits=2),
                "faithfulness_hi": _fmt_optional(ft, digits=2) if ragas_enabled else "",
                "context_relevance_hi": _fmt_optional(cr, digits=2) if ragas_enabled else "",
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


def _benchmark_segment_rows(agg: dict[str, Any], *, ragas_enabled: bool) -> list[dict[str, Any]]:
    bd = agg.get("benchmark_segment_breakdown")
    if not isinstance(bd, dict) or not bd:
        return []
    ks = agg.get("k_values") or []
    if not ks:
        return []
    k_hi = max(int(k) for k in ks)
    rows: list[dict[str, Any]] = []
    for seg in sorted(bd.keys()):
        block = bd.get(seg)
        if not isinstance(block, dict) or int(block.get("n", 0)) <= 0:
            continue
        fm = block.get("faithfulness_mean") or {}
        cr = block.get("context_relevance_mean") or {}
        f_hi = fm.get(str(k_hi)) if isinstance(fm, dict) else None
        cr_hi = cr.get(str(k_hi)) if isinstance(cr, dict) else None
        rows.append(
            {
                "segment": _latex_escape_text(str(seg)),
                "n": str(int(block["n"])),
                "recall_hi": _fmt(block.get("recall_mean", {}).get(str(k_hi), 0.0)),
                "ndcg_hi": _fmt(block.get("ndcg_mean", {}).get(str(k_hi), 0.0)),
                "lexical_hi": _fmt(block.get("lexical_grounded_mean", {}).get(str(k_hi), 0.0)),
                "faithfulness_hi": _fmt_optional(f_hi, digits=4) if ragas_enabled else "---",
                "context_relevance_hi": _fmt_optional(cr_hi, digits=4) if ragas_enabled else "---",
                "n_label_fail": str(int(block.get("n_label_failures", 0))),
                "k_hi": int(k_hi),
            }
        )
    return rows


def _ogts_paper_context(ogts_data: dict[str, Any] | None) -> dict[str, Any]:
    if not ogts_data or not isinstance(ogts_data.get("aggregate"), dict):
        return {"ogts_available": False}

    agg: dict[str, Any] = ogts_data["aggregate"]
    order_pref = ["linear_retry", "iterative_repair", "ogts"]
    strat_keys = [k for k in order_pref if k in agg]
    if not strat_keys:
        return {"ogts_available": False}

    pretty = {
        "linear_retry": ("Lin.", "linear retry ($k{=}5$, $T{=}0.8$)"),
        "iterative_repair": ("Iter.", "iterative repair ($k{=}5$, $T{=}0.8$)"),
        "ogts": ("OGTS", "OGTS ($d{=}3$, $b{=}3$, $T{=}0.8$)"),
    }

    summary_rows: list[dict[str, Any]] = []
    for key in strat_keys:
        block = agg[key]
        n_tasks = int(block.get("n_tasks", 0))
        n_passed = int(block.get("n_passed", 0))
        mean_calls = float(block.get("mean_oracle_calls", 0.0))
        short, full = pretty.get(key, (key, key))
        summary_rows.append(
            {
                "strategy_short": _latex_escape_text(short),
                "strategy_long": _latex_escape_text(full),
                "pass_at_n": f"{n_passed}/{n_tasks}",
                "pass_pct": _fmt(100.0 * float(block.get("pass_rate", 0.0)), digits=1),
                "oracle_total": str(int(block.get("total_oracle_calls", 0))),
                "mean_calls": _fmt(mean_calls, digits=1),
            }
        )

    fam_src = agg[strat_keys[0]].get("by_family") or {}
    families = sorted(str(k) for k in fam_src.keys())

    fam_rows: list[dict[str, Any]] = []
    lin_key = "linear_retry" if "linear_retry" in strat_keys else strat_keys[0]
    ogs_key = "ogts" if "ogts" in strat_keys else strat_keys[-1]

    for fam in families:
        row_cells: list[str] = []
        n_tasks_f: int | None = None
        for sk in strat_keys:
            fb = (agg[sk].get("by_family") or {}).get(fam)
            if not fb:
                row_cells.extend(["---", "---"])
                continue
            if n_tasks_f is None:
                n_tasks_f = int(fb["n"])
            row_cells.append(f"{int(fb['passed'])}/{int(fb['n'])}")
            row_cells.append(str(int(fb["oracle_calls"])))

        delta_tex = "---"
        lin_b = ((agg.get(lin_key) or {}).get("by_family") or {}).get(fam)
        ogs_b = ((agg.get(ogs_key) or {}).get("by_family") or {}).get(fam)
        if isinstance(lin_b, dict) and isinstance(ogs_b, dict):
            d = int(ogs_b["passed"]) - int(lin_b["passed"])
            if d > 0:
                delta_tex = f"+{d}"
            else:
                delta_tex = str(int(d))

        fam_rows.append(
            {
                "family": _latex_escape_text(fam),
                "n_tasks_f": str(int(n_tasks_f or 0)),
                "cells": row_cells,
                "delta_pass": _latex_escape_text(delta_tex),
            }
        )

    col_spec = "lr" + ("rr" * len(strat_keys))
    if lin_key in strat_keys and ogs_key in strat_keys and lin_key != ogs_key:
        col_spec += "r"

    headers_short = [pretty.get(sk, (sk, sk))[0] for sk in strat_keys]

    return {
        "ogts_available": True,
        "ogts_summary_rows": summary_rows,
        "ogts_family_rows": fam_rows,
        "ogts_family_col_spec": col_spec,
        "ogts_family_headers": [_latex_escape_text(h) for h in headers_short],
        "ogts_show_delta": lin_key in strat_keys and ogs_key in strat_keys and lin_key != ogs_key,
        "ogts_json_timestamp": _latex_escape_text(str(ogts_data.get("timestamp", ""))),
    }


def build_neurips_context(
    *,
    atlas_metrics: dict[str, Any],
    eval_aggregate: dict[str, Any] | None,
    eval_skipped: bool,
    eval_skip_reason: str,
    eval_json_path: Path | None,
    rag_db_display: str,
    queries_file_display: str,
    ogts_aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Returns kwargs for ``render_neurips_rag_atlas_paper``."""
    atlas_tex = _atlas_metrics_tex(atlas_metrics)

    eval_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    benchmark_segment_rows: list[dict[str, Any]] = []
    mrr = ""
    n_queries = ""
    n_with_gold = ""
    n_failures = ""
    ts = ""
    embedding_model = ""
    chunk_count_display = ""

    ragas_enabled = bool(eval_aggregate.get("ragas", {}).get("enabled")) if eval_aggregate else False

    if eval_aggregate and not eval_skipped:
        eval_rows = _summary_rows(eval_aggregate, ragas_enabled=ragas_enabled)
        difficulty_rows = _difficulty_rows(eval_aggregate, ragas_enabled=ragas_enabled)
        per_query_rows = _per_query_rows(eval_aggregate, ragas_enabled=ragas_enabled)
        failure_rows = _failure_rows(eval_aggregate)
        example_rows = _example_rows(eval_aggregate)
        benchmark_segment_rows = _benchmark_segment_rows(eval_aggregate, ragas_enabled=ragas_enabled)
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

    ogts_ctx = _ogts_paper_context(ogts_aggregate)

    return {
        "atlas": atlas_tex,
        "eval_use_ragas": ragas_enabled,
        "eval_available": bool(eval_rows) and not eval_skipped,
        "eval_skipped": eval_skipped,
        "eval_skip_reason": reason_esc,
        "eval_rows": eval_rows,
        "difficulty_rows": difficulty_rows,
        "per_query_rows": per_query_rows,
        "failure_rows": failure_rows,
        "example_rows": example_rows,
        "benchmark_segment_rows": benchmark_segment_rows,
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
        **ogts_ctx,
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
