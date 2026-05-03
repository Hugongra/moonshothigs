#!/usr/bin/env python3
"""Print a short Markdown table from a run JSON (for pasting into the paper)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_json", type=Path)
    args = p.parse_args()
    data = json.loads(args.run_json.read_text(encoding="utf-8"))
    ks = data.get("k_values", [])
    print("| Metric | " + " | ".join(f"k={k}" for k in ks) + " |")
    print("|--------|" + "|".join(["---"] * len(ks)) + "|")
    rm = data.get("recall_mean", {})
    print("| Recall@k (mean) | " + " | ".join(f"{rm.get(str(k), 0):.3f}" for k in ks) + " |")
    nm = data.get("ndcg_mean", {})
    print("| nDCG@k (mean) | " + " | ".join(f"{nm.get(str(k), 0):.3f}" for k in ks) + " |")
    lg = data.get("lexical_grounded_mean", {})
    print("| Lexical groundedness | " + " | ".join(f"{lg.get(str(k), 0):.3f}" for k in ks) + " |")
    print()
    print(f"**Mean reciprocal rank (MRR):** {data.get('mrr_mean', 0):.3f}")
    print(f"\n*n_queries={data.get('n_queries')}, rag_db={data.get('rag_db')}*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
