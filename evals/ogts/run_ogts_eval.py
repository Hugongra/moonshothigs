#!/usr/bin/env python3
"""
Run OGTS evaluation: linear-retry vs Oracle-Guided Tree Search on 50 tasks.

Examples:

  # Smoke test with dummy generator (no API key needed):
  python evals/ogts/run_ogts_eval.py --generator dummy --tasks evals/ogts/data/ogts_50_tasks.jsonl

  # Real evaluation with OpenAI:
  OPENAI_API_KEY=sk-... python evals/ogts/run_ogts_eval.py \
    --generator openai --model gpt-4o-mini \
    --tasks evals/ogts/data/ogts_50_tasks.jsonl

  # Only OGTS strategy:
  python evals/ogts/run_ogts_eval.py --generator openai --strategies ogts

Output: evals/ogts/results/ogts_eval_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.ogts.generators import CodeGenerator, DummyGenerator, NoisyGenerator, OracleGenerator, OpenAIGenerator
from evals.ogts.oracles import summarize_attempt_for_json
from evals.ogts.strategies import RunStats, linear_retry, ogts
from evals.ogts.task_suite import load_tasks_jsonl
from evals.ogts.types import AttemptResult, OgtsTask


def _run_one(
    task: OgtsTask,
    gen: CodeGenerator,
    strategy: str,
    *,
    k: int,
    depth: int,
    branch: int,
    temperature: float,
) -> dict:
    t0 = time.perf_counter()
    if strategy == "linear_retry":
        res, stats = linear_retry(task=task, gen=gen, k=k, temperature=temperature)
    elif strategy == "ogts":
        res, stats = ogts(task=task, gen=gen, depth=depth, branch=branch, temperature=temperature)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    elapsed = time.perf_counter() - t0

    return {
        "task_id": task.id,
        "task_title": task.title,
        "strategy": strategy,
        "passed": bool(stats.passed),
        "best_score": float(stats.best_score),
        "best_status": str(stats.best_status),
        "oracle_calls": int(stats.oracle_calls),
        "attempts": int(stats.attempts),
        "elapsed_s": round(elapsed, 3),
        "result": summarize_attempt_for_json(res) if res else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="OGTS execution-grounded evaluation")
    p.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "evals" / "ogts" / "data" / "ogts_50_tasks.jsonl",
    )
    p.add_argument(
        "--generator",
        choices=["dummy", "oracle", "noisy", "openai"],
        default="openai",
        help="'oracle' = ground-truth upper bound; 'noisy' = simulated buggy LLM (no API); 'openai' = real model.",
    )
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--bug-rate", type=float, default=0.6, help="Bug injection probability for noisy generator")
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=["linear_retry", "ogts"],
        default=["linear_retry", "ogts"],
    )
    p.add_argument("--k", type=int, default=5, help="pass@k budget for linear_retry")
    p.add_argument("--depth", type=int, default=3, help="OGTS tree depth")
    p.add_argument("--branch", type=int, default=3, help="OGTS branches per depth")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (for testing)")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    tasks = load_tasks_jsonl(args.tasks)
    if args.max_tasks:
        tasks = tasks[: int(args.max_tasks)]

    if args.generator == "dummy":
        gen: CodeGenerator = DummyGenerator(mode="always_fail")
    elif args.generator == "oracle":
        gen = OracleGenerator()
    elif args.generator == "noisy":
        gen = NoisyGenerator(bug_rate=float(args.bug_rate))
    else:
        gen = OpenAIGenerator(model=args.model)

    print(
        f"[ogts-eval] tasks={len(tasks)}, generator={args.generator}({getattr(gen, 'model', 'n/a')}), "
        f"strategies={args.strategies}, k={args.k}, depth={args.depth}, branch={args.branch}, T={args.temperature}",
        flush=True,
    )

    all_results: list[dict] = []
    t_global = time.perf_counter()

    for i, task in enumerate(tasks, 1):
        for strat in args.strategies:
            print(f"[ogts-eval] [{i}/{len(tasks)}] {task.id} ({task.title}) strategy={strat}", flush=True)
            row = _run_one(
                task,
                gen,
                strat,
                k=args.k,
                depth=args.depth,
                branch=args.branch,
                temperature=args.temperature,
            )
            tag = "PASS" if row["passed"] else "FAIL"
            print(
                f"[ogts-eval]   -> {tag} score={row['best_score']:.3f} "
                f"oracle_calls={row['oracle_calls']} elapsed={row['elapsed_s']:.1f}s",
                flush=True,
            )
            all_results.append(row)

    wall = time.perf_counter() - t_global

    # Aggregate by strategy.
    agg: dict[str, dict] = {}
    for strat in args.strategies:
        rows = [r for r in all_results if r["strategy"] == strat]
        n = len(rows)
        n_pass = sum(1 for r in rows if r["passed"])
        total_oracle = sum(r["oracle_calls"] for r in rows)
        total_elapsed = sum(r["elapsed_s"] for r in rows)
        by_family: dict[str, dict] = {}
        for r in rows:
            fam = r["task_title"]
            if fam not in by_family:
                by_family[fam] = {"n": 0, "passed": 0, "oracle_calls": 0}
            by_family[fam]["n"] += 1
            by_family[fam]["passed"] += int(r["passed"])
            by_family[fam]["oracle_calls"] += r["oracle_calls"]
        agg[strat] = {
            "n_tasks": n,
            "n_passed": n_pass,
            "pass_rate": round(n_pass / max(1, n), 4),
            "total_oracle_calls": total_oracle,
            "mean_oracle_calls": round(total_oracle / max(1, n), 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "by_family": by_family,
        }

    output_obj = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "generator": args.generator,
            "model": getattr(gen, "model", None),
            "strategies": args.strategies,
            "k": args.k,
            "depth": args.depth,
            "branch": args.branch,
            "temperature": args.temperature,
            "n_tasks": len(tasks),
            "tasks_file": str(args.tasks.resolve()),
        },
        "aggregate": agg,
        "wall_time_s": round(wall, 2),
        "per_task": all_results,
    }

    out_dir = ROOT / "evals" / "ogts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (out_dir / f"ogts_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}.json")
    out_path.write_text(json.dumps(output_obj, indent=2), encoding="utf-8")

    print(f"\n[ogts-eval] ===== SUMMARY (wall={wall:.1f}s) =====", flush=True)
    for strat, s in agg.items():
        print(
            f"  {strat}: pass={s['n_passed']}/{s['n_tasks']} ({s['pass_rate']:.1%}), "
            f"oracle_calls={s['total_oracle_calls']} (mean={s['mean_oracle_calls']:.1f}), "
            f"elapsed={s['total_elapsed_s']:.1f}s",
            flush=True,
        )
        for fam, fb in s["by_family"].items():
            print(f"    {fam}: {fb['passed']}/{fb['n']} pass, {fb['oracle_calls']} oracle calls", flush=True)

    print(f"\n[ogts-eval] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
