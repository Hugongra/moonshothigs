#!/usr/bin/env python3
"""Run OGSR evaluation with several API seeds and report mean ± std of global pass_rate.

Full suite (70 tasks, 5 seeds): ``python run_seeds_llama.py``

Faster smoke / variance probe (fewer tasks and seeds, roughly linear in task count):
  ``python run_seeds_llama.py --max-tasks 5 --seeds 42 1337``
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_SCRIPT = ROOT / "evals" / "ogts" / "run_ogsr_eval.py"
RESULTS_DIR = ROOT / "evals" / "results"

DEFAULT_SEEDS = (42, 1337, 2026, 9999, 12345)
LLAMA_MODEL = "meta-llama/llama-3.1-70b-instruct"


def _ogsr_pass_rate(data: dict) -> float:
    agg = data.get("aggregate") or {}
    for key in ("ogsr", "ogts"):
        block = agg.get(key)
        if isinstance(block, dict) and "pass_rate" in block:
            return float(block["pass_rate"])
    raise KeyError("aggregate.ogsr.pass_rate (or ogts alias) not found in JSON")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed OGSR pass_rate mean ± stdev (Llama 3.1 70B via OpenRouter).")
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        metavar="N",
        help="Only first N tasks from the default suite (smaller N = faster; pass_rate is on those N only).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="INT",
        help=f"Seeds to run (default: {list(DEFAULT_SEEDS)}).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS

    if not EVAL_SCRIPT.is_file():
        print(f"[run-seeds] missing eval script: {EVAL_SCRIPT}", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    n = len(seeds)
    tag = "" if args.max_tasks is None else f"_n{args.max_tasks}"

    for i, seed in enumerate(seeds, 1):
        out_path = RESULTS_DIR / f"ogsr_seed_{seed}{tag}.json"
        cmd = [
            py,
            str(EVAL_SCRIPT),
            "--generator",
            "openai",
            "--model",
            LLAMA_MODEL,
            "--strategies",
            "ogsr",
            "--depth",
            "3",
            "--branch",
            "3",
            "--seed",
            str(seed),
            "--output",
            str(out_path),
        ]
        if args.max_tasks is not None:
            cmd.extend(["--max-tasks", str(args.max_tasks)])
        print(f"[run-seeds] === {i}/{n}  seed={seed}  output={out_path} ===", flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            print(f"[run-seeds] run_ogsr_eval.py exited {proc.returncode} (seed={seed})", file=sys.stderr)
            return proc.returncode

    rates: list[float] = []
    for seed in seeds:
        path = RESULTS_DIR / f"ogsr_seed_{seed}{tag}.json"
        if not path.is_file():
            print(f"[run-seeds] expected result missing: {path}", file=sys.stderr)
            return 1
        with path.open(encoding="utf-8") as f:
            rates.append(_ogsr_pass_rate(json.load(f)))
        print(f"[run-seeds]   seed {seed:5d}  pass_rate={rates[-1]:.4f}", flush=True)

    mean = statistics.mean(rates)
    std = statistics.stdev(rates) if len(rates) > 1 else 0.0

    print("\n[run-seeds] ===== OGSR global pass_rate (across seeds) =====", flush=True)
    print(f"  model: {LLAMA_MODEL}, depth=3, branch=3", flush=True)
    if args.max_tasks is not None:
        print(
            f"  max_tasks: {args.max_tasks} (subset of 70-task suite; not comparable to full-suite pass_rate)",
            flush=True,
        )
    print(f"  seeds: {list(seeds)}", flush=True)
    print(f"  per-seed pass_rate: {[round(x, 4) for x in rates]}", flush=True)
    print(f"  mean ± std (fraction): {mean:.4f} ± {std:.4f}", flush=True)
    print(f"  mean ± std (percent):  {100.0 * mean:.2f}% ± {100.0 * std:.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
