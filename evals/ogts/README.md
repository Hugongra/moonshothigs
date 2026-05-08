# OGSR — Oracle-Guided Sequential Refinement evaluation

Execution-grounded evaluation harness for the **Oracle-Grounded AI Scientist (OGAS)** project.
Instead of measuring *retrieval* quality alone, this harness measures whether an LLM can
**generate correct code** verified by deterministic oracles.

The Python package path remains `evals/ogts/` for historical compatibility with frozen JSONL task IDs and results (`ogts_eval_*.json`).

## Task suite

`data/ogts_50_tasks.jsonl` — 50 micro-tasks across 4 families:

| Family | N | Entrypoint | Oracle |
|--------|---|------------|--------|
| AMS closed-form | 14 | `ams(s, b, br)` | numeric tolerance |
| Weighted log-loss | 12 | `weighted_log_loss(y, p, w)` | numeric tolerance |
| nDCG@k (binary) | 12 | `ndcg_at_k(relevances, k)` | numeric tolerance |
| Threshold scan for AMS | 12 | `best_threshold(y, p, w)` | numeric tolerance |

Each task carries 6 test cases with pre-computed expected values. Regenerate
deterministically: `python -c "from evals.ogts.task_suite import make_ogts_tasks_50, write_tasks_jsonl; ..."`.

## Strategies

- **`linear_retry`** (pass@\(k\)): sample \(k\) independent modules; return first pass.
- **`ogsr`** (**Oracle-Guided Sequential Refinement**): at each of \(d\) sequential refinement stages, draw \(b\) parallel candidates; run the oracle on each; return first pass; otherwise **greedy collapse** to the single best-scoring failure and refine the prompt once before the next stage.
- **`ogts`** — CLI **alias** for `ogsr` (same implementation).

## Generators

| Name | Flag | Needs API? | Purpose |
|------|------|------------|---------|
| `dummy` | `--generator dummy` | No | Always fails (harness smoke test) |
| `oracle` | `--generator oracle` | No | Ground-truth solutions (upper bound) |
| `noisy` | `--generator noisy` | No | Correct solutions with random bugs (local benchmarking) |
| `openai` | `--generator openai` | Yes (`OPENAI_API_KEY`) | Real LLM evaluation |

## Quick start

```bash
# Smoke test (no API):
python evals/ogts/run_ogsr_eval.py --generator oracle --max-tasks 5

# Local benchmark with simulated bugs:
python evals/ogts/run_ogsr_eval.py --generator noisy --bug-rate 0.5

# Real evaluation:
OPENAI_API_KEY=sk-... python evals/ogts/run_ogsr_eval.py \
  --generator openai --model gpt-4o-mini \
  --strategies linear_retry ogsr \
  --k 5 --depth 3 --branch 3 --temperature 0.8
```

Output: `evals/ogts/results/ogts_eval_*.json`

## Files

| Path | Purpose |
|------|---------|
| `types.py` | Task, AttemptResult, StrategyConfig dataclasses |
| `oracles.py` | Deterministic oracle evaluation (numeric, JSON) |
| `code_runner.py` | Write + import generated modules in isolation |
| `generators.py` | Code generators (dummy, oracle, noisy, OpenAI) |
| `strategies.py` | linear_retry, iterative_repair, and OGSR (`ogsr`; `ogts` alias) |
| `task_suite.py` | 50-task builder + JSONL I/O |
| `run_ogsr_eval.py` | CLI runner |
