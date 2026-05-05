# OGTS — Oracle-Guided Tree Search evaluation

Execution-grounded evaluation harness for the RAG AI Scientist project.
Instead of measuring *retrieval* quality, OGTS measures whether an LLM can
**generate correct code** verified by deterministic oracles.

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

- **`linear_retry`** (pass@k): sample k independent modules; return first pass.
- **`ogts`** (Oracle-Guided Tree Search): at each of d depths, draw b branches;
  run oracle on each; return first pass; otherwise refine prompt from best survivor.

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
python evals/ogts/run_ogts_eval.py --generator oracle --max-tasks 5

# Local benchmark with simulated bugs:
python evals/ogts/run_ogts_eval.py --generator noisy --bug-rate 0.5

# Real evaluation:
OPENAI_API_KEY=sk-... python evals/ogts/run_ogts_eval.py \
  --generator openai --model gpt-4o-mini \
  --strategies linear_retry ogts \
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
| `strategies.py` | linear_retry and OGTS tree search |
| `task_suite.py` | 50-task builder + JSONL I/O |
| `run_ogts_eval.py` | CLI runner |
