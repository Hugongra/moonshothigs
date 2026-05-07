#!/usr/bin/env bash
# Run RAGAS retrieval eval + OGTS (OpenAI) using secrets from repo-root .env (gitignored).
#
#   printf '%s\n' "OPENAI_API_KEY=..." > .env   # create locally; never commit
#   scripts/run_api_experiments.sh
#
# Optional env overrides:
#   RAGAS_MAX_QUERIES=-1          # all queries (costly); default 120
#   OGTS_MODEL=gpt-4o-mini
#   SKIP_OGTS=1 / SKIP_RAGAS=1

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set. Add $ROOT/.env with OPENAI_API_KEY=sk-..." >&2
  echo "Note: Cursor/agent terminals often do NOT inherit interactive exports." >&2
  exit 2
fi

RAG_DB="${RAG_DB:-$ROOT/.cursor/rag_db}"
COLL_NAME="$(awk -F': *' '/^[[:space:]]*collection_name:/ {print $2; exit}' configs/references.yaml 2>/dev/null || true)"
COLL_NAME="${COLL_NAME:-higgs-rag}"
QUERIES="${QUERIES:-$ROOT/evals/data/rag_queries_500_neurips_mirror.jsonl}"
EVAL_OUT="${EVAL_OUT:-$ROOT/evals/results/neurips_full_ragas_experiment.json}"
RAGAS_MAX_QUERIES="${RAGAS_MAX_QUERIES:-120}"
OGTS_MODEL="${OGTS_MODEL:-gpt-4o-mini}"
OGTS_OUT="${OGTS_OUT:-$ROOT/evals/ogts/results/ogts_eval_three_strategies.json}"

PY="${PYTHON:-python3}"

echo "[experiments] rag_db=$RAG_DB collection=$COLL_NAME queries=$QUERIES"

if [[ "${SKIP_RAGAS:-0}" != "1" ]]; then
  echo "[experiments] retrieval eval + RAGAS (max_queries=$RAGAS_MAX_QUERIES) -> $EVAL_OUT"
  "$PY" "$ROOT/evals/run_retrieval_eval.py" \
    --rag-db "$RAG_DB" \
    --collection "$COLL_NAME" \
    --queries "$QUERIES" \
    --k-list 5 10 \
    --ragas \
    --ragas-max-queries "$RAGAS_MAX_QUERIES" \
    --output "$EVAL_OUT"
else
  echo "[experiments] SKIP_RAGAS=1"
fi

if [[ "${SKIP_OGTS:-0}" != "1" ]]; then
  echo "[experiments] OGTS linear_retry + iterative_repair + ogts -> $OGTS_OUT"
  "$PY" "$ROOT/evals/ogts/run_ogts_eval.py" \
    --generator openai \
    --model "$OGTS_MODEL" \
    --tasks "$ROOT/evals/ogts/data/ogts_50_tasks.jsonl" \
    --strategies linear_retry iterative_repair ogts \
    --k 5 --depth 3 --branch 3 --temperature 0.8 \
    --output "$OGTS_OUT"
else
  echo "[experiments] SKIP_OGTS=1"
fi

echo "[experiments] done."
echo "  eval JSON: $EVAL_OUT"
echo "  OGTS JSON: $OGTS_OUT"
echo "Rebuild paper (example):"
echo "  $PY run_neurips_pipeline.py --reuse-eval-json \"$EVAL_OUT\" --reuse-ogts-json \"$OGTS_OUT\" --no-compile"
