#!/usr/bin/env bash
# End-to-end driver: uses the `rag-ai-scientist` package (MCP indexer) to build
# the local Chroma DB, then runs retrieval eval + ATLAS replication + NeurIPS paper.
#
# Usage:
#   scripts/build_rag_and_paper.sh
#   SKIP_INDEX=1 scripts/build_rag_and_paper.sh          # skip setup-rag step
#   SKIP_EVAL=1 scripts/build_rag_and_paper.sh           # paper + atlas only
#   OVERLEAF_SYNC=1 scripts/build_rag_and_paper.sh       # also push bundle to Overleaf
#   ENABLE_RAGAS=1 scripts/build_rag_and_paper.sh          # Faithfulness + Context relevance (needs API keys)
#   OGTS_JSON=evals/ogts/results/latest.json scripts/build_rag_and_paper.sh  # inject OGTS tables into §4
#   RAG_CLI=/path/to/ragsci/bin/rag-ai-scientist scripts/build_rag_and_paper.sh
#
# Environments:
#   RAG_CLI   : absolute path to `rag-ai-scientist` (default: from PATH)
#   RAG_PY    : Python with langchain-chroma + langchain-huggingface (default: $RAG_CLI env)
#   PIPE_PY   : Python with pandas + sklearn + jinja2 (default: ./.venv/bin/python)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAG_CLI="${RAG_CLI:-$(command -v rag-ai-scientist || true)}"
PIPE_PY="${PIPE_PY:-$ROOT/.venv/bin/python}"
RAG_PY="${RAG_PY:-}"

if [[ -z "$RAG_CLI" ]]; then
  echo "ERROR: rag-ai-scientist CLI not found on PATH; set RAG_CLI=..." >&2
  exit 2
fi

if [[ -z "$RAG_PY" ]]; then
  # Derive the python that owns the CLI (usually the same conda env).
  RAG_PY="$(dirname "$RAG_CLI")/python"
fi

COLL_NAME="$(awk -F': *' '/^[[:space:]]*collection_name:/ {print $2; exit}' configs/references.yaml 2>/dev/null || true)"
COLL_NAME="${COLL_NAME:-higgs-rag}"
RAG_DB="$ROOT/.cursor/rag_db"
EVAL_OUT="$ROOT/evals/results/neurips_full.json"

echo "[build-rag-paper] RAG_CLI=$RAG_CLI"
echo "[build-rag-paper] RAG_PY=$RAG_PY"
echo "[build-rag-paper] PIPE_PY=$PIPE_PY"
echo "[build-rag-paper] collection=$COLL_NAME"

if [[ "${SKIP_INDEX:-0}" != "1" ]]; then
  echo "[build-rag-paper] step 1/3 -- rag-ai-scientist setup-rag"
  "$RAG_CLI" setup-rag \
    --project-root "$ROOT" \
    --collection-name "$COLL_NAME" \
    --force
else
  echo "[build-rag-paper] step 1/3 -- skipped (SKIP_INDEX=1)"
fi

RAGAS_ARGS=()
if [[ "${ENABLE_RAGAS:-0}" == "1" ]]; then
  RAGAS_ARGS=(--ragas --ragas-max-queries -1)
  echo "[build-rag-paper] RAGAS enabled (full-query sweep)"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  echo "[build-rag-paper] step 2/3 -- retrieval eval via $RAG_PY"
  "$RAG_PY" evals/run_retrieval_eval.py \
    --rag-db "$RAG_DB" \
    --collection "$COLL_NAME" \
    --k-list 5 10 \
    "${RAGAS_ARGS[@]}" \
    --output "$EVAL_OUT"
else
  echo "[build-rag-paper] step 2/3 -- skipped (SKIP_EVAL=1)"
fi

echo "[build-rag-paper] step 3/3 -- NeurIPS paper (ATLAS case study + eval tables)"
PIPE_ARGS=(--no-compile)
if [[ -f "$EVAL_OUT" ]]; then
  PIPE_ARGS=(--reuse-eval-json "$EVAL_OUT" "${PIPE_ARGS[@]}")
else
  PIPE_ARGS=(--skip-eval "${PIPE_ARGS[@]}")
fi
if [[ "${ENABLE_RAGAS:-0}" == "1" ]]; then
  PIPE_ARGS+=(--ragas --ragas-max-queries -1)
fi
if [[ -n "${OGTS_JSON:-}" && -f "${OGTS_JSON}" ]]; then
  PIPE_ARGS+=(--reuse-ogts-json "$OGTS_JSON")
fi
if [[ "${OVERLEAF_SYNC:-0}" == "1" ]]; then
  PIPE_ARGS+=(--sync-overleaf)
fi
"$PIPE_PY" run_neurips_pipeline.py "${PIPE_ARGS[@]}"

echo "[build-rag-paper] done. Outputs:"
echo "  - $RAG_DB"
echo "  - $EVAL_OUT"
echo "  - $ROOT/output/neurips_paper/"
echo "  - $ROOT/output/neurips_overleaf_bundle/"
echo "  - $ROOT/output/neurips_paper/pipeline_manifest.json"
echo "  - $ROOT/output/pipeline_manifest.json"
