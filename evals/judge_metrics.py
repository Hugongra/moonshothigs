"""Optional LLM-judge metrics (RAGAS) for retrieval evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagasJudgeConfig:
    provider: str  # "openai" | "azure_openai" | "disabled"
    model: str
    temperature: float


def judge_config_from_env() -> RagasJudgeConfig:
    raw = os.environ.get("RAGAS_JUDGE_PROVIDER")
    if raw is None or str(raw).strip() == "":
        # Default: use OpenAI when a key is present (common `--ragas` UX).
        prov = "openai" if os.environ.get("OPENAI_API_KEY", "").strip() else "disabled"
    else:
        prov = str(raw).strip().lower()
    model = (os.environ.get("RAGAS_JUDGE_MODEL") or "gpt-4o-mini").strip()
    temp = float(os.environ.get("RAGAS_JUDGE_TEMPERATURE") or "0.0")
    return RagasJudgeConfig(provider=prov, model=model, temperature=temp)


def faithfulness_and_context_relevance(
    *,
    query: str,
    contexts: list[str],
    answer: str,
    cfg: RagasJudgeConfig,
) -> tuple[float | None, float | None, str]:
    """
    Returns (faithfulness, context_relevance, status).

    This is intentionally optional: without API keys / ragas install, callers should skip.
    """
    if cfg.provider in {"", "disabled", "none"}:
        return None, None, "disabled"

    try:
        # ragas>=0.4: lowercase `context_relevance` was removed from ragas.metrics;
        # use legacy NVidia ContextRelevance with output column name `context_relevance`.
        from ragas.metrics import ContextRelevance, faithfulness  # type: ignore
        from ragas import evaluate  # type: ignore
        from datasets import Dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, None, f"ragas_unavailable: {exc}"

    # Lazy LLM wiring: ragas supports Langchain chat models via env in many setups.
    # We keep imports local so environments without ragas still work.
    try:
        from ragas.llms import LangchainLLMWrapper  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, None, f"judge_deps_missing: {exc}"

    if cfg.provider != "openai":
        return None, None, f"unsupported_provider:{cfg.provider}"

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, None, "missing_OPENAI_API_KEY"

    llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
    ragas_llm = LangchainLLMWrapper(llm)

    ds = Dataset.from_dict(
        {
            "question": [query],
            "contexts": [contexts],
            "answer": [answer],
        }
    )

    try:
        metrics = [faithfulness, ContextRelevance(name="context_relevance")]
        out = evaluate(ds, metrics=metrics, llm=ragas_llm)
        row = out.to_pandas().iloc[0].to_dict()
        f = row.get("faithfulness")
        cr = row.get("context_relevance")
        return (float(f) if f is not None else None, float(cr) if cr is not None else None, "ok")
    except Exception as exc:  # pragma: no cover
        return None, None, f"ragas_failed: {exc}"


def cheap_answer_from_contexts(query: str, contexts: list[str]) -> str:
    """
    Baseline 'generation' for judge metrics without an LLM generator.

    This is not a strong QA model; it exists so RAGAS faithfulness can be computed relative to
    an explicit answer string when users haven't wired a generator yet.
    """
    # Concatenate a bounded amount of context.
    blob = "\n\n".join(contexts)[:8000]
    return (
        "Answer (extractive baseline):\n"
        f"Q: {query}\n\n"
        "Supporting context (truncated):\n"
        f"{blob}"
    )
