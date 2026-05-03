"""Load the same Chroma + HF embeddings stack as rag-ai-scientist MCP (local eval only)."""

from __future__ import annotations

import os
from pathlib import Path


def load_vectorstore(
    persist_directory: Path | str,
    *,
    collection_name: str | None = None,
):
    """
    Open a persisted Chroma DB created by rag-ai-scientist indexing.

    Uses ``sentence-transformers/all-MiniLM-L6-v2`` with normalized embeddings
    to match ``.cursor/mcp_server.py``.
    """
    persist_directory = Path(persist_directory)
    if not persist_directory.is_dir():
        raise FileNotFoundError(f"Chroma persist dir not found: {persist_directory}")

    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install eval deps: pip install -r evals/requirements-eval.txt"
        ) from exc

    from langchain_huggingface import HuggingFaceEmbeddings

    coll = collection_name or os.environ.get("RAG_COLLECTION_NAME", "rag-ai-scientist")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )
    return Chroma(
        persist_directory=str(persist_directory.resolve()),
        collection_name=coll,
        embedding_function=embeddings,
    )


def default_rag_db_path(project_root: Path) -> Path | None:
    """Prefer project-local ``.cursor/rag_db`` (built by ``rag-ai-scientist setup-rag``),
    then fall back to a sibling rag-ai-scientist DB for exploratory use."""
    local = project_root / ".cursor" / "rag_db"
    sibling = project_root.parent / "rag-ai-scientist" / ".cursor" / "rag_db"
    if local.is_dir():
        return local
    if sibling.is_dir():
        return sibling
    return None


def default_collection_name(project_root: Path) -> str:
    """Read collection name from ``configs/references.yaml`` if present."""
    cfg = project_root / "configs" / "references.yaml"
    if not cfg.is_file():
        return "rag-ai-scientist"
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover
        return "rag-ai-scientist"
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return "rag-ai-scientist"
    return str(data.get("indexing", {}).get("collection_name") or "rag-ai-scientist")
