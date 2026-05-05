"""
Build a Chroma vector store from ``configs/references.yaml`` (same sources as ``setup-rag``).

Used for embedding-model benchmarks: each model gets its own persist directory so MRR/nDCG
are comparable without reusing mismatched embedding spaces.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from .rag_store import embedding_from_model_id


def _load_yaml(project_root: Path) -> dict[str, Any]:
    import yaml

    cfg_path = project_root / "configs" / "references.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _iter_source_files(project_root: Path, cfg: dict[str, Any]) -> Iterable[tuple[Path, str]]:
    """Yield (absolute_path, logical_name) for every file under configured sources."""
    for block in cfg.get("sources", []):
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (block.get("extensions") or [])}
        for rel in block.get("paths", []):
            base = (project_root / rel).resolve()
            if base.is_file():
                if base.suffix.lower() in exts:
                    yield base, f"{block.get('name', 'source')}/{base.name}"
                continue
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    rel_name = str(p.relative_to(base)).replace("\\", "/")
                    yield p, f"{block.get('name', 'source')}/{rel_name}"


def _load_file_documents(path: Path, logical_name: str) -> list[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

    suf = path.suffix.lower()
    if suf == ".pdf":
        docs = PyMuPDFLoader(str(path)).load()
    elif suf in {".md", ".txt", ".tex"}:
        try:
            docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
        except TypeError:
            docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        return []
    for d in docs:
        d.metadata = dict(d.metadata or {})
        d.metadata.setdefault("source", str(path))
        d.metadata.setdefault("doc_path", logical_name)
    return docs


def build_chroma_from_references_yaml(
    project_root: Path,
    persist_directory: Path,
    *,
    embedding_model_id: str,
    collection_name: str | None = None,
    force: bool = False,
) -> int:
    """
    Chunk configured sources, embed with ``embedding_model_id``, persist Chroma.

    Returns the number of chunks stored.
    """
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    project_root = project_root.resolve()
    persist_directory = Path(persist_directory).resolve()
    cfg = _load_yaml(project_root)
    idx = cfg.get("indexing") or {}
    coll = collection_name or str(idx.get("collection_name") or "higgs-rag")

    raw_docs: list[Document] = []
    for path, logical in _iter_source_files(project_root, cfg):
        try:
            raw_docs.extend(_load_file_documents(path, logical))
        except Exception:
            continue

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(idx.get("chunk_size", 900)),
        chunk_overlap=int(idx.get("chunk_overlap", 250)),
    )
    chunks = splitter.split_documents(raw_docs)

    if force and persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.parent.mkdir(parents=True, exist_ok=True)

    emb = embedding_from_model_id(embedding_model_id)
    Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=str(persist_directory),
        collection_name=coll,
    )
    return len(chunks)
