"""Load optional ``.env`` files into ``os.environ`` (minimal dotenv; stdlib only).

Rules:
- Reads ``<repo_root>/.env`` first, then ``<cwd>/.env`` (deduplicated by resolved path).
- Skips UTF-8 BOM.
- Sets a key only when it is **missing** or its current value is **blank** (whitespace-only).
  So an empty ``export OPENAI_API_KEY=`` in the shell can still be replaced from ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_value_blank(key: str) -> bool:
    return key not in os.environ or not str(os.environ.get(key, "")).strip()


def _load_dotenv_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if text.startswith("\ufeff"):
        text = text[1:]
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lstrip("\ufeff")
        if not key:
            continue
        val = value.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if not val:
            continue
        if _env_value_blank(key):
            os.environ[key] = val


def load_repo_dotenv(repo_root: Path, *, filename: str = ".env") -> None:
    candidates = [repo_root / filename, Path.cwd() / filename]
    done_resolved: set[Path] = set()
    for path in candidates:
        try:
            rp = path.resolve()
        except OSError:
            continue
        if rp in done_resolved or not path.is_file():
            continue
        done_resolved.add(rp)
        _load_dotenv_file(path)


def dotenv_paths_tried(repo_root: Path, *, filename: str = ".env") -> tuple[Path, Path]:
    """Return the two standard locations (for error messages)."""
    return repo_root / filename, Path.cwd() / filename
