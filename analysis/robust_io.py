"""Robust filesystem helpers for paper generation.

Goals:

* Atomic writes (tempfile + ``os.replace``) so partial files never surface.
* Post-write verification (size, recent mtime).
* Template validation (no unrendered Jinja markers remain).
* Summary banner helpers (size + mtime + sha256 for logs).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


UNRENDERED_PATTERNS = (
    re.compile(r"<<\s*\w"),      # unreplaced `<< name >>` (this project's variable delimiter)
    re.compile(r"{%\s*\w+\s"),   # leftover `{% for ... %}` style blocks
)


class RenderValidationError(RuntimeError):
    """Raised when a rendered artifact fails robustness checks."""


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write ``content`` to ``path`` via a temp file in the same directory + rename.

    Fails loudly if the directory cannot be written (no silent success).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise
    return path


def atomic_copy(src: Path, dest: Path) -> Path:
    """Copy ``src`` → ``dest`` via a temp file in the destination dir."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.",
        suffix=".tmp",
        dir=str(dest.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise
    return dest


def validate_rendered_tex(path: Path, *, min_bytes: int = 400) -> None:
    """Raise ``RenderValidationError`` if TeX looks empty or un-rendered."""
    if not path.is_file():
        raise RenderValidationError(f"expected rendered TeX at {path}, but file is missing")
    size = path.stat().st_size
    if size < min_bytes:
        raise RenderValidationError(
            f"{path} is suspiciously small ({size} bytes < {min_bytes}); re-check template inputs"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in UNRENDERED_PATTERNS:
        match = pattern.search(text)
        if match:
            snippet = text[max(0, match.start() - 30): match.end() + 60].replace("\n", " ")
            raise RenderValidationError(
                f"{path.name} contains unrendered template markers near: '…{snippet}…'"
            )


def assert_recent(path: Path, started_at: float, *, slack_seconds: float = 2.0) -> None:
    """Confirm ``path`` mtime is >= ``started_at - slack_seconds`` (catches blocked writes)."""
    mtime = path.stat().st_mtime
    if mtime < started_at - slack_seconds:
        raise RenderValidationError(
            f"{path} is stale (mtime={mtime:.0f} < pipeline_start={started_at:.0f}); "
            "write may have been blocked (e.g. sandbox, permissions). Inspect logs."
        )


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ArtifactSummary:
    path: Path
    size: int
    mtime: float
    sha256: str

    @property
    def mtime_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.mtime))

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "size": self.size,
            "mtime": self.mtime,
            "mtime_iso": self.mtime_iso,
            "sha256": self.sha256,
        }


def summarize(path: Path) -> ArtifactSummary:
    st = path.stat()
    return ArtifactSummary(path=path, size=st.st_size, mtime=st.st_mtime, sha256=sha256_hex(path))


def banner(title: str, rows: list[ArtifactSummary]) -> str:
    lines = [f"[{title}]"]
    for row in rows:
        lines.append(
            f"  {row.path}  ({row.size:,} B, {row.mtime_iso}, sha256={row.sha256[:12]}…)"
        )
    return "\n".join(lines)
