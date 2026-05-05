from __future__ import annotations

import importlib.util
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class LoadedModule:
    module: ModuleType
    path: Path


def write_module(code: str, *, work_dir: Path, module_name: str = "submission") -> Path:
    """
    Write a standalone Python module to `work_dir` and return its path.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{module_name}.py"
    # Ensure there is at least something to import even if model returns blank.
    code = (code or "").strip()
    if not code:
        code = "raise RuntimeError('empty submission')\n"
    path.write_text(code + "\n", encoding="utf-8")
    return path


def import_module_from_path(path: Path, *, module_name: str = "submission") -> LoadedModule:
    """
    Import a module from `path` in an isolated name.

    NOTE: This executes arbitrary code. This harness is for controlled evaluation only.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure import can resolve local imports within work_dir.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    finally:
        if sys.path and sys.path[0] == str(path.parent):
            sys.path.pop(0)
    return LoadedModule(module=module, path=path)

