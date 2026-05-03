#!/usr/bin/env python3
"""
Sync a local Overleaf bundle into an Overleaf project via its Git integration.

The pipeline produces ``output/neurips_overleaf_bundle/`` (main.tex + figures/).
This script clones / pulls the project's Overleaf git repo, copies the bundle
contents on top of it, commits, and pushes.

Usage:

    python scripts/sync_overleaf.py
    python scripts/sync_overleaf.py --config configs/overleaf.local.yaml
    python scripts/sync_overleaf.py --dry-run

Credentials:

    `git_token` in the YAML config, or environment variable ``OVERLEAF_GIT_TOKEN``
    (env var takes precedence). The project_id can also come from
    ``OVERLEAF_PROJECT_ID``.

Exit codes:

    0 success (pushed or nothing-to-do), 2 configuration error, 3 git error.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path, *, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True, env=env)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"PyYAML required: pip install pyyaml ({exc})")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class Config:
    project_id: str
    git_token: str
    clone_dir: Path
    bundle_dir: Path
    main_tex_name: str
    author_name: str
    author_email: str
    commit_message: str
    skip_if_unchanged: bool

    @property
    def remote_url(self) -> str:
        # Overleaf supports https://git:<token>@git.overleaf.com/<project>
        safe_token = urllib.parse.quote(self.git_token, safe="")
        return f"https://git:{safe_token}@git.overleaf.com/{self.project_id}"

    @property
    def masked_remote(self) -> str:
        return f"https://git:***@git.overleaf.com/{self.project_id}"


def _resolve_config(config_path: Path) -> Config:
    if not config_path.is_file():
        raise SystemExit(
            f"Overleaf config not found: {config_path}. "
            "Copy configs/overleaf.example.yaml to configs/overleaf.local.yaml and fill it in."
        )
    data = _load_yaml(config_path)

    project_id = os.environ.get("OVERLEAF_PROJECT_ID") or data.get("project_id") or ""
    git_token = os.environ.get("OVERLEAF_GIT_TOKEN") or data.get("git_token") or ""
    if not project_id:
        raise SystemExit("Overleaf project_id missing (config or OVERLEAF_PROJECT_ID).")
    if not git_token or git_token == "REPLACE_WITH_TOKEN":
        raise SystemExit("Overleaf git_token missing (config or OVERLEAF_GIT_TOKEN).")

    def _path(key: str, default: str) -> Path:
        raw = data.get(key) or default
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (ROOT / p).resolve()

    return Config(
        project_id=str(project_id),
        git_token=str(git_token),
        clone_dir=_path("clone_dir", ".overleaf_mirror"),
        bundle_dir=_path("bundle_dir", "output/neurips_overleaf_bundle"),
        main_tex_name=str(data.get("main_tex_name") or "main.tex"),
        author_name=str(data.get("author_name") or "RAG AI Scientist Pipeline"),
        author_email=str(data.get("author_email") or "pipeline@local"),
        commit_message=str(data.get("commit_message") or "auto: regenerate from pipeline ({sha} @ {ts})"),
        skip_if_unchanged=bool(data.get("skip_if_unchanged", True)),
    )


def _ensure_mirror(cfg: Config, *, verbose: bool) -> Path:
    mirror = cfg.clone_dir
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if (mirror / ".git").is_dir():
        if verbose:
            print(f"[overleaf-sync] pulling latest into {mirror}")
        _run(["git", "fetch", "origin"], mirror)
        _run(["git", "reset", "--hard", "origin/master"], mirror)
        return mirror

    if verbose:
        print(f"[overleaf-sync] cloning {cfg.masked_remote} → {mirror}")
    _run(["git", "clone", cfg.remote_url, str(mirror)], ROOT)
    return mirror


def _mirror_bundle(cfg: Config, mirror: Path, *, verbose: bool) -> list[Path]:
    """Copy the bundle into the mirror (overwriting existing main.tex + figures)."""
    import shutil

    if not cfg.bundle_dir.is_dir():
        raise SystemExit(f"Bundle dir missing: {cfg.bundle_dir}. Run the paper pipeline first.")

    changed: list[Path] = []
    for src in cfg.bundle_dir.rglob("*"):
        if src.is_dir():
            continue
        if src.name.startswith("."):
            continue
        rel = src.relative_to(cfg.bundle_dir)
        dest = mirror / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        changed.append(dest)
    if verbose:
        print(f"[overleaf-sync] copied {len(changed)} files from {cfg.bundle_dir}")
    return changed


def _git_has_changes(mirror: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(mirror), check=True, text=True, capture_output=True
    )
    return bool(result.stdout.strip())


def _sha256_prefix(path: Path, *, n: int = 10) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main() -> int:
    p = argparse.ArgumentParser(description="Push the NeurIPS Overleaf bundle to Overleaf via Git.")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "overleaf.local.yaml",
        help="YAML config (default: configs/overleaf.local.yaml)",
    )
    p.add_argument("--dry-run", action="store_true", help="Stage files but do not commit/push.")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    verbose = not args.quiet

    try:
        cfg = _resolve_config(args.config)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[overleaf-sync] config error: {exc}", file=sys.stderr)
        return 2

    try:
        mirror = _ensure_mirror(cfg, verbose=verbose)
    except subprocess.CalledProcessError as exc:
        print(f"[overleaf-sync] git error: {exc.stderr.strip() or exc}", file=sys.stderr)
        return 3

    _mirror_bundle(cfg, mirror, verbose=verbose)

    if not _git_has_changes(mirror):
        if cfg.skip_if_unchanged:
            if verbose:
                print("[overleaf-sync] no changes to push (skip_if_unchanged=true)")
            return 0

    main_tex = mirror / cfg.main_tex_name
    sha = _sha256_prefix(main_tex) if main_tex.is_file() else "nomaintex"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit_msg = cfg.commit_message.format(sha=sha, ts=ts)

    try:
        _run(["git", "add", "-A"], mirror)
        if args.dry_run:
            if verbose:
                print("[overleaf-sync] dry-run: staged files ready, skipping commit/push")
                print(_run(["git", "status"], mirror).stdout)
            return 0
        _run(
            [
                "git",
                "-c",
                f"user.name={cfg.author_name}",
                "-c",
                f"user.email={cfg.author_email}",
                "commit",
                "-m",
                commit_msg,
            ],
            mirror,
        )
        _run(["git", "push", "origin", "HEAD:master"], mirror)
    except subprocess.CalledProcessError as exc:
        print(f"[overleaf-sync] git error: {exc.stderr.strip() or exc}", file=sys.stderr)
        return 3

    if verbose:
        print(f"[overleaf-sync] pushed to {cfg.masked_remote} ({commit_msg})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
