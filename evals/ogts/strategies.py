from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .code_runner import import_module_from_path, write_module
from .generators import CodeGenerator
from .oracles import oracle_eval
from .types import AttemptResult, OgtsTask


@dataclass(frozen=True)
class RunStats:
    oracle_calls: int
    attempts: int
    passed: bool
    best_score: float
    best_status: str


def _eval_code(task: OgtsTask, code: str, *, work_dir: Path) -> AttemptResult:
    path = write_module(code, work_dir=work_dir, module_name="submission")
    lm = import_module_from_path(path, module_name="submission")
    return oracle_eval(task, lm.module)


def linear_retry(
    *,
    task: OgtsTask,
    gen: CodeGenerator,
    k: int,
    temperature: float,
) -> tuple[AttemptResult | None, RunStats]:
    """
    pass@k baseline: sample k independent modules; return first pass if any.
    """
    oracle_calls = 0
    best: AttemptResult | None = None
    with tempfile.TemporaryDirectory(prefix=f"ogts_{task.id}_linear_") as td:
        base = Path(td)
        for i in range(int(k)):
            code = gen.generate(prompt=task.prompt, temperature=float(temperature))
            try:
                res = _eval_code(task, code, work_dir=base / f"attempt_{i+1:02d}")
            except Exception as exc:
                res = AttemptResult(ok=False, score=0.0, status=f"fail:exec:{type(exc).__name__}", details={"msg": str(exc)})
            oracle_calls += 1
            if best is None or res.score > best.score:
                best = res
            if res.ok:
                return res, RunStats(
                    oracle_calls=oracle_calls,
                    attempts=i + 1,
                    passed=True,
                    best_score=float(res.score),
                    best_status=str(res.status),
                )
    return best, RunStats(
        oracle_calls=oracle_calls,
        attempts=int(k),
        passed=bool(best.ok) if best else False,
        best_score=float(best.score) if best else 0.0,
        best_status=str(best.status) if best else "fail:no_attempts",
    )


def _oracle_feedback_snippet(res: AttemptResult, *, max_cases: int = 2, max_chars: int = 2800) -> str:
    """Compact JSON for iterative repair prompts (CodeT-style execution feedback)."""
    fails = []
    if isinstance(res.details, dict):
        raw = res.details.get("failures")
        if isinstance(raw, list):
            fails = raw[:max_cases]
    blob = json.dumps(fails, ensure_ascii=False)
    if len(blob) > max_chars:
        return blob[:max_chars] + "…"
    return blob


def iterative_repair(
    *,
    task: OgtsTask,
    gen: CodeGenerator,
    k: int,
    temperature: float,
) -> tuple[AttemptResult | None, RunStats]:
    r"""
    Sequential self-debugging baseline: up to ``k`` attempts; each failure feeds oracle
    mismatch details into the next prompt (cf. test-driven repair / CodeT).

    Same oracle-call budget cap as ``linear_retry`` for comparable cost curves.
    """
    oracle_calls = 0
    best: AttemptResult | None = None
    ctx = task.prompt

    with tempfile.TemporaryDirectory(prefix=f"ogts_{task.id}_repair_") as td:
        base = Path(td)
        for i in range(int(k)):
            code = gen.generate(prompt=ctx, temperature=float(temperature))
            try:
                res = _eval_code(task, code, work_dir=base / f"repair_{i+1:02d}")
            except Exception as exc:
                res = AttemptResult(ok=False, score=0.0, status=f"fail:exec:{type(exc).__name__}", details={"msg": str(exc)})
            oracle_calls += 1
            if best is None or res.score > best.score:
                best = res
            if res.ok:
                return res, RunStats(
                    oracle_calls=oracle_calls,
                    attempts=i + 1,
                    passed=True,
                    best_score=float(res.score),
                    best_status=str(res.status),
                )
            fb = _oracle_feedback_snippet(res)
            ctx = (
                task.prompt
                + "\n\nPrevious attempt failed automated checks.\n"
                + f"Oracle status: {res.status}\n"
                + "Mismatch summary (JSON):\n"
                + fb
                + "\nWrite a corrected module. Return ONLY Python code.\n"
            )

    return best, RunStats(
        oracle_calls=oracle_calls,
        attempts=int(k),
        passed=bool(best.ok) if best else False,
        best_score=float(best.score) if best else 0.0,
        best_status=str(best.status) if best else "fail:no_attempts",
    )


def ogts(
    *,
    task: OgtsTask,
    gen: CodeGenerator,
    depth: int,
    branch: int,
    temperature: float,
) -> tuple[AttemptResult | None, RunStats]:
    """
    Oracle-Guided Tree Search (OGTS):
      - For each depth step, sample `branch` candidates.
      - Run oracle on every candidate.
      - If any pass, return the first passing candidate.
      - Otherwise select the highest-scoring survivor and append a short failure context
        (here: status string) to the prompt for the next depth.

    Note: we keep the “context” simple (status + a fixed instruction) to stay deterministic
    and avoid leaking oracle internals.
    """
    oracle_calls = 0
    best: AttemptResult | None = None
    ctx = task.prompt

    with tempfile.TemporaryDirectory(prefix=f"ogts_{task.id}_tree_") as td:
        base = Path(td)
        for d in range(int(depth)):
            survivors: list[tuple[AttemptResult, str]] = []
            for b in range(int(branch)):
                code = gen.generate(prompt=ctx, temperature=float(temperature))
                try:
                    res = _eval_code(task, code, work_dir=base / f"d{d+1:02d}_b{b+1:02d}")
                except Exception as exc:
                    res = AttemptResult(ok=False, score=0.0, status=f"fail:exec:{type(exc).__name__}", details={"msg": str(exc)})
                oracle_calls += 1
                if best is None or res.score > best.score:
                    best = res
                if res.ok:
                    return res, RunStats(
                        oracle_calls=oracle_calls,
                        attempts=(d * int(branch)) + (b + 1),
                        passed=True,
                        best_score=float(res.score),
                        best_status=str(res.status),
                    )
                survivors.append((res, code))

            # No pass at this depth: pick best-scoring attempt and refine prompt.
            survivors.sort(key=lambda t: t[0].score, reverse=True)
            top_res, top_code = survivors[0]
            # Minimal refinement: keep code and add a brief instruction.
            ctx = (
                task.prompt
                + "\n\nPrevious attempt did not pass the oracle.\n"
                + f"Status: {top_res.status}\n"
                + "Write a corrected module. Return ONLY Python code.\n"
            )

    return best, RunStats(
        oracle_calls=oracle_calls,
        attempts=int(depth) * int(branch),
        passed=bool(best.ok) if best else False,
        best_score=float(best.score) if best else 0.0,
        best_status=str(best.status) if best else "fail:no_attempts",
    )

