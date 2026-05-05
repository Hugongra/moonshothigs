from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any, Callable

from .types import AttemptResult, OgtsTask, OracleKind


def _safe_float(x: Any) -> float:
    try:
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception as exc:
        raise TypeError(f"not a float: {x!r}") from exc


def _normalize_number(x: Any, *, digits: int | None) -> float:
    v = _safe_float(x)
    if digits is None:
        return v
    return float(round(v, digits))


def _distance_numeric(a: float, b: float) -> float:
    # A bounded score in [0, 1] where 1 is perfect.
    denom = max(1.0, abs(a), abs(b))
    rel = abs(a - b) / denom
    return float(max(0.0, 1.0 - rel))


def oracle_eval(task: OgtsTask, module: Any) -> AttemptResult:
    """
    Evaluate a generated module against the task oracle.

    The module must define the task entrypoint (a function). The oracle payload carries
    all test cases and expected outputs.
    """
    fn = getattr(module, task.entrypoint, None)
    if fn is None or not callable(fn):
        return AttemptResult(
            ok=False,
            score=0.0,
            status=f"fail:missing_entrypoint:{task.entrypoint}",
            details={"entrypoint": task.entrypoint},
        )

    kind: OracleKind = task.oracle_kind
    payload = task.oracle_payload or {}
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        return AttemptResult(ok=False, score=0.0, status="fail:oracle_payload_empty", details={})

    digits = payload.get("round_digits")
    tol = payload.get("tol")

    scores: list[float] = []
    failures: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        args = case.get("args", [])
        kwargs = case.get("kwargs", {})
        expected = case.get("expected")
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            failures.append({"case": i, "error": f"exception:{type(exc).__name__}", "msg": str(exc)})
            scores.append(0.0)
            continue

        try:
            if kind == "json_equal":
                if out == expected:
                    scores.append(1.0)
                else:
                    failures.append({"case": i, "expected": expected, "got": out})
                    scores.append(0.0)
            elif kind == "numeric_equal":
                a = _normalize_number(out, digits=int(digits) if digits is not None else None)
                b = _normalize_number(expected, digits=int(digits) if digits is not None else None)
                if a == b:
                    scores.append(1.0)
                else:
                    failures.append({"case": i, "expected": b, "got": a})
                    scores.append(_distance_numeric(a, b))
            elif kind == "numeric_close":
                a = _safe_float(out)
                b = _safe_float(expected)
                t = float(tol) if tol is not None else 1e-6
                if math.isfinite(a) and math.isfinite(b) and abs(a - b) <= t:
                    scores.append(1.0)
                else:
                    failures.append({"case": i, "expected": b, "got": a, "tol": t})
                    scores.append(_distance_numeric(a, b))
            else:
                return AttemptResult(ok=False, score=0.0, status=f"fail:unknown_oracle_kind:{kind}", details={})
        except Exception as exc:
            failures.append({"case": i, "error": f"oracle_error:{type(exc).__name__}", "msg": str(exc)})
            scores.append(0.0)

    mean_score = float(sum(scores) / max(1, len(scores)))
    ok = all(s >= 1.0 for s in scores)
    return AttemptResult(
        ok=ok,
        score=1.0 if ok else mean_score,
        status="pass" if ok else "fail:oracle_mismatch",
        details={"failures": failures, "scores": scores},
    )


def summarize_attempt_for_json(res: AttemptResult) -> dict[str, Any]:
    return {
        "ok": bool(res.ok),
        "score": float(res.score),
        "status": str(res.status),
        "details": res.details,
    }

