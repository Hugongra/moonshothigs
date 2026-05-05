from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


OracleKind = Literal[
    "numeric_equal",  # exact numbers (ints/floats) after normalization/rounding
    "numeric_close",  # float tolerance
    "json_equal",  # JSON-serializable exact match
]


@dataclass(frozen=True)
class OgtsTask:
    """
    A single execution-grounded task.

    The model is prompted to generate a standalone Python module implementing `entrypoint`.
    The oracle then calls that entrypoint and verifies outputs against `oracle_payload`.
    """

    id: str
    title: str
    prompt: str
    entrypoint: str  # e.g., "solve" or "ams"
    oracle_kind: OracleKind
    oracle_payload: dict[str, Any]


@dataclass(frozen=True)
class AttemptResult:
    ok: bool
    score: float  # higher is better; 1.0 means pass for now
    status: str  # "pass" | "fail:<reason>"
    details: dict[str, Any]


@dataclass(frozen=True)
class StrategyConfig:
    name: str  # "linear_retry" | "ogts"
    # Budgets
    k: int | None = None  # linear-retry pass@k
    depth: int | None = None  # ogts depth d
    branch: int | None = None  # ogts width b
    temperature: float | None = None

