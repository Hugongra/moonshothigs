from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import OgtsTask


def _rnd(seed: int) -> random.Random:
    return random.Random(seed)


def _task_prompt_header(title: str, entrypoint: str, signature: str) -> str:
    return (
        f"You are writing a standalone Python module.\n"
        f"Implement `{entrypoint}{signature}`.\n\n"
        f"Task: {title}\n"
        f"Requirements:\n"
        f"- Use only the Python standard library.\n"
        f"- Do not read/write files.\n"
        f"- Deterministic output.\n"
        f"- Keep it short and correct.\n"
    )


def make_task_ams(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = []
    for _ in range(6):
        s = r.uniform(0.0, 80.0)
        b = r.uniform(0.0, 200.0)
        br = 10.0
        # Closed-form from challenge (as in many writeups):
        # AMS = sqrt( 2 * ((s + b + br) * ln(1 + s/(b+br)) - s) )
        import math

        ams = math.sqrt(max(0.0, 2.0 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)))
        cases.append({"args": [s, b], "kwargs": {"br": br}, "expected": ams})
    prompt = _task_prompt_header(
        "Compute Approximate Median Significance (AMS) used in the ATLAS Higgs ML challenge.",
        "ams",
        "(s: float, b: float, br: float = 10.0) -> float",
    )
    prompt += (
        "\nReturn the scalar AMS. Use:\n"
        "  AMS = sqrt( 2 * ((s+b+br) * ln(1 + s/(b+br)) - s) )\n"
        "and clamp the radicand at 0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="AMS closed-form",
        prompt=prompt,
        entrypoint="ams",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


def make_task_ndcg(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = []
    import math

    def ndcg(rels: list[int], k: int) -> float:
        rels = rels[:k]
        dcg = 0.0
        for i, rel in enumerate(rels, start=1):
            dcg += float(rel) / math.log2(i + 1)
        ideal = sorted(rels, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal, start=1):
            idcg += float(rel) / math.log2(i + 1)
        return 0.0 if idcg == 0.0 else dcg / idcg

    for _ in range(6):
        k = r.choice([3, 5, 7])
        rels = [r.choice([0, 1]) for _ in range(r.choice([k, k + 2, k + 4]))]
        cases.append({"args": [rels, k], "kwargs": {}, "expected": ndcg(rels, k)})

    prompt = _task_prompt_header(
        "Compute nDCG@k for binary relevance with log2 discount.",
        "ndcg_at_k",
        "(relevances: list[int], k: int) -> float",
    )
    prompt += (
        "\nDefine DCG = sum_i rel_i / log2(i+1) for i=1..k (1-based rank).\n"
        "Define IDCG by sorting the first k relevances descending. Return 0 if IDCG=0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="nDCG@k (binary)",
        prompt=prompt,
        entrypoint="ndcg_at_k",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


def make_task_weighted_log_loss(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    import math

    def wlogloss(y: list[int], p: list[float], w: list[float]) -> float:
        eps = 1e-15
        num = 0.0
        den = 0.0
        for yi, pi, wi in zip(y, p, w):
            pi = min(1.0 - eps, max(eps, float(pi)))
            wi = float(wi)
            den += wi
            num += wi * (-(yi * math.log(pi) + (1 - yi) * math.log(1 - pi)))
        return num / den

    cases = []
    for _ in range(6):
        n = r.choice([5, 8, 10])
        y = [r.choice([0, 1]) for _ in range(n)]
        p = [r.random() for _ in range(n)]
        w = [r.uniform(0.5, 3.0) for _ in range(n)]
        cases.append({"args": [y, p, w], "kwargs": {}, "expected": wlogloss(y, p, w)})

    prompt = _task_prompt_header(
        "Compute weighted binary log-loss for labels and probabilities.",
        "weighted_log_loss",
        "(y: list[int], p: list[float], w: list[float]) -> float",
    )
    prompt += "\nClip probabilities to [1e-15, 1-1e-15]. Return sum(w*loss)/sum(w).\n"
    return OgtsTask(
        id=task_id,
        title="Weighted log-loss",
        prompt=prompt,
        entrypoint="weighted_log_loss",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


def make_task_threshold_scan(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    import math

    # Simple AMS-like score with br=10 using predicted probabilities and weights:
    def ams_from_pred(y: list[int], p: list[float], w: list[float], t: float) -> float:
        s = 0.0
        b = 0.0
        for yi, pi, wi in zip(y, p, w):
            if float(pi) >= t:
                if int(yi) == 1:
                    s += float(wi)
                else:
                    b += float(wi)
        br = 10.0
        return math.sqrt(max(0.0, 2.0 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)))

    def best_t(y: list[int], p: list[float], w: list[float]) -> float:
        grid = [i / 100.0 for i in range(1, 100)]
        best = None
        best_score = -1.0
        for t in grid:
            sc = ams_from_pred(y, p, w, t)
            if sc > best_score:
                best_score = sc
                best = t
        return float(best if best is not None else 0.5)

    cases = []
    for _ in range(6):
        n = r.choice([30, 40])
        y = [r.choice([0, 1]) for _ in range(n)]
        p = [r.random() for _ in range(n)]
        w = [r.uniform(0.5, 2.0) for _ in range(n)]
        cases.append({"args": [y, p, w], "kwargs": {}, "expected": best_t(y, p, w)})

    prompt = _task_prompt_header(
        "Find the best probability threshold in [0.01..0.99] (step=0.01) maximizing AMS.",
        "best_threshold",
        "(y: list[int], p: list[float], w: list[float]) -> float",
    )
    prompt += (
        "\nCompute AMS with br=10 using s=sum(w for positives with p>=t) and "
        "b=sum(w for negatives with p>=t). Search t in {0.01,0.02,...,0.99}. "
        "Return the best t (if ties, return the smallest t).\n"
    )
    return OgtsTask(
        id=task_id,
        title="Threshold scan for AMS",
        prompt=prompt,
        entrypoint="best_threshold",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-12, "cases": cases},
    )


def make_ogts_tasks_50(seed: int = 1337) -> list[OgtsTask]:
    """
    Build a deterministic suite of 50 execution-grounded micro-tasks.

    These tasks are small by design: they are intended to test whether an agent can
    synthesize correct code and pass an oracle, not longform engineering.
    """
    tasks: list[OgtsTask] = []
    # 4 families × 12 = 48, plus 2 extra AMS to make 50.
    makers = [make_task_ams, make_task_weighted_log_loss, make_task_ndcg, make_task_threshold_scan]
    r = _rnd(seed)
    i = 0
    for maker in makers:
        for _ in range(12):
            i += 1
            tasks.append(maker(f"ogts_{i:03d}", r.randint(0, 10_000_000)))
    for _ in range(2):
        i += 1
        tasks.append(make_task_ams(f"ogts_{i:03d}", r.randint(0, 10_000_000)))
    assert len(tasks) == 50
    return tasks


def write_tasks_jsonl(tasks: list[OgtsTask], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    return path.resolve()


def load_tasks_jsonl(path: Path) -> list[OgtsTask]:
    tasks: list[OgtsTask] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(OgtsTask(**obj))
    return tasks

