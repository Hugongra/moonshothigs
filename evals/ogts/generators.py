from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Protocol


class CodeGenerator(Protocol):
    def generate(self, *, prompt: str, temperature: float) -> str: ...


# ---------------------------------------------------------------------------
# Ground-truth solutions keyed by entrypoint name.  Used by OracleGenerator
# to supply known-correct code (upper bound), and by DummyGenerator for
# selective pass modes.
# ---------------------------------------------------------------------------

_SOLUTIONS: dict[str, str] = {
    "ams": (
        "import math\n\n"
        "def ams(s: float, b: float, br: float = 10.0) -> float:\n"
        "    s, b, br = float(s), float(b), float(br)\n"
        "    rad = 2.0 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)\n"
        "    return math.sqrt(rad) if rad > 0.0 else 0.0\n"
    ),
    "ndcg_at_k": (
        "import math\n\n"
        "def ndcg_at_k(relevances: list[int], k: int) -> float:\n"
        "    rels = relevances[:k]\n"
        "    dcg = sum(float(r) / math.log2(i + 2) for i, r in enumerate(rels))\n"
        "    ideal = sorted(rels, reverse=True)\n"
        "    idcg = sum(float(r) / math.log2(i + 2) for i, r in enumerate(ideal))\n"
        "    return 0.0 if idcg == 0.0 else dcg / idcg\n"
    ),
    "weighted_log_loss": (
        "import math\n\n"
        "def weighted_log_loss(y: list[int], p: list[float], w: list[float]) -> float:\n"
        "    eps = 1e-15\n"
        "    num = 0.0\n"
        "    den = 0.0\n"
        "    for yi, pi, wi in zip(y, p, w):\n"
        "        pi = min(1.0 - eps, max(eps, float(pi)))\n"
        "        wi = float(wi)\n"
        "        den += wi\n"
        "        num += wi * (-(yi * math.log(pi) + (1 - yi) * math.log(1 - pi)))\n"
        "    return num / den\n"
    ),
    "best_threshold": (
        "import math\n\n"
        "def best_threshold(y: list[int], p: list[float], w: list[float]) -> float:\n"
        "    grid = [i / 100.0 for i in range(1, 100)]\n"
        "    best_t = 0.5\n"
        "    best_score = -1.0\n"
        "    for t in grid:\n"
        "        s = sum(wi for yi, pi, wi in zip(y, p, w) if float(pi) >= t and int(yi) == 1)\n"
        "        b = sum(wi for yi, pi, wi in zip(y, p, w) if float(pi) >= t and int(yi) == 0)\n"
        "        br = 10.0\n"
        "        rad = 2.0 * ((s + b + br) * math.log(1.0 + s / (b + br)) - s)\n"
        "        sc = math.sqrt(rad) if rad > 0.0 else 0.0\n"
        "        if sc > best_score:\n"
        "            best_score = sc\n"
        "            best_t = t\n"
        "    return best_t\n"
    ),
}


def _inject_bug(code: str, rng: random.Random) -> str:
    """Randomly introduce one subtle bug to make a ~noisy generator."""
    bugs = [
        (r"math\.log\(", "math.log1p("),
        (r"\+ 2\)", "+ 1)"),
        (r"1\.0 - eps", "1.0 - 1e-2"),
        (r"reverse=True", "reverse=False"),
        (r"float\(wi\)", "1.0"),
    ]
    pattern, replacement = rng.choice(bugs)
    return re.sub(pattern, replacement, code, count=1)


@dataclass(frozen=True)
class DummyGenerator:
    """
    Deterministic generator for smoke tests.

    It returns intentionally incorrect code (unless the prompt contains a known marker),
    so harness logic can be validated without API keys.
    """

    mode: str = "always_fail"  # or "trivial_pass_ams"

    def generate(self, *, prompt: str, temperature: float) -> str:
        if self.mode == "trivial_pass_ams" and "Implement `ams" in prompt:
            return _SOLUTIONS["ams"]
        return "def nope():\n    return 0\n"


@dataclass(frozen=True)
class OracleGenerator:
    """
    Supplies the ground-truth solution (100% pass rate).  Used as the *upper bound*
    baseline: proves the harness works and shows what perfect oracle efficiency looks like.
    """

    def generate(self, *, prompt: str, temperature: float) -> str:
        for key, sol in _SOLUTIONS.items():
            if f"Implement `{key}" in prompt:
                return sol
        return "def nope():\n    raise NotImplementedError\n"


@dataclass(frozen=True)
class NoisyGenerator:
    """
    Simulates a flawed LLM that knows the solution family but introduces random bugs
    with probability ``bug_rate``.  Good for benchmarking OGTS vs linear-retry locally
    without an API key.
    """

    bug_rate: float = 0.6
    seed: int = 42

    def generate(self, *, prompt: str, temperature: float) -> str:
        rng = random.Random(hash((prompt, temperature, self.seed, random.random())))
        for key, sol in _SOLUTIONS.items():
            if f"Implement `{key}" in prompt:
                if rng.random() < self.bug_rate:
                    return _inject_bug(sol, rng)
                return sol
        return "def nope():\n    raise NotImplementedError\n"


@dataclass(frozen=True)
class OpenAIGenerator:
    """
    Minimal OpenAI generator.

    Requires:
      - `pip install openai`
      - `OPENAI_API_KEY`
    """

    model: str = "gpt-4o-mini"

    def generate(self, *, prompt: str, temperature: float) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install openai") from exc

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAIGenerator")

        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You write correct, minimal Python code. "
                        "Return ONLY the Python module source code, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
        )
        # responses API can return multiple output items; we join text fragments.
        out_parts: list[str] = []
        for item in resp.output:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    out_parts.append(getattr(c, "text", "") or "")
        return "\n".join(out_parts).strip()

