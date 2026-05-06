from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Protocol

# OpenRouter periodically retires short slugs (404 "No endpoints found"). Try these next.
_OPENROUTER_MODEL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "anthropic/claude-3.5-sonnet": (
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-sonnet-4",
    ),
}
_fallback_warnings_emitted: set[tuple[str, str]] = set()


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


def _looks_like_openrouter_model(model: str) -> bool:
    """OpenRouter model IDs use `provider/model` (e.g. `anthropic/claude-3.5-sonnet`)."""
    m = model.strip()
    return "/" in m and not m.startswith("ft:")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _openrouter_models_to_try(primary: str) -> list[str]:
    m = primary.strip()
    out = [m]
    for alt in _OPENROUTER_MODEL_FALLBACKS.get(m, ()):
        if alt not in out:
            out.append(alt)
    return out


def _warn_openrouter_fallback(requested: str, using: str) -> None:
    key = (requested, using)
    if key in _fallback_warnings_emitted:
        return
    _fallback_warnings_emitted.add(key)
    print(
        f"[ogts OpenRouter] model {requested!r} has no endpoints; retrying with {using!r}. "
        "Update --model to pin this id explicitly.",
        file=sys.stderr,
        flush=True,
    )


@dataclass(frozen=True)
class OpenAIGenerator:
    """
    LLM code generator via the OpenAI Python SDK (`chat.completions`).

    Direct OpenAI:
      - `OPENAI_API_KEY`

    OpenRouter (same SDK, different base URL):
      - `OPENROUTER_API_KEY`
      - enable with `--openrouter` / `OGTS_USE_OPENROUTER=1`, or pass a `provider/model`
        id so OpenRouter is selected automatically.

    Optional OpenRouter attribution headers:
      - `OPENROUTER_HTTP_REFERER`, `OPENROUTER_TITLE`
    """

    model: str = "gpt-4o-mini"
    openrouter: bool = False

    def _use_openrouter(self) -> bool:
        if self.openrouter or _env_truthy("OGTS_USE_OPENROUTER"):
            return True
        return _looks_like_openrouter_model(self.model)

    def generate(self, *, prompt: str, temperature: float) -> str:
        try:
            from openai import NotFoundError, OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install openai") from exc

        use_or = self._use_openrouter()
        if use_or:
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "OpenRouter mode requires OPENROUTER_API_KEY "
                    "(set --openrouter or use a provider/model id like anthropic/claude-3.7-sonnet)."
                )
            default_headers: dict[str, str] = {}
            referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
            title = os.environ.get("OPENROUTER_TITLE", "").strip()
            if referer:
                default_headers["HTTP-Referer"] = referer
            if title:
                default_headers["X-Title"] = title
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers=default_headers or None,  # type: ignore[arg-type]
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY for OpenAIGenerator (direct OpenAI)")
            client = OpenAI(api_key=api_key)

        system = (
            "You write correct, minimal Python code. "
            "Return ONLY the Python module source code, no markdown fences."
        )
        models = _openrouter_models_to_try(self.model) if use_or else [self.model.strip()]
        last_exc: Exception | None = None
        resp = None
        for mid in models:
            try:
                resp = client.chat.completions.create(
                    model=mid,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(temperature),
                )
                if use_or and mid != self.model.strip():
                    _warn_openrouter_fallback(self.model.strip(), mid)
                break
            except NotFoundError as exc:
                last_exc = exc
                continue
        if resp is None:
            hint = (
                " Check https://openrouter.ai/models for current IDs "
                "(e.g. anthropic/claude-3.7-sonnet)."
                if use_or
                else ""
            )
            raise RuntimeError(
                f"No chat completion endpoint for models tried {models!r}.{hint}"
            ) from last_exc

        choice0 = resp.choices[0].message
        text = (getattr(choice0, "content", None) or "").strip()
        return text

