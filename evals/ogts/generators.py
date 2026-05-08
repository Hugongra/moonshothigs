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
    # ── Bioinformatics: Sequence Alignment ──
    "hamming_distance": (
        "def hamming_distance(seq1: str, seq2: str) -> int:\n"
        "    s1, s2 = seq1.upper(), seq2.upper()\n"
        "    if len(s1) != len(s2):\n"
        "        return -1\n"
        "    return sum(a != b for a, b in zip(s1, s2))\n"
    ),
    "edit_distance": (
        "def edit_distance(seq1: str, seq2: str) -> int:\n"
        "    m, n = len(seq1), len(seq2)\n"
        "    dp = list(range(n + 1))\n"
        "    for i in range(1, m + 1):\n"
        "        prev = dp[0]\n"
        "        dp[0] = i\n"
        "        for j in range(1, n + 1):\n"
        "            tmp = dp[j]\n"
        "            if seq1[i-1] == seq2[j-1]:\n"
        "                dp[j] = prev\n"
        "            else:\n"
        "                dp[j] = 1 + min(prev, dp[j], dp[j-1])\n"
        "            prev = tmp\n"
        "    return dp[n]\n"
    ),
    "needleman_wunsch_score": (
        "def needleman_wunsch_score(seq1: str, seq2: str, match: int, mismatch: int, gap: int) -> int:\n"
        "    m, n = len(seq1), len(seq2)\n"
        "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
        "    for i in range(1, m+1): dp[i][0] = dp[i-1][0] + gap\n"
        "    for j in range(1, n+1): dp[0][j] = dp[0][j-1] + gap\n"
        "    for i in range(1, m+1):\n"
        "        for j in range(1, n+1):\n"
        "            sc = match if seq1[i-1]==seq2[j-1] else mismatch\n"
        "            dp[i][j] = max(dp[i-1][j-1]+sc, dp[i-1][j]+gap, dp[i][j-1]+gap)\n"
        "    return dp[m][n]\n"
    ),
    "smith_waterman_score": (
        "def smith_waterman_score(seq1: str, seq2: str, match: int, mismatch: int, gap: int) -> int:\n"
        "    m, n = len(seq1), len(seq2)\n"
        "    best = 0\n"
        "    dp = [[0]*(n+1) for _ in range(m+1)]\n"
        "    for i in range(1, m+1):\n"
        "        for j in range(1, n+1):\n"
        "            sc = match if seq1[i-1]==seq2[j-1] else mismatch\n"
        "            v = max(0, dp[i-1][j-1]+sc, dp[i-1][j]+gap, dp[i][j-1]+gap)\n"
        "            dp[i][j] = v\n"
        "            if v > best: best = v\n"
        "    return best\n"
    ),
    "lcs_length": (
        "def lcs_length(seq1: str, seq2: str) -> int:\n"
        "    m, n = len(seq1), len(seq2)\n"
        "    dp = [0]*(n+1)\n"
        "    for i in range(1, m+1):\n"
        "        prev = 0\n"
        "        for j in range(1, n+1):\n"
        "            tmp = dp[j]\n"
        "            dp[j] = prev+1 if seq1[i-1]==seq2[j-1] else max(dp[j], dp[j-1])\n"
        "            prev = tmp\n"
        "    return dp[n]\n"
    ),
    "reverse_complement": (
        "def reverse_complement(seq: str) -> str:\n"
        "    comp = {'A':'T','T':'A','C':'G','G':'C'}\n"
        "    return ''.join(comp.get(c,c) for c in reversed(seq.upper()))\n"
    ),
    "sequence_identity": (
        "def sequence_identity(aligned1: str, aligned2: str) -> float:\n"
        "    if not aligned1 or len(aligned1) != len(aligned2): return 0.0\n"
        "    m = sum(1 for a,b in zip(aligned1,aligned2) if a==b and a!='-')\n"
        "    return m / len(aligned1)\n"
    ),
    "count_kmers": (
        "def count_kmers(seq: str, k: int) -> dict:\n"
        "    if k <= 0 or k > len(seq): return {}\n"
        "    c = {}\n"
        "    for i in range(len(seq)-k+1):\n"
        "        km = seq[i:i+k]\n"
        "        c[km] = c.get(km, 0) + 1\n"
        "    return c\n"
    ),
    "consensus_sequence": (
        "def consensus_sequence(sequences: list) -> str:\n"
        "    if not sequences: return ''\n"
        "    res = []\n"
        "    for j in range(len(sequences[0])):\n"
        "        f = {}\n"
        "        for s in sequences: f[s[j]] = f.get(s[j],0)+1\n"
        "        res.append(max(f, key=lambda c: (f[c], -ord(c))))\n"
        "    return ''.join(res)\n"
    ),
    "transition_transversion_ratio": (
        "def transition_transversion_ratio(seq1: str, seq2: str) -> float:\n"
        "    if not seq1 or len(seq1)!=len(seq2): return -1.0\n"
        "    ts = {('A','G'),('G','A'),('C','T'),('T','C')}\n"
        "    s1,s2 = seq1.upper(),seq2.upper()\n"
        "    ti=tv=0\n"
        "    for a,b in zip(s1,s2):\n"
        "        if a!=b:\n"
        "            if (a,b) in ts: ti+=1\n"
        "            else: tv+=1\n"
        "    return ti/tv if tv>0 else -1.0\n"
    ),
    # ── Bioinformatics: Genomic QA ──
    "gc_content": (
        "def gc_content(seq: str) -> float:\n"
        "    if not seq: return 0.0\n"
        "    s = seq.upper()\n"
        "    return sum(1 for c in s if c in ('G','C')) / len(s)\n"
    ),
    "translate_dna": (
        "def translate_dna(seq: str) -> str:\n"
        "    T = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',\n"
        "         'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',\n"
        "         'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',\n"
        "         'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',\n"
        "         'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',\n"
        "         'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',\n"
        "         'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',\n"
        "         'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}\n"
        "    s = seq.upper()\n"
        "    return ''.join(T.get(s[i:i+3],'?') for i in range(0,len(s)-2,3))\n"
    ),
    "mean_phred_quality": (
        "def mean_phred_quality(quality_string: str) -> float:\n"
        "    if not quality_string: return 0.0\n"
        "    return sum(ord(c)-33 for c in quality_string)/len(quality_string)\n"
    ),
    "n50_statistic": (
        "def n50_statistic(contig_lengths: list) -> int:\n"
        "    if not contig_lengths: return 0\n"
        "    s = sorted(contig_lengths, reverse=True)\n"
        "    half = sum(s)/2.0\n"
        "    c = 0\n"
        "    for ln in s:\n"
        "        c += ln\n"
        "        if c >= half: return ln\n"
        "    return 0\n"
    ),
    "find_motif_positions": (
        "def find_motif_positions(sequence: str, motif: str) -> list:\n"
        "    if not sequence or not motif: return []\n"
        "    pos, st = [], 0\n"
        "    while True:\n"
        "        i = sequence.find(motif, st)\n"
        "        if i == -1: break\n"
        "        pos.append(i); st = i + 1\n"
        "    return pos\n"
    ),
    "melting_temperature": (
        "def melting_temperature(seq: str) -> float:\n"
        "    if not seq: return 0.0\n"
        "    s = seq.upper()\n"
        "    at = sum(1 for c in s if c in ('A','T'))\n"
        "    gc = sum(1 for c in s if c in ('G','C'))\n"
        "    return float(2*at + 4*gc)\n"
    ),
    "validate_dna": (
        "def validate_dna(seq: str) -> dict:\n"
        "    v = {'A','C','G','T'}\n"
        "    inv = sorted(set(c for c in seq if c not in v))\n"
        "    return {'valid': len(inv)==0, 'length': len(seq), 'invalid_chars': inv}\n"
    ),
    "codon_frequency": (
        "def codon_frequency(seq: str) -> dict:\n"
        "    s = seq.upper()\n"
        "    if len(s) < 3: return {}\n"
        "    f = {}\n"
        "    for i in range(0, len(s)-2, 3):\n"
        "        c = s[i:i+3]; f[c] = f.get(c,0)+1\n"
        "    return f\n"
    ),
    "restriction_fragment_lengths": (
        "def restriction_fragment_lengths(seq: str, site: str) -> list:\n"
        "    if not seq: return []\n"
        "    if not site: return [len(seq)]\n"
        "    return [len(p) for p in seq.split(site)]\n"
    ),
    "molecular_weight_dna": (
        "def molecular_weight_dna(seq: str) -> float:\n"
        "    if not seq: return 0.0\n"
        "    w = {'A':331.2,'C':307.2,'G':347.2,'T':322.2}\n"
        "    s = seq.upper()\n"
        "    t = sum(w.get(c,0.0) for c in s)\n"
        "    if len(s) > 1: t -= (len(s)-1)*18.02\n"
        "    return t\n"
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

    ``seed``: when set, forwarded to ``chat.completions.create(seed=...)`` for reproducible
    sampling (provider/model must support the parameter).
    """

    model: str = "gpt-4o-mini"
    openrouter: bool = False
    seed: int | None = None

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
                raise RuntimeError(
                    "Missing OPENAI_API_KEY for OpenAIGenerator (direct OpenAI). "
                    "Export OPENAI_API_KEY or put it in a .env file at the repository root "
                    "(loaded automatically by evals/ogts/run_ogsr_eval.py)."
                )
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
                create_kwargs: dict = {
                    "model": mid,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": float(temperature),
                }
                if self.seed is not None:
                    create_kwargs["seed"] = int(self.seed)
                resp = client.chat.completions.create(**create_kwargs)
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

