#!/usr/bin/env python3
"""
Generate the scaled retrieval-eval JSONL aligned to the NeurIPS-style draft PDF.

Important honesty note:
- The PDF's *aggregate* retrieval numbers may reflect a different machine/corpus snapshot.
- What we *can* do reproducibly is match chunk counts via ``configs/references.yaml`` and
  replicate curated starter queries with embedding-preserving wrappers.

Synthetic expansion remains a **scaled proxy**. Optionally reserve the tail fraction of the
budget for **negative_absent** queries: plausible domain questions tied to path patterns that
do **not** exist in any indexed chunk metadata (honesty / distractor stress).

Canonical positive schedule (500 slots): 250 easy / 150 medium / 100 hard derived from the
10-query seed JSON's difficulties (see ``_canonical_positive_plan``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_NEGATIVE_SENTINEL_PATTERN = "__benchmark_negative__/NO_CORPUS_DOCUMENT.md"
_NEGATIVE_LEXICAL_TOKEN = "__NEGATIVE_CONTROL_TOKEN_NOT_IN_INDEX__"

_NEGATIVE_QUERY_TEMPLATES = [
    (
        "What is the certified fake-factor uncertainty band for tau-ID in ATLAS Run-7 "
        "UltraLegacy validation notes referenced as ATLAS-CONF-2099-042?"
    ),
    (
        "Which CMS Analysis Note defines the loose muon isolation working point used "
        "for the fictional HE-LHC Phase-9 HH→bbbb search documented as CMS AN-9000/777?"
    ),
    (
        "State the exact dilepton mass window (GeV) mandated for the synthetic "
        "'Moonshot-III' bump hunt dataset shipped only on portal moonshot.example.invalid."
    ),
    (
        "Under the ATLAS internal memo TMP-LPX/003 (not part of this repository), "
        "what jet veto threshold applies to the tau-embedding calibration closure test?"
    ),
    (
        "Which archived Kaggle competition CSV column stores QCD jet multiplicity for "
        "the placeholder dataset `zz_benchmark_negative_only.csv`?"
    ),
    (
        "Quote the regulator b_r used in the Approximate Median Significance definition "
        "for the hypothetical FCC-hh ttbar challenge PDF `fcc_ttbar_synthetic_only.pdf` "
        "that is not indexed here."
    ),
    (
        "What sentinel numeric code denotes missing values in the fictional "
        "`DER_track_isolation_fcc` feature from dataset record CERN-OPEN-FAKE-001?"
    ),
    (
        "Which TWiki page (URL) documents the non-existent Phase-14 tau trigger menu "
        "`tau180_medium1_fake_wp` referenced only in negative-control benchmarks?"
    ),
]


def _wrap_query(base: str, *, variant: int) -> str:
    tags = [
        "",
        "[retrieval-eval] ",
        "Q: ",
        "Question: ",
        "(gold-labeled) ",
        "Please retrieve supporting passages for: ",
    ]
    suffixes = [
        "",
        " Answer using repository documents only.",
        " Prefer primary sources (PDFs) over secondary notes when both apply.",
    ]
    pre = tags[variant % len(tags)]
    suf = suffixes[(variant // len(tags)) % len(suffixes)]
    return f"{pre}{base}{suf}".strip()


def _load_seed_rows(seed_json: Path) -> list[dict]:
    data = json.loads(seed_json.read_text(encoding="utf-8"))
    rows = data.get("per_query", [])
    if len(rows) < 10:
        raise ValueError(f"Expected per_query>=10 in {seed_json}, got {len(rows)}")
    return rows


def _canonical_positive_plan(seed_rows: list[dict]) -> list[tuple[str, str]]:
    """Return ordered list of (seed_id, difficulty_out) with length 500."""
    easy_ids = [r["id"] for r in seed_rows if (r.get("difficulty") == "easy")]
    med_ids = [r["id"] for r in seed_rows if (r.get("difficulty") == "medium")]
    if len(easy_ids) != 5 or len(med_ids) != 5:
        raise ValueError("Seed set must contain exactly 5 easy + 5 medium queries for this generator.")

    plan: list[tuple[str, str]] = []
    for sid in easy_ids:
        for _ in range(50):
            plan.append((sid, "easy"))
    for sid in med_ids:
        for _ in range(30):
            plan.append((sid, "medium"))
    for sid in med_ids:
        for _ in range(20):
            plan.append((sid, "hard"))

    if len(plan) != 500:
        raise RuntimeError(f"internal canonical plan size mismatch: {len(plan)}")
    return plan


def _take_plan(plan500: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    if n <= len(plan500):
        return plan500[:n]
    out: list[tuple[str, str]] = []
    for i in range(n):
        out.append(plan500[i % len(plan500)])
    return out


def generate_queries_from_seed(
    *,
    seed_json: Path,
    n_total: int,
    negative_ratio: float = 0.0,
    min_positive: int = 10,
) -> list[dict]:
    seed_rows = _load_seed_rows(seed_json)
    seed_by_id = {r["id"]: r for r in seed_rows}

    canonical = _canonical_positive_plan(seed_rows)

    n_negative = min(
        max(0, int(round(float(negative_ratio) * int(n_total)))),
        max(0, int(n_total) - int(min_positive)),
    )
    n_positive = int(n_total) - n_negative
    if n_positive < 1:
        raise ValueError("n_total too small for requested negative_ratio")

    plan_slice = _take_plan(canonical, n_positive)

    out: list[dict] = []
    for i, (sid, diff_out) in enumerate(plan_slice):
        base = seed_by_id[sid]
        q = _wrap_query(str(base["query"]), variant=i)

        patterns = list(base.get("relevant_path_patterns") or [])
        req = list(base.get("required_terms") or [])

        if sid == "higgs_channels_methodology":
            patterns = sorted(set(patterns + ["1207.7235v2.pdf", "atlas-higgs-challenge-2014.pdf"]))
        if sid == "cms_higgs_paper":
            patterns = sorted(set(patterns + ["atlas-higgs-challenge-2014.pdf"]))
        if sid == "mass_discriminant":
            req = []

        out.append(
            {
                "id": f"{sid}__v{i:03d}",
                "query": q,
                "relevant_path_patterns": patterns,
                "required_terms": req,
                "difficulty": diff_out,
                "benchmark_segment": "positive",
                "seed_id": sid,
                "seed_run": str(seed_json),
            }
        )

    base_variant = len(out)
    for j in range(n_negative):
        tmpl = _NEGATIVE_QUERY_TEMPLATES[j % len(_NEGATIVE_QUERY_TEMPLATES)]
        variant = base_variant + j
        q = _wrap_query(tmpl, variant=variant)
        out.append(
            {
                "id": f"negative_absent__{j:03d}",
                "query": q,
                "relevant_path_patterns": [_NEGATIVE_SENTINEL_PATTERN],
                "required_terms": [_NEGATIVE_LEXICAL_TOKEN],
                "difficulty": "hard",
                "benchmark_segment": "negative_absent",
                "notes": (
                    "Negative control: gold path substring is absent from indexed chunks; "
                    "recall@k should stay ~0 while lexical/RAGAS signals reveal overconfidence."
                ),
                "seed_run": str(seed_json),
            }
        )

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate scaled-query JSONL aligned to NeurIPS draft tables.")
    p.add_argument(
        "--seed-json",
        type=Path,
        default=ROOT / "evals" / "results" / "neurips_full.json",
        help="Use per_query entries from this eval JSON as the seed gold set",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evals" / "data" / "rag_queries_500_neurips_mirror.jsonl",
        help="Where to write the generated JSONL",
    )
    p.add_argument("--n", type=int, default=500)
    p.add_argument(
        "--negative-ratio",
        type=float,
        default=0.08,
        help="Fraction of queries that are negative_absent controls (default: 0.08 ≈ 40 of 500). "
        "Use 0 to reproduce legacy all-positive suites.",
    )
    args = p.parse_args()

    queries = generate_queries_from_seed(
        seed_json=args.seed_json,
        n_total=int(args.n),
        negative_ratio=float(args.negative_ratio),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queries) + "\n", encoding="utf-8")

    seg_counts: dict[str, int] = {}
    for q in queries:
        seg = q.get("benchmark_segment", "positive")
        seg_counts[seg] = seg_counts.get(seg, 0) + 1

    print(f"Wrote {len(queries)} queries to {args.output} (seed={args.seed_json})")
    print(f"benchmark_segment counts: {seg_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
