"""
Bioinformatics task generators for the OGTS evaluation suite.

Two families, 10 tasks each (20 total):

  Sequence Alignment
  ~~~~~~~~~~~~~~~~~~
  hamming_distance, edit_distance, needleman_wunsch_score,
  smith_waterman_score, lcs_length, reverse_complement,
  sequence_identity, count_kmers, consensus_sequence,
  transition_transversion_ratio

  Genomic QA
  ~~~~~~~~~~
  gc_content, translate_dna, mean_phred_quality, n50_statistic,
  find_motif_positions, melting_temperature, validate_dna,
  codon_frequency, restriction_fragment_lengths, molecular_weight_dna

Usage::

    from evals.ogts.bioinformatics_tasks import make_bioinformatics_tasks_20
    from evals.ogts.task_suite import write_tasks_jsonl
    from pathlib import Path

    tasks = make_bioinformatics_tasks_20(seed=2024)
    write_tasks_jsonl(tasks, Path("evals/ogts/data/bio_20_tasks.jsonl"))
"""
from __future__ import annotations

import random
from typing import Any

from .types import OgtsTask

# ──────────────────────────── constants ────────────────────────────

FAMILY_ALIGNMENT = "Sequence Alignment"
FAMILY_GENOMIC_QA = "Genomic QA"

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_NUC_WEIGHTS = {"A": 331.2, "C": 307.2, "G": 347.2, "T": 322.2}

_CODON_TABLE_PROMPT = (
    "CODON_TABLE = {\n"
    "    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',\n"
    "    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',\n"
    "    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',\n"
    "    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',\n"
    "    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',\n"
    "    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',\n"
    "    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',\n"
    "    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',\n"
    "    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',\n"
    "    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',\n"
    "    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',\n"
    "    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',\n"
    "    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',\n"
    "    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',\n"
    "    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',\n"
    "    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',\n"
    "}"
)

# ──────────────────────────── utilities ────────────────────────────


def _rnd(seed: int) -> random.Random:
    return random.Random(seed)


def _random_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def _random_aligned_pair(rng: random.Random, length: int) -> tuple[str, str]:
    """Generate two aligned sequences (may contain '-') of the same length."""
    a1: list[str] = []
    a2: list[str] = []
    for _ in range(length):
        r = rng.random()
        if r < 0.10:
            a1.append("-")
            a2.append(rng.choice("ACGT"))
        elif r < 0.20:
            a1.append(rng.choice("ACGT"))
            a2.append("-")
        else:
            b1 = rng.choice("ACGT")
            b2 = rng.choice("ACGT") if rng.random() < 0.3 else b1
            a1.append(b1)
            a2.append(b2)
    return "".join(a1), "".join(a2)


def _random_quality_string(rng: random.Random, length: int) -> str:
    """Random Phred+33 quality string (Q 0-40)."""
    return "".join(chr(rng.randint(33, 73)) for _ in range(length))


def _task_prompt_header(title: str, entrypoint: str, signature: str, family: str) -> str:
    return (
        f"You are writing a standalone Python module.\n"
        f"Implement `{entrypoint}{signature}`.\n\n"
        f"Domain: Bioinformatics \u2014 {family}\n"
        f"Task: {title}\n"
        f"Requirements:\n"
        f"- Use only the Python standard library.\n"
        f"- Do not read/write files.\n"
        f"- Deterministic output.\n"
        f"- Keep it short and correct.\n"
    )


def _case(args: list[Any], expected: Any) -> dict[str, Any]:
    return {"args": args, "kwargs": {}, "expected": expected}


# ──────────────────── reference implementations ────────────────────
# These compute oracle-expected values.  They are NOT exposed to the LLM
# under test; only their *outputs* appear in oracle_payload.


def _ref_hamming(seq1: str, seq2: str) -> int:
    s1, s2 = seq1.upper(), seq2.upper()
    if len(s1) != len(s2):
        return -1
    return sum(a != b for a, b in zip(s1, s2))


def _ref_edit_distance(seq1: str, seq2: str) -> int:
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def _ref_nw_score(seq1: str, seq2: str, match: int, mismatch: int, gap: int) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + gap
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + gap
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sc = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + sc,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )
    return dp[m][n]


def _ref_sw_score(seq1: str, seq2: str, match: int, mismatch: int, gap: int) -> int:
    m, n = len(seq1), len(seq2)
    best = 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sc = match if seq1[i - 1] == seq2[j - 1] else mismatch
            val = max(0, dp[i - 1][j - 1] + sc, dp[i - 1][j] + gap, dp[i][j - 1] + gap)
            dp[i][j] = val
            if val > best:
                best = val
    return best


def _ref_lcs_length(seq1: str, seq2: str) -> int:
    m, n = len(seq1), len(seq2)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def _ref_reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT.get(c, c) for c in reversed(seq.upper()))


def _ref_sequence_identity(a1: str, a2: str) -> float:
    if not a1 or len(a1) != len(a2):
        return 0.0
    matches = sum(1 for a, b in zip(a1, a2) if a == b and a != "-")
    return matches / len(a1)


def _ref_count_kmers(seq: str, k: int) -> dict[str, int]:
    if k <= 0 or k > len(seq):
        return {}
    counts: dict[str, int] = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def _ref_consensus(sequences: list[str]) -> str:
    if not sequences:
        return ""
    result: list[str] = []
    for j in range(len(sequences[0])):
        freq: dict[str, int] = {}
        for seq in sequences:
            c = seq[j]
            freq[c] = freq.get(c, 0) + 1
        # Ties broken by alphabetically first (smallest ord)
        best = max(freq, key=lambda ch: (freq[ch], -ord(ch)))
        result.append(best)
    return "".join(result)


def _ref_ti_tv(seq1: str, seq2: str) -> float:
    if not seq1 or len(seq1) != len(seq2):
        return -1.0
    transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    s1, s2 = seq1.upper(), seq2.upper()
    ti = tv = 0
    for a, b in zip(s1, s2):
        if a != b:
            if (a, b) in transitions:
                ti += 1
            else:
                tv += 1
    return ti / tv if tv > 0 else -1.0


def _ref_gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    gc = sum(1 for c in s if c in ("G", "C"))
    return gc / len(s)


def _ref_translate(seq: str) -> str:
    s = seq.upper()
    result: list[str] = []
    for i in range(0, len(s) - 2, 3):
        codon = s[i : i + 3]
        result.append(_CODON_TABLE.get(codon, "?"))
    return "".join(result)


def _ref_mean_phred(quality: str) -> float:
    if not quality:
        return 0.0
    return sum(ord(c) - 33 for c in quality) / len(quality)


def _ref_n50(lengths: list[int]) -> int:
    if not lengths:
        return 0
    s = sorted(lengths, reverse=True)
    total = sum(s)
    half = total / 2.0
    cumulative = 0
    for ln in s:
        cumulative += ln
        if cumulative >= half:
            return ln
    return 0


def _ref_find_motif(seq: str, motif: str) -> list[int]:
    if not seq or not motif:
        return []
    positions: list[int] = []
    start = 0
    while True:
        pos = seq.find(motif, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def _ref_melting_temp(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    at = sum(1 for c in s if c in ("A", "T"))
    gc = sum(1 for c in s if c in ("G", "C"))
    return float(2 * at + 4 * gc)


def _ref_validate_dna(seq: str) -> dict[str, Any]:
    valid_chars = {"A", "C", "G", "T"}
    invalid = sorted(set(c for c in seq if c not in valid_chars))
    return {"valid": len(invalid) == 0, "length": len(seq), "invalid_chars": invalid}


def _ref_codon_freq(seq: str) -> dict[str, int]:
    s = seq.upper()
    if len(s) < 3:
        return {}
    freq: dict[str, int] = {}
    for i in range(0, len(s) - 2, 3):
        codon = s[i : i + 3]
        freq[codon] = freq.get(codon, 0) + 1
    return freq


def _ref_restriction_fragments(seq: str, site: str) -> list[int]:
    if not seq:
        return []
    if not site:
        return [len(seq)]
    return [len(p) for p in seq.split(site)]


def _ref_molecular_weight(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    total = sum(_NUC_WEIGHTS.get(c, 0.0) for c in s)
    if len(s) > 1:
        total -= (len(s) - 1) * 18.02
    return total


# ═══════════════════════════════════════════════════════════════════
#  Sequence Alignment family  (10 tasks)
# ═══════════════════════════════════════════════════════════════════


def make_task_hamming_distance(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", ""], 0),
        _case(["ACGT", "AC"], -1),
    ]
    for _ in range(4):
        n = r.randint(5, 20)
        s1 = _random_dna(r, n)
        s2 = _random_dna(r, n)
        cases.append(_case([s1, s2], _ref_hamming(s1, s2)))

    prompt = _task_prompt_header(
        "Hamming Distance between DNA sequences",
        "hamming_distance",
        "(seq1: str, seq2: str) -> int",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCompute the Hamming distance (number of mismatching positions) between two "
        "DNA sequences. Compare case-insensitively (convert to uppercase).\n"
        "Edge cases:\n"
        "- Both sequences empty \u2192 return 0.\n"
        "- Different lengths \u2192 return -1.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Hamming Distance",
        prompt=prompt,
        entrypoint="hamming_distance",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_edit_distance(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", ""], 0),
        _case(["", "ACGT"], 4),
        _case(["ACGT", ""], 4),
    ]
    for _ in range(4):
        n1, n2 = r.randint(3, 12), r.randint(3, 12)
        s1, s2 = _random_dna(r, n1), _random_dna(r, n2)
        cases.append(_case([s1, s2], _ref_edit_distance(s1, s2)))

    prompt = _task_prompt_header(
        "Edit (Levenshtein) Distance between DNA sequences",
        "edit_distance",
        "(seq1: str, seq2: str) -> int",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCompute the Levenshtein edit distance between two DNA sequences.\n"
        "Allowed operations: insert, delete, substitute (each costs 1).\n"
        "Edge cases:\n"
        "- Empty string vs non-empty \u2192 return the length of the non-empty string.\n"
        "- Both empty \u2192 return 0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Edit Distance",
        prompt=prompt,
        entrypoint="edit_distance",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_needleman_wunsch(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", "", 1, -1, -2], _ref_nw_score("", "", 1, -1, -2)),
        _case(["", "ACG", 1, -1, -2], _ref_nw_score("", "ACG", 1, -1, -2)),
    ]
    for _ in range(4):
        n1, n2 = r.randint(4, 10), r.randint(4, 10)
        s1, s2 = _random_dna(r, n1), _random_dna(r, n2)
        match = r.choice([1, 2])
        mismatch = r.choice([-1, -2])
        gap = r.choice([-1, -2])
        cases.append(_case([s1, s2, match, mismatch, gap], _ref_nw_score(s1, s2, match, mismatch, gap)))

    prompt = _task_prompt_header(
        "Needleman-Wunsch Global Alignment Score",
        "needleman_wunsch_score",
        "(seq1: str, seq2: str, match: int, mismatch: int, gap: int) -> int",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCompute the optimal global alignment score using the Needleman-Wunsch algorithm.\n"
        "Parameters:\n"
        "- match: score added when characters are equal (positive).\n"
        "- mismatch: score added when characters differ (negative).\n"
        "- gap: score added for each gap (negative).\n"
        "Recurrence:\n"
        "  dp[i][j] = max(\n"
        "      dp[i-1][j-1] + (match if seq1[i-1]==seq2[j-1] else mismatch),\n"
        "      dp[i-1][j] + gap,\n"
        "      dp[i][j-1] + gap\n"
        "  )\n"
        "Initialise dp[i][0] = i*gap, dp[0][j] = j*gap.\n"
        "Edge cases:\n"
        "- Empty sequence(s) \u2192 alignment uses only gaps.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Needleman-Wunsch Score",
        prompt=prompt,
        entrypoint="needleman_wunsch_score",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_smith_waterman(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", "", 2, -1, -1], 0),
        _case(["", "ACGT", 2, -1, -1], 0),
    ]
    for _ in range(4):
        n1, n2 = r.randint(5, 12), r.randint(5, 12)
        s1, s2 = _random_dna(r, n1), _random_dna(r, n2)
        match = r.choice([1, 2, 3])
        mismatch = r.choice([-1, -2])
        gap = r.choice([-1, -2])
        cases.append(_case([s1, s2, match, mismatch, gap], _ref_sw_score(s1, s2, match, mismatch, gap)))

    prompt = _task_prompt_header(
        "Smith-Waterman Local Alignment Score",
        "smith_waterman_score",
        "(seq1: str, seq2: str, match: int, mismatch: int, gap: int) -> int",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCompute the optimal local alignment score using the Smith-Waterman algorithm.\n"
        "Same recurrence as Needleman-Wunsch but clamp each cell to at least 0:\n"
        "  dp[i][j] = max(0,\n"
        "      dp[i-1][j-1] + (match if seq1[i-1]==seq2[j-1] else mismatch),\n"
        "      dp[i-1][j] + gap,\n"
        "      dp[i][j-1] + gap)\n"
        "Return the maximum value in the entire DP table.\n"
        "Edge cases:\n"
        "- Any empty sequence \u2192 return 0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Smith-Waterman Score",
        prompt=prompt,
        entrypoint="smith_waterman_score",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_lcs_length(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", ""], 0),
        _case(["", "ACGT"], 0),
        _case(["ACGT", "ACGT"], 4),
    ]
    for _ in range(4):
        n1, n2 = r.randint(5, 15), r.randint(5, 15)
        s1, s2 = _random_dna(r, n1), _random_dna(r, n2)
        cases.append(_case([s1, s2], _ref_lcs_length(s1, s2)))

    prompt = _task_prompt_header(
        "Longest Common Subsequence Length",
        "lcs_length",
        "(seq1: str, seq2: str) -> int",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCompute the length of the longest common subsequence (LCS) of two sequences.\n"
        "A subsequence is obtained by deleting zero or more characters without changing "
        "the relative order of the remaining characters.\n"
        "Edge cases:\n"
        "- Any empty sequence \u2192 return 0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="LCS Length",
        prompt=prompt,
        entrypoint="lcs_length",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_reverse_complement(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], ""),
        _case(["A"], "T"),
        _case(["ACGT"], "ACGT"),  # palindromic
    ]
    for _ in range(4):
        n = r.randint(5, 25)
        s = _random_dna(r, n)
        cases.append(_case([s], _ref_reverse_complement(s)))

    prompt = _task_prompt_header(
        "Reverse Complement of a DNA sequence",
        "reverse_complement",
        "(seq: str) -> str",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nReturn the reverse complement of a DNA sequence.\n"
        "Complement mapping: A\u2194T, C\u2194G.  Reverse the complemented string.\n"
        "Input is uppercase (A, C, G, T only).\n"
        "Edge cases:\n"
        "- Empty string \u2192 return empty string.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Reverse Complement",
        prompt=prompt,
        entrypoint="reverse_complement",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_sequence_identity(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", ""], 0.0),
        _case(["ACGT", "AC"], 0.0),  # different lengths
    ]
    for _ in range(4):
        n = r.randint(8, 20)
        a1, a2 = _random_aligned_pair(r, n)
        cases.append(_case([a1, a2], _ref_sequence_identity(a1, a2)))

    prompt = _task_prompt_header(
        "Sequence Identity of two aligned sequences",
        "sequence_identity",
        "(aligned1: str, aligned2: str) -> float",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCompute the fraction of identical positions between two aligned sequences.\n"
        "Both strings have the same length and may contain '-' for gaps.\n"
        "Identity = (positions where aligned1[i] == aligned2[i] and neither is '-') "
        "/ len(aligned1).\n"
        "Edge cases:\n"
        "- Both empty \u2192 return 0.0.\n"
        "- Different lengths \u2192 return 0.0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Sequence Identity",
        prompt=prompt,
        entrypoint="sequence_identity",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


def make_task_count_kmers(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", 3], {}),
        _case(["AC", 3], {}),
        _case(["ACGT", 0], {}),
    ]
    for _ in range(4):
        n = r.randint(8, 20)
        k = r.randint(2, 4)
        s = _random_dna(r, n)
        cases.append(_case([s, k], _ref_count_kmers(s, k)))

    prompt = _task_prompt_header(
        "K-mer Frequency Count",
        "count_kmers",
        "(seq: str, k: int) -> dict[str, int]",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCount the frequency of every k-mer (substring of length k) using a "
        "sliding window of step 1.\n"
        "Return a dict mapping each observed k-mer string to its integer count.\n"
        "Edge cases:\n"
        "- k > len(seq) \u2192 return empty dict {}.\n"
        "- k <= 0 \u2192 return empty dict {}.\n"
        "- Empty sequence \u2192 return empty dict {}.\n"
    )
    return OgtsTask(
        id=task_id,
        title="K-mer Count",
        prompt=prompt,
        entrypoint="count_kmers",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_consensus_sequence(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([[]], ""),
        _case([["ACGT"]], "ACGT"),
    ]
    for _ in range(4):
        n_seqs = r.randint(3, 7)
        length = r.randint(5, 12)
        seqs = [_random_dna(r, length) for _ in range(n_seqs)]
        cases.append(_case([seqs], _ref_consensus(seqs)))

    prompt = _task_prompt_header(
        "Consensus Sequence from multiple aligned sequences",
        "consensus_sequence",
        "(sequences: list[str]) -> str",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nGiven a list of aligned DNA sequences (all the same length), return the "
        "consensus sequence.  At each column choose the most frequent base.  "
        "If there is a tie, choose the alphabetically first base (e.g. 'A' before 'C').\n"
        "Edge cases:\n"
        "- Empty list \u2192 return empty string.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Consensus Sequence",
        prompt=prompt,
        entrypoint="consensus_sequence",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_ti_tv_ratio(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", ""], -1.0),
        _case(["ACGT", "AC"], -1.0),   # different lengths
        _case(["AAAA", "AAAA"], -1.0),  # no differences
    ]
    for _ in range(4):
        n = r.randint(8, 20)
        s1 = _random_dna(r, n)
        # mutate ~40 % of positions to guarantee a mix of ti and tv
        s2list = list(s1)
        for j in range(n):
            if r.random() < 0.4:
                s2list[j] = r.choice("ACGT")
        s2 = "".join(s2list)
        cases.append(_case([s1, s2], _ref_ti_tv(s1, s2)))

    prompt = _task_prompt_header(
        "Transition / Transversion Ratio",
        "transition_transversion_ratio",
        "(seq1: str, seq2: str) -> float",
        FAMILY_ALIGNMENT,
    )
    prompt += (
        "\nCompute the transition/transversion ratio (Ti/Tv) between two equal-length "
        "DNA sequences.  Compare case-insensitively (convert to uppercase).\n"
        "Transitions: A\u2194G, C\u2194T (purine\u2194purine or pyrimidine\u2194pyrimidine).\n"
        "Transversions: all other substitutions (A\u2194C, A\u2194T, G\u2194C, G\u2194T).\n"
        "Ignore positions where both bases are identical.\n"
        "Return transitions / transversions.\n"
        "Edge cases:\n"
        "- If transversions = 0 (including when there are no mismatches at all), return -1.0.\n"
        "- Empty sequences or different lengths \u2192 return -1.0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Ti/Tv Ratio",
        prompt=prompt,
        entrypoint="transition_transversion_ratio",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


# ═══════════════════════════════════════════════════════════════════
#  Genomic QA family  (10 tasks)
# ═══════════════════════════════════════════════════════════════════


def make_task_gc_content(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], 0.0),
        _case(["AAAA"], 0.0),
        _case(["GCGC"], 1.0),
    ]
    for _ in range(4):
        n = r.randint(10, 30)
        s = _random_dna(r, n)
        cases.append(_case([s], _ref_gc_content(s)))

    prompt = _task_prompt_header(
        "GC Content of a DNA sequence",
        "gc_content",
        "(seq: str) -> float",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nCompute the GC content as the fraction of G and C bases in the sequence.\n"
        "Compare case-insensitively (convert to uppercase).\n"
        "GC content = (count of G + count of C) / length.\n"
        "Edge cases:\n"
        "- Empty sequence \u2192 return 0.0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="GC Content",
        prompt=prompt,
        entrypoint="gc_content",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


def make_task_translate_dna(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], ""),
        _case(["AT"], ""),
        _case(["ATGAAATTT"], "MKF"),
        _case(["ATGTAATGA"], "M**"),
    ]
    for _ in range(3):
        n = r.randint(4, 10) * 3  # multiple of 3
        s = _random_dna(r, n)
        cases.append(_case([s], _ref_translate(s)))

    prompt = _task_prompt_header(
        "Translate DNA to Protein",
        "translate_dna",
        "(seq: str) -> str",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nTranslate a DNA sequence to a protein string using the standard genetic code.\n"
        "Read codons in reading frame 0 (triplets from position 0).  Stop codons map to '*'.\n"
        "If a codon is not in the standard table, output '?' for that codon.\n"
        "Ignore incomplete codons at the end (len(seq) mod 3 leftover bases).\n"
        "Convert the input to uppercase before processing.\n\n"
        "Reference codon table:\n" + _CODON_TABLE_PROMPT + "\n\n"
        "Edge cases:\n"
        "- Empty sequence or length < 3 \u2192 return empty string.\n"
    )
    return OgtsTask(
        id=task_id,
        title="DNA Translation",
        prompt=prompt,
        entrypoint="translate_dna",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_mean_phred_quality(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], 0.0),
        _case(["!"], 0.0),   # '!' = ASCII 33, Phred 0
        _case(["I"], 40.0),  # 'I' = ASCII 73, Phred 40
    ]
    for _ in range(4):
        n = r.randint(10, 50)
        q = _random_quality_string(r, n)
        cases.append(_case([q], _ref_mean_phred(q)))

    prompt = _task_prompt_header(
        "Mean Phred Quality Score from a FASTQ quality string",
        "mean_phred_quality",
        "(quality_string: str) -> float",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nCompute the mean Phred quality score from a FASTQ quality string.\n"
        "Encoding: Phred+33 \u2014 each character's ASCII value minus 33 gives the "
        "quality score.\n"
        "Return the arithmetic mean of all quality scores.\n"
        "Edge cases:\n"
        "- Empty string \u2192 return 0.0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Mean Phred Quality",
        prompt=prompt,
        entrypoint="mean_phred_quality",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


def make_task_n50(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([[]], 0),
        _case([[100]], 100),
        _case([[10, 20, 30]], 30),  # total=60, half=30, sorted desc [30,20,10], 30>=30
    ]
    for _ in range(4):
        n = r.randint(5, 15)
        lengths = [r.randint(100, 10000) for _ in range(n)]
        cases.append(_case([lengths], _ref_n50(lengths)))

    prompt = _task_prompt_header(
        "N50 statistic for genome assembly contigs",
        "n50_statistic",
        "(contig_lengths: list[int]) -> int",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nCompute the N50 statistic from a list of contig lengths.\n"
        "Sort contigs by length descending, then accumulate lengths until the "
        "cumulative sum reaches at least 50% of the total assembly length.  "
        "The contig length that crosses the 50% threshold is the N50.\n"
        "Edge cases:\n"
        "- Empty list \u2192 return 0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="N50 Statistic",
        prompt=prompt,
        entrypoint="n50_statistic",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_find_motif_positions(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", "AC"], []),
        _case(["ACGT", ""], []),
        _case(["AAAA", "AA"], [0, 1, 2]),      # overlapping occurrences
        _case(["ACGTACGT", "ACGT"], [0, 4]),
    ]
    for _ in range(3):
        n = r.randint(15, 30)
        seq = _random_dna(r, n)
        motif_len = r.randint(2, 4)
        start = r.randint(0, n - motif_len)
        motif = seq[start : start + motif_len]  # guaranteed at least one hit
        cases.append(_case([seq, motif], _ref_find_motif(seq, motif)))

    prompt = _task_prompt_header(
        "Find all positions of a motif in a DNA sequence",
        "find_motif_positions",
        "(sequence: str, motif: str) -> list[int]",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nFind all starting positions (0-based) where the motif occurs in the sequence.\n"
        "Include overlapping occurrences.  Return positions in ascending order.\n"
        "Edge cases:\n"
        "- Empty sequence or empty motif \u2192 return empty list [].\n"
    )
    return OgtsTask(
        id=task_id,
        title="Motif Finder",
        prompt=prompt,
        entrypoint="find_motif_positions",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_melting_temperature(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], 0.0),
        _case(["A"], 2.0),
        _case(["G"], 4.0),
        _case(["ACGT"], 12.0),  # 2*(1+1) + 4*(1+1)
    ]
    for _ in range(3):
        n = r.randint(10, 30)
        s = _random_dna(r, n)
        cases.append(_case([s], _ref_melting_temp(s)))

    prompt = _task_prompt_header(
        "Melting Temperature of a short DNA oligo (Wallace rule)",
        "melting_temperature",
        "(seq: str) -> float",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nCompute the melting temperature (Tm) using the Wallace rule:\n"
        "  Tm = 2 * (count_A + count_T) + 4 * (count_G + count_C)\n"
        "Compare case-insensitively (convert to uppercase).\n"
        "Edge cases:\n"
        "- Empty sequence \u2192 return 0.0.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Melting Temperature",
        prompt=prompt,
        entrypoint="melting_temperature",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-9, "cases": cases},
    )


def make_task_validate_dna(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], {"valid": True, "length": 0, "invalid_chars": []}),
        _case(["ACGT"], {"valid": True, "length": 4, "invalid_chars": []}),
        _case(["ACGXNT"], {"valid": False, "length": 6, "invalid_chars": ["N", "X"]}),
        _case(["acgt"], {"valid": False, "length": 4, "invalid_chars": ["a", "c", "g", "t"]}),
    ]
    for _ in range(3):
        n = r.randint(10, 20)
        chars: list[str] = []
        for _ in range(n):
            if r.random() < 0.15:
                chars.append(r.choice("NXnx123"))
            else:
                chars.append(r.choice("ACGT"))
        s = "".join(chars)
        cases.append(_case([s], _ref_validate_dna(s)))

    prompt = _task_prompt_header(
        "Validate a DNA sequence string",
        "validate_dna",
        "(seq: str) -> dict",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nValidate whether a string is a valid DNA sequence.\n"
        "Valid characters: A, C, G, T (uppercase only \u2014 lowercase letters are invalid).\n"
        "Return a dict with exactly three keys:\n"
        "  'valid':  bool  \u2014 True if all characters are valid.\n"
        "  'length': int   \u2014 total length of the input string.\n"
        "  'invalid_chars': list[str] \u2014 sorted list of unique invalid characters found.\n"
        "Edge cases:\n"
        "- Empty string \u2192 {'valid': True, 'length': 0, 'invalid_chars': []}.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Validate DNA",
        prompt=prompt,
        entrypoint="validate_dna",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_codon_frequency(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], {}),
        _case(["AT"], {}),
        _case(["ATGATG"], {"ATG": 2}),
    ]
    for _ in range(4):
        n = r.randint(5, 12) * 3
        s = _random_dna(r, n)
        cases.append(_case([s], _ref_codon_freq(s)))

    prompt = _task_prompt_header(
        "Codon Frequency in a DNA sequence",
        "codon_frequency",
        "(seq: str) -> dict[str, int]",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nCount the frequency of each codon in a DNA sequence, reading in frame 0.\n"
        "A codon is a triplet of bases: seq[0:3], seq[3:6], seq[6:9], \u2026\n"
        "Ignore incomplete codons at the end.  Convert to uppercase first.\n"
        "Return a dict mapping each codon string to its integer count.\n"
        "Edge cases:\n"
        "- Empty sequence or length < 3 \u2192 return empty dict {}.\n"
    )
    return OgtsTask(
        id=task_id,
        title="Codon Frequency",
        prompt=prompt,
        entrypoint="codon_frequency",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_restriction_fragments(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case(["", "GAATTC"], []),
        _case(["ACGTACGT", ""], [8]),
        _case(["ACGTACGT", "XXXX"], [8]),  # site not found
        _case(["GAATTCGAATTC", "GAATTC"], [0, 0, 0]),  # site at edges
    ]
    for _ in range(3):
        # build a sequence with known site occurrences
        site = _random_dna(r, r.randint(4, 6))
        parts = [_random_dna(r, r.randint(3, 10)) for _ in range(r.randint(2, 4))]
        seq = site.join(parts)
        cases.append(_case([seq, site], _ref_restriction_fragments(seq, site)))

    prompt = _task_prompt_header(
        "Restriction Fragment Lengths after enzyme digestion",
        "restriction_fragment_lengths",
        "(seq: str, site: str) -> list[int]",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nSimulate restriction enzyme digestion: split the DNA sequence at every\n"
        "non-overlapping occurrence of the restriction site (left to right).\n"
        "The site is consumed by the cut (removed from fragments).\n"
        "Return the list of fragment lengths in left-to-right order.\n"
        "Equivalent to: [len(part) for part in seq.split(site)].\n"
        "Edge cases:\n"
        "- Empty sequence \u2192 return empty list [].\n"
        "- Empty site or site not found \u2192 return [len(seq)].\n"
    )
    return OgtsTask(
        id=task_id,
        title="Restriction Fragments",
        prompt=prompt,
        entrypoint="restriction_fragment_lengths",
        oracle_kind="json_equal",
        oracle_payload={"cases": cases},
    )


def make_task_molecular_weight(task_id: str, seed: int) -> OgtsTask:
    r = _rnd(seed)
    cases = [
        _case([""], 0.0),
        _case(["A"], 331.2),
        _case(["AC"], 331.2 + 307.2 - 18.02),  # 620.38
    ]
    for _ in range(4):
        n = r.randint(5, 25)
        s = _random_dna(r, n)
        cases.append(_case([s], _ref_molecular_weight(s)))

    prompt = _task_prompt_header(
        "Approximate Molecular Weight of single-stranded DNA",
        "molecular_weight_dna",
        "(seq: str) -> float",
        FAMILY_GENOMIC_QA,
    )
    prompt += (
        "\nCompute the approximate molecular weight of a single-stranded DNA molecule.\n"
        "Nucleotide weights (Da):\n"
        "  A = 331.2,  C = 307.2,  G = 347.2,  T = 322.2\n"
        "For each phosphodiester bond (n\u22121 bonds for n nucleotides), subtract 18.02 Da "
        "(water lost during condensation).\n"
        "Formula:  MW = sum(weight[base] for base in seq) \u2212 (len(seq) \u2212 1) * 18.02\n"
        "Convert to uppercase before processing.\n"
        "Edge cases:\n"
        "- Empty sequence \u2192 return 0.0.\n"
        "- Single nucleotide \u2192 return just the nucleotide weight (no bond).\n"
    )
    return OgtsTask(
        id=task_id,
        title="Molecular Weight",
        prompt=prompt,
        entrypoint="molecular_weight_dna",
        oracle_kind="numeric_close",
        oracle_payload={"tol": 1e-6, "cases": cases},
    )


# ═══════════════════════════════════════════════════════════════════
#  Aggregator
# ═══════════════════════════════════════════════════════════════════

ALIGNMENT_MAKERS = [
    make_task_hamming_distance,
    make_task_edit_distance,
    make_task_needleman_wunsch,
    make_task_smith_waterman,
    make_task_lcs_length,
    make_task_reverse_complement,
    make_task_sequence_identity,
    make_task_count_kmers,
    make_task_consensus_sequence,
    make_task_ti_tv_ratio,
]

GENOMIC_QA_MAKERS = [
    make_task_gc_content,
    make_task_translate_dna,
    make_task_mean_phred_quality,
    make_task_n50,
    make_task_find_motif_positions,
    make_task_melting_temperature,
    make_task_validate_dna,
    make_task_codon_frequency,
    make_task_restriction_fragments,
    make_task_molecular_weight,
]


def make_bioinformatics_tasks_20(seed: int = 2024) -> list[OgtsTask]:
    """
    Build a deterministic suite of 20 bioinformatics execution-grounded tasks.

    10 from *Sequence Alignment*, 10 from *Genomic QA*.
    Each task is instantiated once with a unique sub-seed derived from *seed*.
    """
    r = _rnd(seed)
    tasks: list[OgtsTask] = []
    i = 0
    for maker in ALIGNMENT_MAKERS + GENOMIC_QA_MAKERS:
        i += 1
        tasks.append(maker(f"bio_{i:03d}", r.randint(0, 10_000_000)))
    assert len(tasks) == 20
    return tasks
