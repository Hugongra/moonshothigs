"""Retrieval metrics for sparse binary relevance labels."""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(is_hit_per_rank: list[bool], k: int) -> float:
    """1 if any of the first k ranks is relevant, else 0."""
    return 1.0 if any(is_hit_per_rank[:k]) else 0.0


def reciprocal_rank(is_hit_per_rank: list[bool]) -> float:
    """1/r where r is 1-based index of first hit; 0 if none."""
    for i, hit in enumerate(is_hit_per_rank):
        if hit:
            return 1.0 / float(i + 1)
    return 0.0


def dcg_binary(relevance_bits: Iterable[float], k: int) -> float:
    """DCG with logarithmic discount; relevance_bits are already truncated to k."""
    bits = list(relevance_bits)[:k]
    s = 0.0
    for i, rel in enumerate(bits):
        gain = rel  # binary 0/1
        if gain > 0:
            s += gain / math.log2(i + 2.0)
    return s


def ndcg_at_k(is_hit_per_rank: list[bool], k: int) -> float:
    """
    nDCG when each retrieved chunk is either relevant (1) or not (0).
    Ideal ordering puts all relevant chunks first — here at most one grade level unless multi-rel added.
    """
    bits = [1.0 if h else 0.0 for h in is_hit_per_rank[:k]]
    dcg = dcg_binary(bits, k)
    ideal_hits = sorted(bits, key=lambda x: (x < 1.0, -x))  # 1s first
    idcg = dcg_binary(ideal_hits, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
