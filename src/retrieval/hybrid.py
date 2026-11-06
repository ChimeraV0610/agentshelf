"""Reciprocal rank fusion."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.config import CFG


def rrf(rank_lists: list[list[str]], k: int | None = None) -> list[tuple[str, float]]:
    """Fuse several ranked id lists. Returns [(id, score)] sorted descending."""
    k = k or CFG["retrieval"]["rrf_k"]
    acc: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for r, pid in enumerate(ranked):
            acc[pid] += 1.0 / (k + r + 1)
    return sorted(acc.items(), key=lambda x: -x[1])


def rrf_scores_for_candidates(score_lists: list[np.ndarray], k: int | None = None) -> np.ndarray:
    """RRF restricted to one candidate list: each score_list is scores over the same candidates."""
    k = k or CFG["retrieval"]["rrf_k"]
    n = len(score_lists[0])
    out = np.zeros(n)
    for s in score_lists:
        order = np.argsort(-np.asarray(s), kind="stable")
        for r, i in enumerate(order):
            out[i] += 1.0 / (k + r + 1)
    return out
