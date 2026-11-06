"""nDCG within each query's labelled candidate list (ESCI Task 1 gains)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.config import CFG

GAIN = {k: float(v) for k, v in CFG["ndcg_gain"].items()}


def dcg(gains: Sequence[float]) -> float:
    gains = np.asarray(gains, dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum(gains / discounts))


def ndcg(labels_in_ranked_order: Sequence[str], k: int | None = None) -> float:
    gains = [GAIN[l] for l in labels_in_ranked_order]
    if k is not None:
        gains_k = gains[:k]
    else:
        gains_k = gains
    ideal = sorted(gains, reverse=True)[: len(gains_k)]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(gains_k) / idcg


def ndcg_from_scores(scores: Sequence[float], labels: Sequence[str], k: int | None = None) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return ndcg([labels[i] for i in order], k=k)


def evaluate_ranker(
    score_fn: Callable[[str, list[str]], np.ndarray],
    examples: pd.DataFrame,
    k: int | None = None,
) -> tuple[float, pd.DataFrame]:
    """score_fn(query, product_ids) -> array of scores (higher = better).

    Returns (mean nDCG, per-query DataFrame).
    """
    rows = []
    for (qid, query), g in examples.groupby(["query_id", "query"], sort=False):
        pids = g["product_id"].tolist()
        labels = g["esci_label"].tolist()
        scores = np.asarray(score_fn(query, pids), dtype=float)
        rows.append({"query_id": qid, "ndcg": ndcg_from_scores(scores, labels, k=k)})
    per_query = pd.DataFrame(rows)
    return float(per_query["ndcg"].mean()), per_query
