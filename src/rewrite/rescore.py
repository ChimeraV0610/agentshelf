"""Re-rank one product with new listing text without rebuilding the indices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CFG
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex


def doc_text_from(title: str, brand: str, color: str, bullets: list[str]) -> str:
    """Same format as src.data.load.build_doc_text."""
    n_bullets = CFG["doc_text"]["n_bullets"]
    max_chars = CFG["doc_text"]["max_chars"]
    parts = [title]
    if brand:
        parts.append(f"Brand: {brand}")
    if color:
        parts.append(f"Color: {color}")
    parts.extend(bullets[:n_bullets])
    return " . ".join(p for p in parts if p)[:max_chars]


class Rescorer:
    def __init__(self, bm25: BM25Index, dense: DenseIndex):
        self.bm25, self.dense = bm25, dense
        self.k = CFG["retrieval"]["k_full"]
        self.rrf_k = CFG["retrieval"]["rrf_k"]
        self._q: dict[str, dict] = {}

    def _prep_query(self, query: str) -> dict:
        if query not in self._q:
            qv = self.dense.encode_queries([query])[0]
            bm_ids, _ = self.bm25.search([query], k=self.k + 1)
            de_ids, _ = self.dense.search_vectors(qv[None, :], k=self.k + 1)
            self._q[query] = {"qv": qv, "bm": bm_ids[0], "de": de_ids[0]}
        return self._q[query]

    def _hybrid_rank(self, prep: dict, pid: str, rb: int, rd: int) -> int | None:
        """RRF rank among the fused top-k lists with pid's old entry removed and the new one inserted."""
        bm = [p for p in prep["bm"] if p != pid][: self.k]
        de = [p for p in prep["de"] if p != pid][: self.k]
        scores: dict[str, float] = {}
        for lst in (bm, de):
            for r, p in enumerate(lst):
                scores[p] = scores.get(p, 0.0) + 1.0 / (self.rrf_k + r + 1)
        mine = (1.0 / (self.rrf_k + rb) if rb <= self.k else 0.0) + (1.0 / (self.rrf_k + rd) if rd <= self.k else 0.0)
        if mine == 0.0:
            return None
        return int(sum(1 for s in scores.values() if s > mine)) + 1

    def ranks(self, pid: str, query: str, doc_text: str) -> dict:
        prep = self._prep_query(query)
        sb = self.bm25.score_single_doc(query, doc_text)
        rb = self.bm25.rank_of_score(query, sb, exclude_pid=pid)
        dv = self.dense.encode_single_doc(doc_text)
        sd = float(dv @ prep["qv"])
        rd = self.dense.rank_of_score(prep["qv"], sd, exclude_pid=pid)
        rh = self._hybrid_rank(prep, pid, rb, rd)
        cap = lambda r: None if r is None or r > self.k else int(r)  # noqa: E731 - match full_corpus_ranks semantics
        return {"rank_bm25": cap(rb), "rank_dense": cap(rd), "rank_hybrid": rh, "score_bm25": sb, "score_dense": sd}
