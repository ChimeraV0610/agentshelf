"""Stratification features and stratified sampling of queries."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import CFG

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(s).lower()))


def query_features(examples: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """One row per query: n_candidates, e_share, difficulty, is_brand_query."""
    brand_map = products.set_index("product_id")["product_brand"].fillna("").str.lower()
    ex = examples.merge(brand_map.rename("brand"), left_on="product_id", right_index=True, how="left")
    rows = []
    for (qid, query), g in ex.groupby(["query_id", "query"], sort=False):
        q_tok = _tokens(query)
        brands = {b for b in g["brand"].fillna("") if b}
        is_brand = any(len(_tokens(b)) > 0 and _tokens(b) <= q_tok for b in brands)
        n = len(g)
        e_share = float((g["esci_label"] == "E").mean())
        rows.append({
            "query_id": qid, "query": query, "n_candidates": n, "e_share": e_share,
            "n_e": int((g["esci_label"] == "E").sum()),
            "is_brand_query": bool(is_brand),
            "split": g["split"].iloc[0],
        })
    df = pd.DataFrame(rows)
    df["difficulty"] = pd.cut(df["e_share"], bins=[-0.01, 0.25, 0.75, 1.01], labels=["hard", "medium", "easy"]).astype(str)
    return df


def stratified_sample(feats: pd.DataFrame, n_total: int, min_per_dept: int, seed: int,
                      dept_col: str = "department") -> pd.DataFrame:
    """Proportional allocation by department with a per-department floor."""
    rng = np.random.default_rng(seed)
    counts = feats[dept_col].value_counts()
    alloc = (counts / counts.sum() * n_total).round().astype(int)
    alloc = alloc.clip(lower=min_per_dept)
    alloc = alloc.clip(upper=counts)  # cannot take more than available
    # rescale to n_total
    while alloc.sum() > n_total:
        biggest = alloc.idxmax()
        alloc[biggest] -= 1
    while alloc.sum() < n_total:
        room = (counts - alloc)
        room = room[room > 0]
        if room.empty:
            break
        alloc[(alloc / counts).idxmin() if (alloc / counts).idxmin() in room.index else room.idxmax()] += 1
    parts = []
    for dept, n in alloc.items():
        pool = feats[feats[dept_col] == dept]
        take = pool.sample(n=min(int(n), len(pool)), random_state=int(rng.integers(0, 2**31)))
        parts.append(take)
    out = pd.concat(parts).reset_index(drop=True)
    return out
