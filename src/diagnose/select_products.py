"""Pick products for the shelf diagnosis / rewrite experiment and build their query sets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent.judge import select_for_prompt
from src.config import CFG

RANK_CAP = 101  # rank assigned when a product is outside the top-k_full list


def product_pool(examples: pd.DataFrame, ranks: pd.DataFrame, depts: pd.DataFrame) -> pd.DataFrame:
    """One row per product with >=min_e E queries and >=min_non_e S/I queries, plus baseline rank stats."""
    d = CFG["diagnosis"]
    ex = examples.merge(depts[["query_id", "department"]], on="query_id", how="left")
    ex["department"] = ex["department"].fillna(CFG["departments"][-1])
    ex["is_e"] = ex["esci_label"] == "E"
    ex["is_non_e"] = ex["esci_label"].isin(["S", "I"])
    agg = ex.groupby("product_id").agg(n_e=("is_e", "sum"), n_non_e=("is_non_e", "sum"), n_queries=("query_id", "nunique"))
    agg = agg[(agg["n_e"] >= d["min_e_queries"]) & (agg["n_non_e"] >= d["min_non_e_queries"])]
    dept = ex[ex["product_id"].isin(agg.index)].groupby("product_id")["department"].agg(lambda s: s.mode().iloc[0])
    r = ranks[(ranks["esci_label"] == "E") & ranks["product_id"].isin(agg.index)].copy()
    r["rank_hybrid"] = r["rank_hybrid"].fillna(RANK_CAP)
    r["rank_bm25"] = r["rank_bm25"].fillna(RANK_CAP)
    r["rank_dense"] = r["rank_dense"].fillna(RANK_CAP)
    rs = r.groupby("product_id").agg(median_rank_hybrid=("rank_hybrid", "median"), top10_rate=("rank_hybrid", lambda s: (s <= 10).mean()),
                                     median_rank_bm25=("rank_bm25", "median"), median_rank_dense=("rank_dense", "median"))
    pool = agg.join(dept).join(rs).reset_index()
    pool["weak_baseline"] = pool["median_rank_hybrid"] > d["weak_rank_threshold"]
    return pool


def select_products(pool: pd.DataFrame, n: int, seed: int, min_per_dept: int = 5) -> pd.DataFrame:
    """Prefer weak-baseline products; proportional by department with a floor."""
    rng = np.random.default_rng(seed)
    weak = pool[pool["weak_baseline"]]
    src = weak if len(weak) >= n else pool
    counts = src["department"].value_counts()
    alloc = (counts / counts.sum() * n).round().astype(int).clip(lower=min_per_dept).clip(upper=counts)
    while alloc.sum() > n:
        alloc[alloc.idxmax()] -= 1
    parts = [src[src["department"] == dpt].sample(n=int(k), random_state=int(rng.integers(0, 2**31)))
             for dpt, k in alloc.items() if k > 0]
    return pd.concat(parts).reset_index(drop=True)


def build_query_sets(selected: pd.DataFrame, examples: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (queries, candidates, links). links: product_id x query_id x esci_label for the diagnosed products.

    Every diagnosed product is force-kept in its queries' prompt candidate lists.
    """
    pids = set(selected["product_id"])
    links = examples[examples["product_id"].isin(pids)][["product_id", "query_id", "esci_label"]].rename(columns={"esci_label": "target_label"})
    qids = links["query_id"].unique()
    ex = examples[examples["query_id"].isin(qids)]
    queries = ex.drop_duplicates("query_id")[["query_id", "query"]].sort_values("query_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    for qid, g in ex.groupby("query_id", sort=True):
        g = g[["query_id", "product_id", "esci_label"]].copy()
        g["shuffled_pos"] = rng.permutation(len(g))
        g = select_for_prompt(g, seed=int(rng.integers(0, 2**31)))
        g.loc[g["product_id"].isin(pids), "in_prompt"] = True  # force-keep targets
        parts.append(g)
    cands = pd.concat(parts).sort_values(["query_id", "shuffled_pos"]).reset_index(drop=True)
    return queries, cands, links
