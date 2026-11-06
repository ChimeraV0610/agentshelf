"""Step 3b: stratified sample of test queries + fixed shuffled candidate lists."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agent.judge import select_for_prompt  # noqa: E402
from src.config import CFG, path  # noqa: E402
from src.data.load import load_processed  # noqa: E402
from src.sampling.strata import query_features, stratified_sample  # noqa: E402


def main():
    s = CFG["sampling"]
    seed = CFG["seed"]
    ex = load_processed("examples_us")
    ex = ex[ex["split"] == s["split"]]
    prod = load_processed("products_us")[["product_id", "product_brand"]]
    feats = query_features(ex, prod)
    depts = pd.read_parquet(path("samples") / "query_departments.parquet")
    feats = feats.merge(depts[["query_id", "department"]], on="query_id", how="left")
    feats["department"] = feats["department"].fillna(CFG["departments"][-1])
    feats = feats[feats["n_candidates"] >= s["min_candidates"]]
    print(f"eligible queries: {len(feats):,}")
    sample = stratified_sample(feats, s["n_queries"], s["min_per_department"], seed)
    sample = sample.sort_values("query_id").reset_index(drop=True)
    sample.to_parquet(path("samples") / "queries_1000.parquet", index=False)

    rng = np.random.default_rng(seed)
    parts = []
    for qid, g in ex[ex["query_id"].isin(sample["query_id"])].groupby("query_id", sort=True):
        g = g[["query_id", "product_id", "esci_label"]].copy()
        g["shuffled_pos"] = rng.permutation(len(g))
        g = select_for_prompt(g, seed=int(rng.integers(0, 2**31)))
        parts.append(g)
    cands = pd.concat(parts).sort_values(["query_id", "shuffled_pos"]).reset_index(drop=True)
    cands.to_parquet(path("samples") / "candidates_1000.parquet", index=False)

    lines = ["# Sample design", "", f"- split: {s['split']}, eligible: {len(feats):,}, sampled: {len(sample):,}",
             f"- seed: {seed}; candidates capped at {s['max_candidates_in_prompt']} per query (all E kept)",
             f"- truncated queries: {int((cands.groupby('query_id')['in_prompt'].sum() < cands.groupby('query_id').size()).sum())}",
             "", "## Department", "", sample["department"].value_counts().to_string(),
             "", "## Brand query", "", sample["is_brand_query"].value_counts().to_string(),
             "", "## Difficulty (E share of candidates)", "", sample["difficulty"].value_counts().to_string(),
             "", "## Eligible-population marginals (for comparison)", "",
             feats["department"].value_counts(normalize=True).round(3).to_string(), "",
             feats["difficulty"].value_counts(normalize=True).round(3).to_string()]
    (path("reports") / "sample_design.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
