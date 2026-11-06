"""Per-product shelf diagnosis: retrieval visibility + agent pick behaviour + reasons."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.diagnose.select_products import RANK_CAP


def _rank_stats(r: pd.DataFrame, col: str) -> dict:
    v = r[col].fillna(RANK_CAP)
    return {"top10_rate": float((v <= 10).mean()), "top50_rate": float((v <= 50).mean()),
            "median_rank": float(v.median()), "mean_rank_capped": float(v.mean())}


def diagnose_product(pid: str, links: pd.DataFrame, examples: pd.DataFrame, ranks: pd.DataFrame,
                     runs: pd.DataFrame, cands: pd.DataFrame) -> dict:
    """runs: agent rows for the product's queries (tag 'diag'). cands: prompt candidate lists."""
    my = links[links["product_id"] == pid]
    e_q = my[my["target_label"] == "E"]["query_id"].tolist()
    ne_q = my[my["target_label"] != "E"]["query_id"].tolist()
    qtext = examples.drop_duplicates("query_id").set_index("query_id")["query"]
    r = ranks[(ranks["product_id"] == pid) & (ranks["query_id"].isin(e_q))]
    retrieval = {k: _rank_stats(r, c) for k, c in [("bm25", "rank_bm25"), ("dense", "rank_dense"), ("hybrid", "rank_hybrid")]}
    per_query = []
    mine = runs[runs["product_id"] == pid].set_index("query_id")
    n_e_in_prompt = cands[cands["in_prompt"] & (cands["esci_label"] == "E")].groupby("query_id").size()
    for qid in e_q + ne_q:
        row = {"query_id": int(qid), "query": qtext.get(qid, ""), "human_label": "E" if qid in e_q else my[my["query_id"] == qid]["target_label"].iloc[0]}
        rr = r[r["query_id"] == qid]
        for c in ["rank_bm25", "rank_dense", "rank_hybrid"]:
            val = rr[c].iloc[0] if len(rr) else None
            row[c] = None if val is None or pd.isna(val) else int(val)
        if qid in mine.index:
            m = mine.loc[qid]
            row.update({"agent_label": m["agent_label"], "picked": bool(m["is_pick"]), "abstained": bool(m.get("abstained", False)),
                        "reason_code": m["reason_code"], "missing_attributes": list(m["missing_attributes"] or []),
                        "n_e_in_prompt": int(n_e_in_prompt.get(qid, 0))})
        per_query.append(row)
    e_rows = [p for p in per_query if p["human_label"] == "E" and "picked" in p]
    ne_rows = [p for p in per_query if p["human_label"] != "E" and "picked" in p]
    pick_rate_e = float(np.mean([p["picked"] for p in e_rows])) if e_rows else float("nan")
    fair = float(np.mean([1.0 / max(p["n_e_in_prompt"], 1) for p in e_rows])) if e_rows else float("nan")
    reasons = Counter(p["reason_code"] for p in e_rows if not p["picked"] and p.get("reason_code"))
    missing = Counter(m.lower() for p in e_rows if not p["picked"] for m in p.get("missing_attributes", []))
    agent_labels_on_e = Counter(p["agent_label"] for p in e_rows if p.get("agent_label"))
    return {
        "product_id": pid, "n_e_queries": len(e_q), "n_non_e_queries": len(ne_q),
        "retrieval": retrieval,
        "agent": {"pick_rate_E": pick_rate_e, "fair_share": fair,
                  "pick_rate_vs_fair_share": (pick_rate_e / fair) if fair and fair == fair and fair > 0 else float("nan"),
                  "pick_rate_nonE": float(np.mean([p["picked"] for p in ne_rows])) if ne_rows else float("nan"),
                  "agent_labels_on_E_queries": dict(agent_labels_on_e),
                  "top_reasons": reasons.most_common(3), "top_missing_attributes": missing.most_common(5)},
        "per_query": per_query,
    }
