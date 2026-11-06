"""Before/after statistics for the rewrite experiment, with negative controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.diagnose.select_products import RANK_CAP


def paired_bootstrap(before: np.ndarray, after: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    if d.size == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return {"delta": float(d.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(d.size)}


def mcnemar_p(before: np.ndarray, after: np.ndarray) -> float:
    b, a = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    n01 = int(((~b) & a).sum())
    n10 = int((b & (~a)).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(mcnemar([[0, n01], [n10, 0]], exact=True).pvalue)


def _pivot(ba: pd.DataFrame, col: str) -> pd.DataFrame:
    p = ba.pivot_table(index=["product_id", "query_id", "label"], columns="variant", values=col, aggfunc="first")
    return p.reset_index()


def summarize(ba: pd.DataFrame, listings: pd.DataFrame) -> dict:
    ba = ba.copy()
    ba["rank_capped"] = ba["rank_hybrid"].fillna(RANK_CAP).astype(float)
    ba["top10"] = ba["rank_capped"] <= 10
    ba["picked"] = ba["picked"].fillna(False).astype(bool)
    ba["is_e"] = ba["label"] == "E"
    out: dict = {"variants": {}, "faithfulness": {}, "per_product": {}}
    variants = [v for v in ba["variant"].unique() if v != "original"]
    for v in variants:
        res: dict = {}
        for subset, mask in [("E_queries", ba["is_e"]), ("nonE_queries", ~ba["is_e"])]:
            sub = ba[mask & ba["variant"].isin(["original", v])]
            pk = _pivot(sub, "picked").dropna(subset=["original", v])
            tp = _pivot(sub, "top10").dropna(subset=["original", v])
            rk = _pivot(sub, "rank_capped").dropna(subset=["original", v])
            res[subset] = {
                "n_pairs": int(len(pk)),
                "pick_rate_before": float(pk["original"].astype(float).mean()) if len(pk) else float("nan"),
                "pick_rate_after": float(pk[v].astype(float).mean()) if len(pk) else float("nan"),
                "pick_delta": paired_bootstrap(pk["original"].astype(float), pk[v].astype(float)),
                "pick_mcnemar_p": mcnemar_p(pk["original"].astype(bool), pk[v].astype(bool)) if len(pk) else float("nan"),
                "top10_before": float(tp["original"].astype(float).mean()) if len(tp) else float("nan"),
                "top10_after": float(tp[v].astype(float).mean()) if len(tp) else float("nan"),
                "top10_delta": paired_bootstrap(tp["original"].astype(float), tp[v].astype(float)),
                "median_rank_before": float(rk["original"].median()) if len(rk) else float("nan"),
                "median_rank_after": float(rk[v].median()) if len(rk) else float("nan"),
                "rank_delta": paired_bootstrap(rk["original"], rk[v]),
            }
        out["variants"][v] = res
        lv = listings[listings["variant"] == v]
        out["faithfulness"][v] = {"n": int(len(lv)), "pass_rate": float(lv["passed"].mean()) if len(lv) else float("nan"),
                                  "rule_pass_rate": float(lv["rule_pass"].mean()) if len(lv) else float("nan"),
                                  "llm_pass_rate": float(lv["llm_pass"].mean()) if len(lv) else float("nan"),
                                  "stuffing_flag_rate": float(lv["stuffing_flag"].fillna(False).astype(bool).mean()) if len(lv) else float("nan"),
                                  "mean_rounds": float(lv["rounds"].mean()) if len(lv) else float("nan")}
        # per product deltas (E queries)
        sub = ba[ba["is_e"] & ba["variant"].isin(["original", v])]
        pk = _pivot(sub, "picked").dropna(subset=["original", v])
        pp = pk.groupby("product_id").apply(lambda g: float(g[v].astype(float).mean() - g["original"].astype(float).mean()), include_groups=False)
        rk = _pivot(sub, "rank_capped").dropna(subset=["original", v])
        rp = rk.groupby("product_id").apply(lambda g: float(g[v].median() - g["original"].median()), include_groups=False)
        per = pd.DataFrame({"pick_delta_E": pp, "median_rank_delta_E": rp}).reset_index()
        out["per_product"][v] = {"table": per.to_dict("records"),
                                 "top5_pick_gain": per.sort_values("pick_delta_E", ascending=False).head(5).to_dict("records"),
                                 "bottom5_pick_gain": per.sort_values("pick_delta_E").head(5).to_dict("records"),
                                 "share_products_improved_pick": float((per["pick_delta_E"] > 0).mean()) if len(per) else float("nan")}
    return out


def headline_table(summary: dict) -> pd.DataFrame:
    rows = []
    for v, res in summary["variants"].items():
        e, ne, f = res["E_queries"], res["nonE_queries"], summary["faithfulness"][v]
        rows.append({"variant": v,
                     "E pick rate before→after": f"{e['pick_rate_before']:.2f} → {e['pick_rate_after']:.2f} (Δ {e['pick_delta']['delta']:+.2f}, p={e['pick_mcnemar_p']:.3f})",
                     "E top-10 before→after": f"{e['top10_before']:.2f} → {e['top10_after']:.2f}",
                     "E median rank before→after": f"{e['median_rank_before']:.0f} → {e['median_rank_after']:.0f}",
                     "non-E pick rate before→after": f"{ne['pick_rate_before']:.2f} → {ne['pick_rate_after']:.2f} (Δ {ne['pick_delta']['delta']:+.2f})",
                     "faithfulness pass": f"{f['pass_rate']:.2f}", "stuffing flagged": f"{f['stuffing_flag_rate']:.2f}"})
    return pd.DataFrame(rows)
