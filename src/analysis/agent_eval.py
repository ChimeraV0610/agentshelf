"""Module 1 metrics: confusion matrix, F1, wrong-buy rate, abstention, agreement, brand view."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

LABELS = ["E", "S", "C", "I"]


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def candidate_level(runs: pd.DataFrame) -> dict:
    r = runs[runs["agent_label"].notna()]
    cm = confusion_matrix(r["esci_label"], r["agent_label"], labels=LABELS)
    return {
        "n": int(len(r)),
        "labelled_share": float(runs["agent_label"].notna().mean()),
        "confusion_matrix": {"labels": LABELS, "rows_human_cols_agent": cm.tolist()},
        "macro_f1": float(f1_score(r["esci_label"], r["agent_label"], labels=LABELS, average="macro")),
        "micro_f1": float(f1_score(r["esci_label"], r["agent_label"], labels=LABELS, average="micro")),
        "per_class_f1": dict(zip(LABELS, f1_score(r["esci_label"], r["agent_label"], labels=LABELS, average=None).round(4).tolist())),
        "kappa_vs_human": float(cohen_kappa_score(r["esci_label"], r["agent_label"])),
        "esci_readme_baseline": {"macro_f1": 0.23, "micro_f1": 0.62},
    }


def query_level(runs: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """One row per query: picked label, wrong_buy, abstained, has_e_in_prompt."""
    rows = []
    for qid, g in runs.groupby("query_id", sort=False):
        pick = g[g["is_pick"]]
        has_e = bool((g["esci_label"] == "E").any())
        if len(pick):
            lab = pick["esci_label"].iloc[0]
            rows.append({"query_id": qid, "abstained": False, "picked_label": lab,
                         "wrong_buy": lab != "E", "s_as_e": lab == "S", "has_e": has_e,
                         "pick_product_id": pick["product_id"].iloc[0]})
        else:
            rows.append({"query_id": qid, "abstained": True, "picked_label": None, "wrong_buy": None,
                         "s_as_e": None, "has_e": has_e, "pick_product_id": None})
    q = pd.DataFrame(rows).merge(queries, on="query_id", how="left")
    return q


def wrong_buy_summary(q: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    picked = q[~q["abstained"]]
    groups = [(None, picked)] if by is None else list(picked.groupby(by))
    rows = []
    for key, g in groups:
        wb = g["wrong_buy"].astype(float).values
        lo, hi = bootstrap_ci(wb)
        rows.append({"group": "all" if key is None else key, "n_picked": len(g),
                     "wrong_buy_rate": float(wb.mean()) if len(g) else float("nan"), "ci_low": lo, "ci_high": hi,
                     "s_as_e_rate": float(g["s_as_e"].astype(float).mean()) if len(g) else float("nan"),
                     "picked_I_rate": float((g["picked_label"] == "I").mean()) if len(g) else float("nan")})
    return pd.DataFrame(rows)


def abstention(q: pd.DataFrame) -> dict:
    ab = q[q["abstained"]]
    return {"abstain_rate": float(q["abstained"].mean()),
            "abstained_but_E_available": float(ab["has_e"].mean()) if len(ab) else float("nan"),
            "n_abstained": int(len(ab))}


def random_pick_baseline(cands: pd.DataFrame) -> float:
    """Expected wrong-buy rate of picking uniformly among in-prompt candidates."""
    c = cands[cands["in_prompt"]]
    return float(c.groupby("query_id")["esci_label"].apply(lambda s: (s != "E").mean()).mean())


def agreement(runs_a: pd.DataFrame, runs_b: pd.DataFrame) -> dict:
    m = runs_a.merge(runs_b, on=["query_id", "product_id"], suffixes=("_a", "_b"))
    m = m[m["agent_label_a"].notna() & m["agent_label_b"].notna()]
    qa = m.groupby("query_id").apply(lambda g: g["is_pick_a"].values.tolist() == g["is_pick_b"].values.tolist(), include_groups=False)
    return {"n_pairs": int(len(m)), "label_kappa": float(cohen_kappa_score(m["agent_label_a"], m["agent_label_b"])),
            "label_agreement": float((m["agent_label_a"] == m["agent_label_b"]).mean()),
            "same_pick_rate": float(qa.mean())}


def brand_view(runs: pd.DataFrame, products: pd.DataFrame, min_candidates: int = 15) -> pd.DataFrame:
    r = runs.merge(products[["product_id", "product_brand"]], on="product_id", how="left")
    r["brand"] = r["product_brand"].fillna("").str.strip().str.lower()
    r = r[r["brand"] != ""]
    cand_share = r["brand"].value_counts(normalize=True)
    cand_n = r["brand"].value_counts()
    pick_share = r[r["is_pick"]]["brand"].value_counts(normalize=True)
    df = pd.DataFrame({"candidate_share": cand_share, "pick_share": pick_share}).fillna(0.0)
    df["n_candidates"] = cand_n
    df["n_picks"] = r[r["is_pick"]]["brand"].value_counts()
    df["n_picks"] = df["n_picks"].fillna(0).astype(int)
    df["preference_index"] = df["pick_share"] / df["candidate_share"]
    df = df[df["n_candidates"] >= min_candidates].sort_values("preference_index", ascending=False)
    return df.reset_index().rename(columns={"index": "brand"})


def brand_field_effect(runs: pd.DataFrame, products: pd.DataFrame) -> dict:
    r = runs[runs["esci_label"] == "E"].merge(products[["product_id", "product_brand"]], on="product_id", how="left")
    r["has_brand"] = r["product_brand"].fillna("").str.strip() != ""
    g = r.groupby("has_brand")["is_pick"].mean()
    return {"pick_rate_E_with_brand": float(g.get(True, float("nan"))), "pick_rate_E_without_brand": float(g.get(False, float("nan")))}


def reason_summary(runs: pd.DataFrame) -> pd.DataFrame:
    r = runs[(runs["esci_label"] == "E") & (~runs["is_pick"]) & runs["reason_code"].notna()]
    return r["reason_code"].value_counts(normalize=True).round(4).rename("share").reset_index()
