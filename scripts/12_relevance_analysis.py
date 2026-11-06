"""Evidence: on 40k US query-product pairs, which listing features predict a human 'Exact' label?
Logistic regression + Exact rate by buyer-term coverage bucket -> results/relevance_analysis.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import CFG, path  # noqa: E402
from src.data.load import clean_text, load_processed, split_bullets  # noqa: E402
from src.engine.audit import FLUFF, KEY_ATTRS, _NUM_UNIT  # noqa: E402

_WORD = re.compile(r"[a-z0-9]+")
STOP = {"for", "the", "and", "with", "a", "an", "of", "to", "in", "on"}


def main(n: int = 40000):
    ex = load_processed("examples_us")
    ex = ex[ex["esci_label"].isin(["E", "S", "C", "I"])].sample(n=n, random_state=CFG["seed"])
    prod = load_processed("products_us").set_index("product_id")
    rows = []
    for r in ex.itertuples(index=False):
        p = prod.loc[r.product_id]
        title = clean_text(p["product_title"])
        bullets = split_bullets(p["product_bullet_point"])
        desc = clean_text(p["product_description"])
        text = " ".join([title] + bullets + [desc]).lower()
        toks = set(_WORD.findall(text))
        q = [w for w in _WORD.findall(r.query.lower()) if w not in STOP]
        cov = sum(w in toks for w in q) / max(len(q), 1)
        n_words = max(len(_WORD.findall(text)), 1)
        rows.append({
            "exact": int(r.esci_label == "E"),
            "query_term_coverage": cov,
            "n_numeric_specs": len(_NUM_UNIT.findall(text)),
            "n_key_attributes": sum(bool(re.search(rx, text, re.I)) for rx in KEY_ATTRS.values()),
            "n_bullets": len(bullets), "has_description": int(len(desc) > 0), "title_len": len(title),
            "fluff_per_100": 100 * sum(text.count(f) for f in FLUFF) / n_words,
        })
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c != "exact"]
    X = StandardScaler().fit_transform(df[feats])
    lr = LogisticRegression(max_iter=1000).fit(X, df["exact"])
    coef = dict(zip(feats, lr.coef_[0].round(3).tolist()))
    df["coverage_bucket"] = pd.cut(df["query_term_coverage"], [-0.01, 0.25, 0.5, 0.75, 1.0], labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    by_bucket = df.groupby("coverage_bucket", observed=True)["exact"].agg(["mean", "size"]).round(3)
    out = {"n_pairs": int(len(df)), "exact_rate": float(df["exact"].mean()),
           "logit_coefficients_standardized": coef,
           "exact_rate_by_query_term_coverage": {str(k): {"exact_rate": float(v["mean"]), "n": int(v["size"])} for k, v in by_bucket.iterrows()},
           "narrative": "Covering the shopper's own words decides whether a product is judged relevant (shelf entry); "
                        "specs, attributes and fluff barely move the human relevance label. Which product the agent BUYS "
                        "among relevant ones has no public labels, so it must be simulated."}
    (path("results") / "relevance_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
