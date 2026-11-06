"""Step 0: load ESCI (US, small), clean, write processed tables and a data profile."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import CFG, path  # noqa: E402
from src.data.load import build_doc_text, clean_text, load_examples_us, load_products_us  # noqa: E402


def main():
    out = path("processed")
    ex = load_examples_us()
    print(f"examples_us: {len(ex):,} rows, {ex['query_id'].nunique():,} queries")
    ex.to_parquet(out / "examples_us.parquet", index=False)

    cols = ["product_id", "product_title", "product_description", "product_bullet_point",
            "product_brand", "product_color"]
    prod = load_products_us(columns=cols)
    prod = prod.drop_duplicates("product_id").reset_index(drop=True)
    print(f"products_us (all): {len(prod):,}")
    small_ids = set(ex["product_id"])
    prod["in_small"] = prod["product_id"].isin(small_ids)
    print(f"products referenced by small US examples: {prod['in_small'].sum():,}")
    prod.to_parquet(out / "products_us.parquet", index=False)

    if CFG["corpus"] == "us_small":
        corpus_df = prod[prod["in_small"]]
    else:
        corpus_df = prod
    texts = [build_doc_text(r) for r in corpus_df.to_dict("records")]
    corpus = pd.DataFrame({"product_id": corpus_df["product_id"].values, "doc_text": texts})
    corpus.to_parquet(out / "corpus.parquet", index=False)
    print(f"corpus ({CFG['corpus']}): {len(corpus):,} docs")

    # ---- profile ----
    dist = ex["esci_label"].value_counts(normalize=True).round(4)
    per_q = ex.groupby("query_id").size()
    splits = ex.drop_duplicates("query_id")["split"].value_counts()
    small = prod[prod["in_small"]]
    empties = {c: float((small[c].map(clean_text) == "").mean()) for c in
               ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]}
    lines = [
        "# Data profile (ESCI, small_version==1, product_locale=='us')", "",
        f"- judgements: {len(ex):,}", f"- queries: {ex['query_id'].nunique():,}",
        f"- products referenced: {len(small_ids):,}", f"- products in US table (all): {len(prod):,}",
        f"- corpus used for retrieval: {CFG['corpus']} = {len(corpus):,} docs", "",
        "## Label distribution", "", dist.to_string(), "",
        "## Candidates per query", "", per_q.describe().round(2).to_string(), "",
        "## Queries per split", "", splits.to_string(), "",
        "## Empty-field rate among referenced products", "",
        *[f"- {k}: {v:.3f}" for k, v in empties.items()],
    ]
    rep = path("reports") / "data_profile.md"
    rep.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
