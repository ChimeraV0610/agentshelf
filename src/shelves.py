"""Build the six demo shelves from ESCI: one real query per category, eight real Exact products with full listings."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import CFG, path
from src.data.load import PRODUCTS_FILE, EXAMPLES_FILE, clean_text, split_bullets
from src.engine.audit import buyer_vocab_from_queries

# category -> (regex over queries for the vocabulary, the shelf query, department profile key)
CATEGORIES = {
    "air purifier": (r"\bair purifier", "air purifier", "Home & Kitchen"),
    "white noise machine": (r"(?:white noise|sound machine)", "best white noise machine for baby", "Baby & Other"),
    "baby carrier": (r"baby carrier", "baby carrier", "Baby & Other"),
    "bark collar": (r"bark collar", "bark collar", "Pet Supplies"),
    "1tb ssd": (r"(?:1\s?tb.*ssd|ssd.*1\s?tb)", "1tb ssd", "Electronics"),
    "backpack": (r"\bbackpack", "backpack", "Clothing, Shoes & Jewelry"),
}


def load_all_us_examples() -> pd.DataFrame:
    t = pq.read_table(path("raw") / EXAMPLES_FILE, filters=[("product_locale", "==", "us")],
                      columns=["query_id", "query", "product_id", "esci_label"])
    return t.to_pandas()


def product_record(pid: str, row) -> dict:
    return {"product_id": pid, "title": clean_text(row["product_title"]), "brand": clean_text(row.get("product_brand")),
            "color": clean_text(row.get("product_color")), "bullets": split_bullets(row.get("product_bullet_point")),
            "description": clean_text(row.get("product_description"))}


def build_shelves(n_products: int = 8, seed: int | None = None) -> dict:
    seed = seed if seed is not None else CFG["seed"]
    ex = load_all_us_examples()
    prod = pd.read_parquet(path("processed") / "products_us.parquet").set_index("product_id")
    q = ex.drop_duplicates("query_id")[["query_id", "query"]]
    shelves = {}
    for cat, (rx, shelf_query, dept) in CATEGORIES.items():
        rng = np.random.default_rng(seed)
        matched = q[q["query"].str.contains(rx, case=False, regex=True)]
        vocab = buyer_vocab_from_queries(matched["query"].tolist(), top_n=25)
        qids = q[q["query"].str.lower() == shelf_query]["query_id"].tolist()
        rows = ex[(ex["query_id"].isin(qids)) & (ex["esci_label"] == "E")].drop_duplicates("product_id")
        recs = [product_record(pid, prod.loc[pid]) for pid in rows["product_id"] if pid in prod.index]
        recs = [r for r in recs if len(r["bullets"]) >= 3 and len(r["title"]) >= 40]  # drop stubs / non-products
        rng.shuffle(recs)
        chosen, seen = [], set()
        for r in recs:  # avoid near-duplicate listings (same model in two colours / sizes)
            key = re.sub(r"[^a-z0-9]", "", r["title"].lower())[:30]
            if key in seen:
                continue
            seen.add(key)
            chosen.append(r)
            if len(chosen) == n_products:
                break
        shelves[cat] = {"query": shelf_query, "department": dept, "n_real_queries": int(len(matched)),
                        "buyer_vocab": vocab, "products": chosen}
    return shelves


def save_shelves(shelves: dict) -> None:
    (path("root") if "root" in CFG["paths"] else path("processed").parent).joinpath("shelves.json").write_text(
        json.dumps(shelves, indent=1, ensure_ascii=False))


def load_shelves() -> dict:
    return json.loads((path("processed").parent / "shelves.json").read_text())


def load_buyers() -> dict:
    return json.loads((path("processed").parent / "buyers.json").read_text())
