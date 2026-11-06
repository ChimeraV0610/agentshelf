"""Load and clean the ESCI dataset (US, small version)."""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import CFG, path

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"


def load_examples_us(split: str | None = None) -> pd.DataFrame:
    filters = [("small_version", "==", 1), ("product_locale", "==", "us")]
    if split:
        filters.append(("split", "==", split))
    table = pq.read_table(path("raw") / EXAMPLES_FILE, filters=filters)
    df = table.to_pandas()
    return df[["example_id", "query", "query_id", "product_id", "esci_label", "split"]]


def load_products_us(columns: list[str] | None = None) -> pd.DataFrame:
    table = pq.read_table(
        path("raw") / PRODUCTS_FILE,
        filters=[("product_locale", "==", "us")],
        columns=columns,
    )
    return table.to_pandas()


def clean_text(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = html.unescape(str(s))
    s = _TAG_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def split_bullets(s) -> list[str]:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return [clean_text(b) for b in s.split("\n") if clean_text(b)]


def build_doc_text(row, n_bullets: int | None = None, max_chars: int | None = None) -> str:
    """Text used for retrieval: title + brand + color + first bullets."""
    n_bullets = n_bullets or CFG["doc_text"]["n_bullets"]
    max_chars = max_chars or CFG["doc_text"]["max_chars"]
    parts = [clean_text(row["product_title"])]
    brand = clean_text(row.get("product_brand"))
    color = clean_text(row.get("product_color"))
    if brand:
        parts.append(f"Brand: {brand}")
    if color:
        parts.append(f"Color: {color}")
    bullets = split_bullets(row.get("product_bullet_point"))[:n_bullets]
    parts.extend(bullets)
    return " . ".join(p for p in parts if p)[:max_chars]


def build_listing_fields(row, desc_chars: int = 600, bullet_chars: int = 400) -> dict:
    """Fields shown to the LLM agent for one candidate."""
    bullets = split_bullets(row.get("product_bullet_point"))
    joined = " | ".join(bullets)
    return {
        "title": clean_text(row.get("product_title")),
        "brand": clean_text(row.get("product_brand")),
        "color": clean_text(row.get("product_color")),
        "bullets": joined[:bullet_chars],
        "description": clean_text(row.get("product_description"))[:desc_chars],
    }


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(path("processed") / f"{name}.parquet")
