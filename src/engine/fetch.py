"""Turn an Amazon link / ASIN into listing fields, and find real competitors for it.

Order: (1) local ESCI product table (1.2M US products, offline, instant); (2) live fetch of
amazon.com (often blocked by bot detection, so it is best-effort); (3) caller falls back to paste.
"""
from __future__ import annotations

import re

import httpx
import pandas as pd

from src.config import path
from src.data.load import clean_text, load_processed, split_bullets
from src.engine.audit import buyer_vocab_from_queries
from src.engine.schemas import SIMULATE_SCHEMA  # noqa: F401  (keeps schema module imported for callers)

_ASIN_RE = re.compile(r"(?:/dp/|/gp/product/|/product/|asin=|^)([A-Z0-9]{10})(?:[/?&]|$)", re.I)


def parse_asin(text: str) -> str | None:
    text = text.strip()
    m = _ASIN_RE.search(text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(B0[A-Z0-9]{8})\b", text, re.I)
    return m.group(1).upper() if m else None


def _fields(title, brand, color, bullets, desc) -> dict:
    bullets = [clean_text(b) for b in bullets if clean_text(b)]
    return {"title": clean_text(title), "brand": clean_text(brand), "color": clean_text(color), "bullets_list": bullets,
            "bullets_full": " | ".join(bullets), "bullets": " | ".join(bullets)[:500],
            "description_full": clean_text(desc), "description": clean_text(desc)[:600]}


class Catalog:
    """Lazy handles on the local product table, ESCI judgements and the BM25 index."""

    def __init__(self):
        self.products = load_processed("products_us").set_index("product_id")
        self.examples = load_processed("examples_us")
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            from src.retrieval.bm25_index import BM25Index

            self._bm25 = BM25Index.load(path("indices") / "bm25")
        return self._bm25

    def lookup(self, asin: str) -> dict | None:
        if asin not in self.products.index:
            return None
        r = self.products.loc[asin]
        return _fields(r["product_title"], r.get("product_brand"), r.get("product_color"), split_bullets(r.get("product_bullet_point")), r.get("product_description"))

    def record(self, pid: str) -> dict:
        r = self.products.loc[pid]
        return {"product_id": pid, "title": clean_text(r["product_title"]), "brand": clean_text(r.get("product_brand")),
                "color": clean_text(r.get("product_color")), "bullets": split_bullets(r.get("product_bullet_point")),
                "description": clean_text(r.get("product_description"))}

    def competitors(self, asin: str | None, fields: dict, n: int = 7) -> tuple[list[dict], str, list[str]]:
        """Returns (competitor records, how they were found, real shopper queries used for vocabulary)."""
        if asin:
            mine = self.examples[(self.examples["product_id"] == asin) & (self.examples["esci_label"] == "E")]
            if len(mine):
                qids = mine["query_id"].tolist()
                pool = self.examples[self.examples["query_id"].isin(qids) & (self.examples["esci_label"] == "E") & (self.examples["product_id"] != asin)]
                counts = pool["product_id"].value_counts()
                pids = [p for p in counts.index if len(split_bullets(self.products.loc[p].get("product_bullet_point"))) >= 1][:n]
                queries = mine["query"].unique().tolist()
                if len(pids) >= 3:
                    return [self.record(p) for p in pids], f"products humans labelled Exact for the same {len(queries)} real search(es) as this ASIN", queries
        q = fields["title"]
        ids, _ = self.bm25.search([q], k=n * 6)
        if not hasattr(self, "_query_table"):
            self._query_table = self.examples.drop_duplicates("query_id")[["query_id", "query"]]
        pids, per_brand, seen_title = [], {}, set()
        for p in ids[0]:  # diversify: at most 2 per brand, no near-duplicate titles
            if p == asin:
                continue
            rec = self.products.loc[p]
            b = clean_text(rec.get("product_brand")).lower()
            key = re.sub(r"[^a-z0-9]", "", clean_text(rec["product_title"]).lower())[:30]
            if per_brand.get(b, 0) >= 2 or key in seen_title or len(split_bullets(rec.get("product_bullet_point"))) < 1:
                continue
            per_brand[b] = per_brand.get(b, 0) + 1
            seen_title.add(key)
            pids.append(p)
            if len(pids) == n:
                break
        recs = [self.record(p) for p in pids]
        # category terms = words shared by most competitor titles (and not brands) -> all real queries that use them
        from collections import Counter

        tok = Counter(w for r in recs for w in set(re.findall(r"[a-z]{3,}", r["title"].lower())))
        brands = {clean_text(r["brand"]).lower() for r in recs}
        generic = {"for", "with", "and", "the", "men", "women", "mens", "womens", "kids", "travel", "home", "new", "large", "small", "set", "pack", "black", "white"}
        cat_terms = [w for w, c in tok.most_common(12) if c >= max(3, len(recs) // 2) and w not in brands and w not in generic][:1]
        qt = getattr(self, "_query_table", self.examples.drop_duplicates("query_id")[["query_id", "query"]])
        mask = qt["query"].str.lower().str.contains("|".join(rf"\b{re.escape(t)}" for t in cat_terms), regex=True) if cat_terms else qt["query"].isin([])
        queries = qt.loc[mask, "query"].tolist()[:2000] or self.examples[self.examples["product_id"].isin(pids)]["query"].unique().tolist()
        return recs, f"closest listings among 1.2M US products (BM25 on the title); vocabulary from {len(queries):,} real searches for {', '.join(cat_terms) or 'similar items'}", queries

    _brand_tokens: set[str] | None = None

    def vocab(self, queries: list[str], fields: dict) -> list[str]:
        """Frequent shopper-query terms, minus brand names (which are noise for a seller's own listing)."""
        if not queries:
            return []
        if self._brand_tokens is None:
            brands = self.products["product_brand"].dropna().str.lower()
            self._brand_tokens = {b.strip() for b in brands.unique() if b.strip() and " " not in b.strip() and len(b.strip()) > 2}
            self._brand_tokens -= {"backpack", "bottle", "kids", "baby", "home", "pro", "plus", "mini", "smart", "light", "men", "women"}
        first = {b.split()[0] for b in self.products["product_brand"].dropna().str.lower().unique() if " " in b}
        return buyer_vocab_from_queries(queries, top_n=25, drop=self._brand_tokens | first, min_count=2)


def fetch_amazon(asin: str, timeout: float = 12.0) -> dict | None:
    """Best-effort live fetch. Returns None when blocked or when no title is found."""
    from bs4 import BeautifulSoup

    url = f"https://www.amazon.com/dp/{asin}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
               "Accept-Language": "en-US,en;q=0.9", "Accept": "text/html,application/xhtml+xml"}
    try:
        r = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200 or "captcha" in r.text[:5000].lower():
        return None
    soup = BeautifulSoup(r.text, "lxml")
    t = soup.select_one("#productTitle")
    if not t:
        return None
    bullets = [li.get_text(" ", strip=True) for li in soup.select("#feature-bullets li")]
    bullets = [b for b in bullets if b and "see more" not in b.lower()]
    desc = soup.select_one("#productDescription")
    brand = soup.select_one("#bylineInfo")
    brand_txt = re.sub(r"^(Visit the |Brand: )|( Store)$", "", brand.get_text(strip=True)) if brand else ""
    return _fields(t.get_text(strip=True), brand_txt, "", bullets, desc.get_text(" ", strip=True) if desc else "")


BUYERS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["buyers"],
                 "properties": {"buyers": {"type": "array", "items": {"type": "string"}}}}

BUYERS_SYSTEM = """You write realistic customer requests that a person would give to an AI shopping assistant.
Given a product listing, infer its product category, then write {n} different requests from different customers
shopping in that category. Each is 1-2 sentences in first person with concrete constraints (room size, body
size, device model, budget range, use case, must-haves, must-nots). Do NOT mention any brand. Vary the
constraints so that different products on the shelf could win. Return JSON only."""


def generate_buyers(client, model: str, fields: dict, n: int = 6) -> list[str]:
    user = f"Product listing:\nTitle: {fields['title']}\nBullets: {fields['bullets_full'][:800]}"
    res = client.call(model=model, system=BUYERS_SYSTEM.format(n=n), user=user, schema=BUYERS_SCHEMA, tag="buyers")
    if res.get("ok") and res.get("data", {}).get("buyers"):
        return [str(b) for b in res["data"]["buyers"]][:n]
    return []
