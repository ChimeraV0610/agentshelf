"""Agent-friendliness audit: 0-100, five transparent rule-based dimensions, plus concrete issues."""
from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")
_NUM_UNIT = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|inch|inches|in\b|\"|cm|mm|ft|feet|lbs?|lb\b|oz|kg|g\b|ml|l\b|liters?|gal|w\b|watts?|v\b|volts?|mah|hz|db|hours?|hrs?|minutes?|mins?|days?|months?|years?|yr|sq\s?ft|square feet|mb/s|gb|tb|pcs|pack|count|ct\b|x\b)", re.I)
FLUFF = ["best", "premium", "amazing", "perfect", "ultimate", "top rated", "#1", "number one", "high quality", "high-quality",
         "great", "excellent", "superior", "incredible", "must have", "must-have", "elegant", "beautiful", "luxury", "unique",
         "world-class", "revolutionary", "stunning", "awesome", "fantastic", "wonderful", "love", "satisfaction guaranteed"]
KEY_ATTRS = {
    "size / dimensions": r"\b(\d+(?:\.\d+)?\s?(?:inch|inches|in\b|\"|cm|mm|ft)|dimensions?|\bsize\b|\bx\s?\d)",
    "weight": r"\b(\d+(?:\.\d+)?\s?(?:lbs?|lb\b|oz|kg|g\b|grams?|pounds?|ounces?)|weight|weighs)",
    "material": r"\b(material|stainless|steel|aluminum|plastic|cotton|polyester|nylon|leather|silicone|wood|bamboo|glass|abs\b|mesh|rubber|fabric)",
    "compatibility": r"\b(compatible|compatibility|fits?|works with|for (?:iphone|samsung|ps5|xbox|macbook|laptop|ipad))",
    "warranty": r"\b(warranty|guarantee|guaranteed)\b",
    "certification": r"\b(certified|certification|ul listed|etl|fcc|ce\b|rohs|fda|bpa[- ]free|energy star|cadr|hepa|h13|ipx?\d|ip\d\d|astm|cpsc|jpma)",
    "package contents": r"\b(package includes?|what'?s in the box|in the box|includes?:|comes with|set of|pack of|\d+\s?(?:pcs|pieces|count|pack))",
}


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def audit_listing(fields: dict, buyer_vocab: list[str] | None = None) -> dict:
    """fields: title, brand, color, bullets_list, description_full.

    buyer_vocab: frequent terms from real shopper queries for this category (not the simulated buyers).
    Returns {'score': int, 'dimensions': {name: {'score': 0-20, 'detail': str}}, 'issues': [str]}.
    """
    title = fields.get("title", "")
    bullets = fields.get("bullets_list", []) or []
    desc = fields.get("description_full", "") or ""
    text = " ".join([title] + bullets + [desc])
    toks = _tokens(text)
    n_words = max(len(toks), 1)
    issues: list[str] = []
    dims: dict[str, dict] = {}

    # 1 buyer vocabulary
    if buyer_vocab:
        present = [w for w in buyer_vocab if w in set(toks)]
        cov = len(present) / len(buyer_vocab)
        missing = [w for w in buyer_vocab if w not in set(toks)][:8]
        dims["buyer vocabulary"] = {"score": round(20 * cov), "detail": f"uses {len(present)}/{len(buyer_vocab)} of the words real shoppers search with"}
        if cov < 0.5:
            issues.append("Shoppers search with words the listing never uses: " + ", ".join(missing))
    else:
        dims["buyer vocabulary"] = {"score": 10, "detail": "no category vocabulary available"}

    # 2 comparable numeric specs
    specs = _NUM_UNIT.findall(text)
    n_specs = len(specs)
    dims["numeric specs"] = {"score": min(20, 4 * n_specs), "detail": f"{n_specs} number+unit specs an agent can compare"}
    if n_specs < 3:
        issues.append("Fewer than 3 comparable numeric specs (size, weight, capacity, runtime, coverage...)")

    # 3 key attributes agents filter on
    found = {k: bool(re.search(rx, text, re.I)) for k, rx in KEY_ATTRS.items()}
    n_found = sum(found.values())
    dims["key attributes"] = {"score": round(20 * n_found / len(KEY_ATTRS)), "detail": "states " + ", ".join(k for k, v in found.items() if v) if n_found else "states none"}
    lacking = [k for k, v in found.items() if not v]
    if lacking:
        issues.append("Does not state: " + ", ".join(lacking))

    # 4 structure
    s = 0
    if 60 <= len(title) <= 200:
        s += 7
    elif title:
        s += 3
        issues.append(f"Title length {len(title)} chars (aim for 60-200 with product type and decisive specs first)")
    if len(bullets) >= 5:
        s += 7
    else:
        s += round(7 * len(bullets) / 5)
        issues.append(f"Only {len(bullets)} bullet points (aim for 5+, each 'Attribute: fact')")
    if len(desc) >= 200:
        s += 6
    elif desc:
        s += 3
    else:
        issues.append("No text description found (image-only A+ content is invisible to agents too)")
    dims["structure"] = {"score": s, "detail": f"title {len(title)} chars, {len(bullets)} bullets, description {len(desc)} chars"}

    # 5 fluff share (lower is better)
    low = text.lower()
    fluff_hits = sum(low.count(f) for f in FLUFF)
    fluff_per_100 = 100 * fluff_hits / n_words
    dims["low fluff"] = {"score": max(0, round(20 - 5 * fluff_per_100)), "detail": f"{fluff_hits} marketing words per {n_words} words"}
    if fluff_per_100 > 2:
        issues.append("Marketing filler that agents ignore: " + ", ".join(sorted({f for f in FLUFF if f in low})[:6]))

    score = int(sum(d["score"] for d in dims.values()))
    return {"score": score, "dimensions": dims, "issues": issues}


def buyer_vocab_from_queries(queries: list[str], top_n: int = 25, drop: set[str] | None = None) -> list[str]:
    stop = {"for", "the", "and", "with", "a", "an", "of", "to", "in", "on", "my", "best", "new", "cheap"} | (drop or set())
    c = Counter(w for q in queries for w in set(_tokens(q)) if w not in stop and len(w) > 1)
    return [w for w, _ in c.most_common(top_n)]
