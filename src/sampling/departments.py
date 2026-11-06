"""Department tagging for queries: LLM (preferred) or keyword fallback (no API needed)."""
from __future__ import annotations

import re

import pandas as pd

from src.agent.prompts import render_department_system, render_department_user
from src.agent.schema import DEPARTMENT_SCHEMA
from src.config import CFG
from src.llm.client import LLMClient, get_client

DEPTS = CFG["departments"]
FALLBACK = DEPTS[-1]

# crude keyword rules, used only when no API key is available or to sanity-check the LLM
_KEYWORDS = {
    "Electronics": r"\b(phone|iphone|samsung|laptop|usb|hdmi|charger|cable|headphone|earbud|bluetooth|speaker|tv|camera|ssd|router|wifi|monitor|keyboard|mouse|tablet|ipad|battery|case for|screen protector|airpod|playstation|xbox|nintendo|switch)\b",
    "Home & Kitchen": r"\b(kitchen|pan|pot|knife|mug|cup|plate|bowl|pillow|blanket|sheet|curtain|rug|lamp|sofa|chair|table|shelf|storage|organizer|vacuum|cleaner|towel|bath|shower|coffee maker|blender|air fryer|mattress|bedding|decor)\b",
    "Clothing, Shoes & Jewelry": r"\b(shirt|t-shirt|dress|jeans|pants|shorts|jacket|coat|hoodie|sweater|shoes|sneaker|boot|sandal|socks|bra|underwear|leggings|hat|cap|belt|watch|necklace|earring|ring|bracelet|purse|wallet|backpack|womens|mens|women's|men's)\b",
    "Beauty & Personal Care": r"\b(shampoo|conditioner|lotion|cream|serum|makeup|lipstick|mascara|nail|perfume|cologne|hair|skin|face|moisturizer|sunscreen|razor|deodorant|brush|eyeliner|foundation|wig)\b",
    "Health & Household": r"\b(vitamin|supplement|protein|probiotic|mask|gloves|thermometer|first aid|bandage|toilet paper|detergent|laundry|tissue|soap|sanitizer|medicine|pain relief|collagen|omega)\b",
    "Grocery": r"\b(coffee|tea|snack|candy|chocolate|cereal|sauce|spice|seasoning|oil|rice|pasta|flour|sugar|honey|nuts|chips|cookies|drink|juice|water bottle case|gummies|organic|keto|gluten free)\b",
    "Toys & Games": r"\b(toy|lego|doll|puzzle|board game|card game|plush|figure|playset|slime|kids game|rc car|nerf|barbie|pokemon)\b",
    "Sports & Outdoors": r"\b(yoga|dumbbell|bike|bicycle|tent|camping|fishing|hiking|golf|ball|fitness|gym|treadmill|resistance band|kayak|swim|goggles|skate|hunting|workout)\b",
    "Tools, Automotive & Garden": r"\b(drill|screwdriver|wrench|saw|tool|car|truck|tire|wiper|led bulb|flashlight|garden|plant|seed|hose|lawn|mower|grill|outdoor light|battery charger car|motorcycle|jeep|ford|toyota|honda)\b",
    "Office, Books & Media": r"\b(book|novel|notebook|pen|pencil|paper|printer|ink|stapler|planner|calendar|desk|dvd|blu-ray|vinyl|cd|journal|sticker|marker|binder)\b",
    "Pet Supplies": r"\b(dog|cat|puppy|kitten|pet|aquarium|fish tank|bird|hamster|leash|collar|litter|chew)\b",
    "Baby & Other": r"\b(baby|infant|toddler|diaper|stroller|crib|pacifier|bottle warmer|onesie)\b",
}
_COMPILED = {d: re.compile(p, re.I) for d, p in _KEYWORDS.items()}


def tag_keyword(queries: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in queries.itertuples(index=False):
        best, best_n = FALLBACK, 0
        for d, rx in _COMPILED.items():
            n = len(rx.findall(r.query))
            if n > best_n:
                best, best_n = d, n
        out.append({"query_id": r.query_id, "department": best, "method": "keyword"})
    return pd.DataFrame(out)


def tag_llm(queries: pd.DataFrame, model: str, batch: int = 50) -> pd.DataFrame:
    client = get_client()
    system = render_department_system()
    rows = list(zip(queries["query_id"].astype(int).tolist(), queries["query"].tolist()))
    jobs = [{"model": model, "system": system, "user": render_department_user(rows[i:i + batch]),
             "schema": DEPARTMENT_SCHEMA, "tag": "dept"} for i in range(0, len(rows), batch)]
    results = client.map(jobs, desc="tag departments")
    print(client.summary())
    tagged: dict[int, str] = {}
    for res in results:
        if res.get("ok") and res.get("data"):
            for item in res["data"].get("items", []):
                d = item.get("department")
                tagged[int(item["query_id"])] = d if d in DEPTS else FALLBACK
    kw = tag_keyword(queries).set_index("query_id")["department"]
    out = pd.DataFrame({"query_id": queries["query_id"].astype(int)})
    out["department"] = [tagged.get(q, kw[q]) for q in out["query_id"]]
    out["method"] = ["llm" if q in tagged else "keyword" for q in out["query_id"]]
    return out
