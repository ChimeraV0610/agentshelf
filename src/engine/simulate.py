"""Extract a buyer’s stated criteria, then rank the shelf against those criteria."""
from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy

import numpy as np

from src.engine.schemas import BUYER_CONSTRAINTS_SCHEMA, SIMULATE_SCHEMA

USER_PID = "__MINE__"

EXTRACT_SYSTEM = """Extract shopping criteria from the buyer request only. Return two arrays:
hard_constraints (required product type, explicit must-haves, exclusions, compatibility,
and firm limits) and preferences (stated nice-to-haves and softer goals).
Each entry must be an exact, contiguous quote copied verbatim, with the original case
and punctuation. Keep qualifying words and negation. Do not infer or paraphrase.
Use an empty array when there are no stated criteria in that group. Return JSON only."""

SIMULATE_SYSTEM = """You are a shopping agent. Use only facts in the supplied listings.
Evaluate only the supplied hard constraints and preferences. Do not add criteria.
Check each hard constraint against each candidate independently. Compare exact
model numbers, connector types, form factors and dimensions. A different identifier
or incompatible size is a conflict even if the product matches every other criterion.
Never mark a candidate met merely because it is the closest option.

For every candidate report hard_constraint_status: conflict if any hard constraint
is contradicted; unknown if none is contradicted but any is unverified; met if all
are supported or there are no hard constraints. Rank candidates with met first,
ordering them by fit to the preferences. Include each candidate exactly once.
Give a reason of at most 12 words grounded in the listing and up to two missing requested facts.
Only met candidates can be bought; if none qualify, there is no purchase. Return JSON only."""


def extract_criteria(client, model: str, need: str) -> tuple[dict, str | None]:
    result = client.call(model=model, system=EXTRACT_SYSTEM, user=need,
                         schema=BUYER_CONSTRAINTS_SCHEMA, tag="buyer-criteria", max_tokens=450)
    if not result.get("ok"):
        return {}, result.get("error") or "buyer criteria extraction failed"

    data = result.get("data") or {}
    criteria = {}
    for key in ("hard_constraints", "preferences"):
        quotes = data.get(key)
        if not isinstance(quotes, list) or any(
            not isinstance(quote, str) or not quote.strip() or quote not in need for quote in quotes
        ):
            return {}, f"buyer criteria must be quoted verbatim: {key}"
        criteria[key] = list(dict.fromkeys(quotes))
    if not any(criteria.values()):
        return {}, "no stated buyer criteria extracted"
    if set(criteria["hard_constraints"]) & set(criteria["preferences"]):
        return {}, "buyer criteria cannot be both required and preferred"
    return criteria, None


_M2_FORM_FACTOR = re.compile(r"\bM\.2\s*(22(?:30|42|60|80)|22110)\b", re.IGNORECASE)


def verify_m2_form_factor(ranking: list[dict], payload: list[dict], hard_constraints: list[str]) -> None:
    """Exact M.2 size requirements are checkable from the visible listing text."""
    size_clauses = [quote for quote in hard_constraints if _M2_FORM_FACTOR.search(quote)]
    if any(re.search(r"\b(?:no|not|without|avoid|except|or)\b", quote, re.IGNORECASE)
           for quote in size_clauses):
        return
    required = set(_M2_FORM_FACTOR.findall(" ".join(size_clauses)))
    if len(required) != 1:
        return
    size = next(iter(required))
    listings = {candidate["cid"]: candidate for candidate in payload}
    for row in ranking:
        if row["hard_constraint_status"] != "met":
            continue
        listing = listings[row["id"]]
        text = " ".join(listing.get(field, "") for field in ("title", "bullets", "description"))
        stated = set(_M2_FORM_FACTOR.findall(text))
        if size not in stated:
            row["hard_constraint_status"] = "conflict" if stated else "unknown"
            row["reason"] = (f"Listing states M.2 {', '.join(sorted(stated))}; buyer requires M.2 {size}."
                             if stated else f"Listing does not verify M.2 {size}.")


def render_user(criteria: dict, payload: list[dict]) -> str:
    lines = [f"BUYER CRITERIA: {json.dumps(criteria, ensure_ascii=False)}",
             "", f"CANDIDATES ({len(payload)}, random order):"]
    for c in payload:
        lines.append(f"[{c['cid']}] {c['title']}")
        lines.append(f"      Brand: {c.get('brand') or 'n/a'} | Color: {c.get('color') or 'n/a'}")
        if c.get("bullets"):
            lines.append(f"      Bullets: {c['bullets']}")
        if c.get("description"):
            lines.append(f"      Description: {c['description']}")
    return "\n".join(lines)


def make_payload(products: list[dict], mine: dict, seed: int, bullet_chars: int = 500, desc_chars: int = 300) -> list[dict]:
    """products: competitor dicts with product_id,title,brand,color,bullets(list),description. mine: fields dict."""
    rng = np.random.default_rng(seed)
    items = [{"product_id": p["product_id"], "title": p["title"], "brand": p.get("brand", ""), "color": p.get("color", ""),
              "bullets": " | ".join(p.get("bullets", []))[:bullet_chars], "description": (p.get("description") or "")[:desc_chars]}
             for p in products]
    items.append({"product_id": USER_PID, "title": mine["title"], "brand": mine.get("brand", ""), "color": mine.get("color", ""),
                  "bullets": " | ".join(mine.get("bullets_list", []))[:bullet_chars], "description": (mine.get("description_full") or "")[:desc_chars]})
    order = rng.permutation(len(items))
    return [{"cid": f"c{i:02d}", **items[j]} for i, j in enumerate(order, start=1)]


def simulate(client, model: str, products: list[dict], mine: dict, buyers: list[str], seed: int = 42,
             tag: str = "sim", progress=None) -> dict:
    """Returns {'per_buyer': [...], 'pick_rate', 'fair_share', 'top3_rate', 'avg_rank', 'n', 'reasons', 'missing_facts'}."""
    if not model.startswith("ollama:") or os.environ.get("AGENTSHELF_MOCK") == "1":
        raise ValueError("Buyer simulation requires a local Ollama model with mock mode disabled.")
    per = []
    n_items = len(products) + 1
    schema = deepcopy(SIMULATE_SCHEMA)
    ranking_schema = schema["properties"]["ranking"]
    ranking_schema.update(minItems=n_items, maxItems=n_items)
    ranking_schema["items"]["properties"]["id"]["enum"] = [f"c{i:02d}" for i in range(1, n_items + 1)]
    for b, need in enumerate(buyers):
        t0 = time.time()
        payload = make_payload(products, mine, seed + b)
        cid_to_pid = {c["cid"]: c["product_id"] for c in payload}
        title_by_cid = {c["cid"]: c["title"] for c in payload}
        criteria, error = extract_criteria(client, model, need)
        if error:
            res = {"ok": False, "error": error}
        else:
            res = client.call(model=model, system=SIMULATE_SYSTEM, user=render_user(criteria, payload),
                              schema=schema, tag=f"{tag}:{b}", max_tokens=2000)
        row = {"need": need, "rank": None, "picked": False, "reason": None, "missing_facts": [], "picked_instead": None,
               "hard_constraint_status": None, "hard_constraints": [], "preferences": [], "no_purchase": False, "omitted": False,
               "error": res.get("error"), "seconds": round(time.time() - t0, 1), "n": n_items, "ranking": []}
        d = (res.get("data") or {}) if res.get("ok") else {}
        row["hard_constraints"] = criteria.get("hard_constraints", [])
        row["preferences"] = criteria.get("preferences", [])
        ranking = d.get("ranking", [])
        ids = [r.get("id") for r in ranking]
        valid = (len(ids) == len(set(ids)) and set(ids) <= cid_to_pid.keys()
                 and all(r.get("hard_constraint_status") in {"met", "conflict", "unknown"}
                         and type(r.get("rank")) is int for r in ranking))
        if not res.get("ok") or not valid or not ranking:
            row["error"] = row["error"] or "invalid candidate ranking"
        else:
            verify_m2_form_factor(ranking, payload, row["hard_constraints"])
            ranking = sorted(ranking, key=lambda r: (r["hard_constraint_status"] != "met", r["rank"]))
            for i, r in enumerate(ranking, start=1):
                cid = r["id"]
                row["ranking"].append({"product_id": cid_to_pid[cid], "title": title_by_cid[cid][:90],
                                       "rank": i, "reason": r.get("reason", ""),
                                       "hard_constraint_status": r["hard_constraint_status"]})
                if cid_to_pid[cid] == USER_PID:
                    row.update(rank=i, reason=r.get("reason"), hard_constraint_status=r["hard_constraint_status"],
                               missing_facts=list(r.get("missing_facts") or [])[:2])
            if row["rank"] is None:
                row.update(rank=n_items, omitted=True, reason="agent did not rank this listing")
            if len(ids) != n_items:
                row["error"] = "incomplete candidate ranking"
            elif ranking[0]["hard_constraint_status"] != "met":
                row["no_purchase"] = True
            else:
                pick = ranking[0]["id"]
                row["picked"] = cid_to_pid[pick] == USER_PID
                if not row["picked"]:
                    row["picked_instead"] = title_by_cid[pick][:90]
        per.append(row)
        if progress:
            progress(b + 1, len(buyers), row)
    ranked = [r for r in per if not r["error"]]
    out = {
        "per_buyer": per, "n": n_items, "n_buyers": len(buyers), "n_ok": len(ranked),
        "pick_rate": float(np.mean([r["picked"] for r in ranked])) if ranked else float("nan"),
        "fair_share": 1.0 / n_items,
        "top3_rate": float(np.mean([r["rank"] <= 3 for r in ranked])) if ranked else float("nan"),
        "avg_rank": float(np.mean([r["rank"] for r in ranked])) if ranked else float("nan"),
        "no_purchase_count": sum(r["no_purchase"] for r in ranked),
        "omitted_count": int(sum(r["omitted"] for r in per)),
        "reasons": [r["reason"] for r in per if r["reason"]],
        "missing_facts": sorted({m.strip().lower() for r in per for m in r["missing_facts"] if m.strip()}),
        "seconds": float(sum(r["seconds"] for r in per)),
    }
    return out
