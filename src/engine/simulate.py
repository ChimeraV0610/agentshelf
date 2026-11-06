"""Simulated AI buyers: each need statement -> one agent call that ranks the whole shelf."""
from __future__ import annotations

import time

import numpy as np

from src.engine.heuristic import heuristic_rank
from src.engine.schemas import SIMULATE_SCHEMA

USER_PID = "__MINE__"

SIMULATE_SYSTEM = """You are an AI shopping agent buying on behalf of a customer. You can only read the product
listings given to you: no images, no reviews, no brand reputation, no outside knowledge. Judge each product
strictly on the facts stated in its listing. If a listing does not state a fact the customer needs, treat that
fact as unknown and count it against the product.

Rank ALL candidates from 1 (best fit for this customer) to N. For each candidate give one sentence citing
listing facts, and list at most 2 facts the customer needs that this listing does not state. Then name the one
you would buy. Respond with JSON only."""


def render_user(need: str, payload: list[dict]) -> str:
    lines = [f"CUSTOMER REQUEST: {need}", "", f"CANDIDATES ({len(payload)}, random order):"]
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
    per = []
    n_items = len(products) + 1
    for b, need in enumerate(buyers):
        t0 = time.time()
        payload = make_payload(products, mine, seed + b)
        cid_to_pid = {c["cid"]: c["product_id"] for c in payload}
        title_by_cid = {c["cid"]: c["title"] for c in payload}
        if model == "heuristic":
            res = heuristic_rank(need, payload, seed + b)
        else:
            res = client.call(model=model, system=SIMULATE_SYSTEM, user=render_user(need, payload), schema=SIMULATE_SCHEMA, tag=f"{tag}:{b}")
        row = {"need": need, "rank": None, "picked": False, "reason": None, "missing_facts": [], "picked_instead": None,
               "error": res.get("error"), "seconds": round(time.time() - t0, 1), "n": n_items, "ranking": []}
        d = res.get("data") if res.get("ok") else None
        if d:
            ranking = sorted([r for r in d.get("ranking", []) if r.get("id") in cid_to_pid], key=lambda r: r.get("rank", 99))
            # normalise ranks to 1..k in listed order (models sometimes skip numbers)
            for i, r in enumerate(ranking, start=1):
                r["rank"] = i
            row["ranking"] = [{"product_id": cid_to_pid[r["id"]], "title": title_by_cid[r["id"]][:90], "rank": r["rank"], "reason": r.get("reason", "")} for r in ranking]
            mine_r = next((r for r in ranking if cid_to_pid[r["id"]] == USER_PID), None)
            if mine_r:
                row.update({"rank": mine_r["rank"], "reason": mine_r.get("reason", ""), "missing_facts": list(mine_r.get("missing_facts") or [])[:2]})
            pick = d.get("pick")
            if pick in cid_to_pid:
                row["picked"] = cid_to_pid[pick] == USER_PID
                if not row["picked"]:
                    row["picked_instead"] = title_by_cid[pick][:90]
        per.append(row)
        if progress:
            progress(b + 1, len(buyers), row)
    ranked = [r for r in per if r["rank"] is not None]
    out = {
        "per_buyer": per, "n": n_items, "n_buyers": len(buyers), "n_ok": len(ranked),
        "pick_rate": float(np.mean([r["picked"] for r in per])) if per else float("nan"),
        "fair_share": 1.0 / n_items,
        "top3_rate": float(np.mean([r["rank"] <= 3 for r in ranked])) if ranked else float("nan"),
        "avg_rank": float(np.mean([r["rank"] for r in ranked])) if ranked else float("nan"),
        "reasons": [r["reason"] for r in per if r["reason"]],
        "missing_facts": sorted({m.strip().lower() for r in per for m in r["missing_facts"] if m.strip()}),
        "seconds": float(sum(r["seconds"] for r in per)),
    }
    return out
