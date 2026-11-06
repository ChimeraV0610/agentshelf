"""Transparent offline stand-in for the LLM agent: term overlap + spec count + small seeded noise.

Only used when no Ollama model and no API key is available. Clearly labelled in the UI.
"""
from __future__ import annotations

import random
import re

from src.engine.audit import _NUM_UNIT

_WORD = re.compile(r"[a-z0-9]+")


def heuristic_rank(need: str, payload: list[dict], seed: int = 0) -> dict:
    rng = random.Random(seed)
    need_toks = set(_WORD.findall(need.lower()))
    scored = []
    for c in payload:
        text = " ".join([c["title"], c.get("bullets", ""), c.get("description", "")]).lower()
        toks = set(_WORD.findall(text))
        overlap = len(need_toks & toks) / max(len(need_toks), 1)
        specs = min(len(_NUM_UNIT.findall(text)), 8) / 8
        score = 0.7 * overlap + 0.2 * specs + 0.1 * rng.random()
        scored.append((score, c["cid"], overlap))
    scored.sort(reverse=True)
    ranking = [{"id": cid, "rank": i + 1, "reason": f"term overlap with the request {ov:.0%} (heuristic, not an LLM)", "missing_facts": []}
               for i, (_, cid, ov) in enumerate(scored)]
    return {"ok": True, "data": {"ranking": ranking, "pick": scored[0][1], "pick_reason": "highest heuristic score"}, "error": None}
