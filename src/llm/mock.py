"""Offline stand-in for LLMClient (AGENTSHELF_MOCK=1): random but schema-valid answers.

Lets every script and the Streamlit app run end to end without API keys. Results
produced this way are meaningless and are tagged mock=True in usage output.
"""
from __future__ import annotations

import random
import re

from src.agent.schema import REASON_CODES
from src.config import CFG


class MockClient:
    budget = 0.0
    spent = 0.0

    def __init__(self, *_, **__):
        self.stats = {"cached": 0, "calls": 0, "errors": 0}

    def _answer(self, job: dict) -> dict:
        rng = random.Random(hash(job["user"]) & 0xFFFFFFFF)
        schema = job.get("schema") or {}
        props = schema.get("properties", {})
        if "judgments" in props:
            cids = re.findall(r"^\[(c\d{2})\]", job["user"], flags=re.M)
            js = [{"id": c, "label": rng.choices(["E", "S", "C", "I"], [0.45, 0.3, 0.05, 0.2])[0],
                   "confidence": round(rng.uniform(0.4, 1.0), 2),
                   "missing_attributes": rng.sample(["size", "color", "compatibility", "quantity", "material"], k=rng.randint(0, 2)),
                   "reason_code": rng.choice(REASON_CODES[1:])} for c in cids]
            es = [j["id"] for j in js if j["label"] == "E"]
            pick = rng.choice(es) if es and rng.random() > 0.1 else None
            for j in js:
                if j["id"] == pick:
                    j["reason_code"] = "best_match"
            return {"judgments": js, "pick": pick, "pick_reason": "mock pick"}
        if "items" in props:
            qids = [int(l.split("\t")[0]) for l in job["user"].splitlines() if "\t" in l]
            return {"items": [{"query_id": q, "department": rng.choice(CFG["departments"])} for q in qids]}
        if "bullets" in props and "title" in props:
            title = re.search(r"Title: (.*)", job["user"]).group(1)
            return {"title": title[: CFG["rewrite"]["title_max_chars"]],
                    "bullets": [{"text": f"Mock bullet {i+1} about {title[:30]}", "evidence": title[:30]}
                                for i in range(CFG["rewrite"]["n_bullets"])],
                    "attributes_not_available": ["size", "material"]}
        if "claims" in props:
            return {"claims": [{"text": "mock claim", "verdict": "SUPPORTED", "evidence": "mock"}],
                    "overall_pass": True, "stuffing_flag": False}
        if "profile" in props:
            return {"profile": "Shoppers specify size, material, pack quantity, compatibility and intended user."}
        return {}

    def call(self, **job) -> dict:
        self.stats["calls"] += 1
        return {"ok": True, "data": self._answer(job), "error": None, "cached": False, "mock": True}

    def map(self, jobs: list[dict], desc: str = "") -> list[dict]:
        return [self.call(**j) for j in jobs]

    def summary(self) -> str:
        return f"MOCK client: calls={self.stats['calls']} (no API used)"
