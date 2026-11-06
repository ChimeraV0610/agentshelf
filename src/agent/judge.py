"""Run the shopping-agent judgment for one query over its candidate list."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.agent.prompts import JUDGE_SYSTEM, render_judge_user
from src.agent.schema import JUDGE_SCHEMA, REASON_CODES
from src.config import CFG
from src.data.load import build_listing_fields

VALID_LABELS = {"E", "S", "C", "I"}


def select_for_prompt(cands: pd.DataFrame, seed: int, max_n: int | None = None) -> pd.DataFrame:
    """Cap the candidate list: keep all E, fill with a seeded random subset of the rest."""
    max_n = max_n or CFG["sampling"]["max_candidates_in_prompt"]
    if len(cands) <= max_n:
        out = cands.copy()
        out["in_prompt"] = True
        return out
    rng = np.random.default_rng(seed)
    e = cands[cands["esci_label"] == "E"]
    rest = cands[cands["esci_label"] != "E"]
    n_rest = max(0, max_n - len(e))
    keep_rest = rest.sample(n=min(n_rest, len(rest)), random_state=int(rng.integers(0, 2**31)))
    keep_ids = set(e["product_id"]) | set(keep_rest["product_id"])
    out = cands.copy()
    out["in_prompt"] = out["product_id"].isin(keep_ids)
    return out


def make_candidate_payload(cands: pd.DataFrame, products_by_id: pd.DataFrame,
                           override_text: dict[str, dict] | None = None) -> list[dict]:
    """cands must be sorted by shuffled_pos and filtered to in_prompt. Returns prompt-ready dicts.

    override_text: {product_id: {'title':..., 'bullets':...}} to swap in a rewritten listing.
    """
    payload = []
    for i, pid in enumerate(cands["product_id"].tolist(), start=1):
        row = products_by_id.loc[pid]
        fields = build_listing_fields(row)
        if override_text and pid in override_text:
            fields = {**fields, **override_text[pid]}
        payload.append({"cid": f"c{i:02d}", "product_id": pid, **fields})
    return payload


def judge_job(query: str, payload: list[dict], model: str, tag: str) -> dict:
    return {"model": model, "system": JUDGE_SYSTEM, "user": render_judge_user(query, payload),
            "schema": JUDGE_SCHEMA, "tag": tag}


def parse_judgment(result: dict, payload: list[dict]) -> list[dict]:
    """Normalise one LLM result into one row per candidate. Missing/invalid -> label None."""
    cid_to_pid = {c["cid"]: c["product_id"] for c in payload}
    rows = {pid: {"product_id": pid, "agent_label": None, "confidence": None, "missing_attributes": [],
                  "reason_code": None, "is_pick": False, "error": result.get("error")} for pid in cid_to_pid.values()}
    data = result.get("data") if result.get("ok") else None
    if not data:
        return list(rows.values())
    for j in data.get("judgments", []):
        pid = cid_to_pid.get(str(j.get("id", "")).strip())
        if pid is None:
            continue
        label = j.get("label")
        rc = j.get("reason_code")
        rows[pid].update({
            "agent_label": label if label in VALID_LABELS else None,
            "confidence": float(j["confidence"]) if isinstance(j.get("confidence"), (int, float)) else None,
            "missing_attributes": [str(m) for m in (j.get("missing_attributes") or [])][:4],
            "reason_code": rc if rc in REASON_CODES else "other",
        })
    pick = data.get("pick")
    pick_pid = cid_to_pid.get(str(pick).strip()) if pick else None
    if pick_pid:
        rows[pick_pid]["is_pick"] = True
    for r in rows.values():
        r["abstained"] = pick_pid is None
        r["pick_reason"] = data.get("pick_reason", "")
    return list(rows.values())
