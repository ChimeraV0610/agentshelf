"""Live mode: a seller pastes their own listing, the pipeline runs on it end to end.

Reuses the batch components: full-corpus BM25 + dense indices (single-doc rescoring),
real ESCI candidate lists for suggested queries, the same agent prompt/schema, and
the faithful rewriter with its audit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agent.judge import judge_job, parse_judgment
from src.config import CFG, path
from src.data.load import build_listing_fields, clean_text, load_processed
from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid import rrf
from src.rewrite.rescore import Rescorer, doc_text_from
from src.rewrite.rewriter import rewrite_listing
from src.sampling.departments import tag_keyword

USER_PID = "__USER_LISTING__"


@dataclass
class LiveResources:
    bm25: BM25Index
    dense: DenseIndex
    rescorer: Rescorer
    examples: pd.DataFrame        # query_id, query, product_id, esci_label
    queries: pd.DataFrame         # query_id, query (unique)
    products: pd.DataFrame        # indexed by product_id
    query_emb: np.ndarray         # aligned with queries
    dept_profiles: dict


def load_resources() -> LiveResources:
    bm25 = BM25Index.load(path("indices") / "bm25")
    dense = DenseIndex.load(path("indices") / "dense", use_faiss=False)
    ex = load_processed("examples_us")
    queries = ex.drop_duplicates("query_id")[["query_id", "query"]].reset_index(drop=True)
    products = load_processed("products_us").set_index("product_id")
    qe_file = path("indices") / "dense" / "query_emb.npy"
    qid_file = path("indices") / "dense" / "query_ids.npy"
    if qe_file.exists() and qid_file.exists() and len(np.load(qid_file)) == len(queries):
        query_emb = np.load(qe_file).astype(np.float32)
    else:
        query_emb = dense.encode_queries(queries["query"].tolist())
        np.save(qe_file, query_emb.astype(np.float16))
        np.save(qid_file, queries["query_id"].values)
    profiles = {}
    pf = path("results") / "rewrite" / "dept_profiles.json"
    if pf.exists():
        import json

        profiles = json.loads(pf.read_text())
    if not profiles:
        from src.rewrite.dept_profile import STATIC_PROFILES

        profiles = dict(STATIC_PROFILES)
    return LiveResources(bm25, dense, Rescorer(bm25, dense), ex, queries, products, query_emb, profiles)


def user_fields(title: str, brand: str = "", color: str = "", bullets_text: str = "", description: str = "") -> dict:
    bullets = [clean_text(b) for b in bullets_text.splitlines() if clean_text(b)]
    return {"title": clean_text(title), "brand": clean_text(brand), "color": clean_text(color),
            "bullets_list": bullets, "bullets_full": " | ".join(bullets), "description_full": clean_text(description),
            "bullets": " | ".join(bullets)[:400], "description": clean_text(description)[:600]}


def doc_text_for(fields: dict) -> str:
    return doc_text_from(fields["title"], fields["brand"], fields["color"], fields["bullets_list"])


def guess_department(fields: dict) -> str:
    df = pd.DataFrame({"query_id": [0], "query": [fields["title"] + " " + fields["bullets_full"]]})
    return tag_keyword(df)["department"].iloc[0]


def suggest_queries(res: LiveResources, fields: dict, k: int = 5) -> list[dict]:
    """Nearest real Amazon queries to the listing (by bge similarity, queries encoded with the query prefix)."""
    v = res.dense.encode_single_doc(doc_text_for(fields))
    sims = res.query_emb @ v
    top = np.argsort(-sims)[: k * 3]
    out, seen = [], set()
    for i in top:
        q = res.queries.iloc[i]
        key = q["query"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append({"query_id": int(q["query_id"]), "query": q["query"], "similarity": float(sims[i])})
        if len(out) >= k:
            break
    return out


def live_candidates(res: LiveResources, query: str, query_id: int | None, n: int, seed: int) -> list[str]:
    """Other candidate product_ids shown next to the user's listing."""
    rng = np.random.default_rng(seed)
    if query_id is not None:
        g = res.examples[res.examples["query_id"] == query_id]
        e = g[g["esci_label"] == "E"]["product_id"].tolist()
        rest = g[g["esci_label"] != "E"]["product_id"].tolist()
        rng.shuffle(e)
        rng.shuffle(rest)
        chosen = (e + rest)[:n]
        return chosen
    bm_ids, _ = res.bm25.search([query], k=50)
    de_ids, _ = res.dense.search([query], k=50)
    fused = [p for p, _ in rrf([bm_ids[0], de_ids[0]])]
    return fused[:n]


def _payload(res: LiveResources, fields: dict, others: list[str], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    pids = others + [USER_PID]
    order = rng.permutation(len(pids))
    payload = []
    for i, j in enumerate(order, start=1):
        pid = pids[j]
        if pid == USER_PID:
            f = {k: fields[k] for k in ["title", "brand", "color", "bullets", "description"]}
        else:
            f = build_listing_fields(res.products.loc[pid])
        payload.append({"cid": f"c{i:02d}", "product_id": pid, **f})
    return payload


def run_live(res: LiveResources, client, model: str, fields: dict, queries: list[dict], n_cands: int = 9,
             seed: int = 42, progress=None) -> list[dict]:
    """queries: [{'query': str, 'query_id': int|None}]. Returns one dict per query."""
    doc_text = doc_text_for(fields)
    results = []
    for k, q in enumerate(queries):
        t0 = time.time()
        ranks = res.rescorer.ranks(USER_PID, q["query"], doc_text)
        others = live_candidates(res, q["query"], q.get("query_id"), n_cands, seed + k)
        payload = _payload(res, fields, others, seed + k)
        job = judge_job(q["query"], payload, model, tag=f"live:{hash(doc_text) & 0xFFFF}")
        r = client.call(**job)
        rows = {x["product_id"]: x for x in parse_judgment(r, payload)}
        mine = rows.get(USER_PID, {})
        picked_pid = next((p for p, x in rows.items() if x.get("is_pick")), None)
        picked_title = None
        if picked_pid and picked_pid != USER_PID:
            picked_title = clean_text(res.products.loc[picked_pid]["product_title"])[:120]
        results.append({"query": q["query"], "query_id": q.get("query_id"), "source": "real Amazon query" if q.get("query_id") else "typed",
                        **{k: ranks[k] for k in ["rank_bm25", "rank_dense", "rank_hybrid"]},
                        "agent_label": mine.get("agent_label"), "picked": bool(mine.get("is_pick")),
                        "abstained": bool(mine.get("abstained")), "reason_code": mine.get("reason_code"),
                        "missing_attributes": mine.get("missing_attributes", []), "pick_reason": mine.get("pick_reason", ""),
                        "picked_instead": picked_title, "n_candidates": len(payload), "error": r.get("error"),
                        "seconds": round(time.time() - t0, 1)})
        if progress:
            progress(k + 1, len(queries), results[-1])
    return results


def rewrite_live(client, fields: dict, department: str, profiles: dict, rewriter_model: str, judge_model: str) -> dict:
    return rewrite_listing(client, fields, profiles.get(department, ""), rewriter_model, judge_model, pid="live")


def fields_from_rewrite(fields: dict, rw: dict) -> dict:
    new = dict(fields)
    new["title"] = rw["title"]
    new["bullets_list"] = rw["bullets"]
    new["bullets_full"] = " | ".join(rw["bullets"])
    new["bullets"] = new["bullets_full"][:400]
    return new


def available_models() -> list[str]:
    """Models callable right now: pulled Ollama models first, then API models with keys."""
    import os

    out = []
    try:
        import httpx

        host = CFG["llm"].get("ollama", {}).get("host", "http://localhost:11434")
        tags = httpx.get(f"{host}/api/tags", timeout=2).json().get("models", [])
        out += [f"ollama:{m['name']}" for m in tags]
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        out += ["claude-sonnet-5-5", "claude-haiku-4-5", "claude-opus-5-5"]
    if os.environ.get("OPENAI_API_KEY"):
        out += ["gpt-4o-mini", "gpt-4o"]
    return out
