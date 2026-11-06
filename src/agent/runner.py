"""Run an agent model over a set of (query, candidate list) pairs and return tidy rows."""
from __future__ import annotations

import pandas as pd

from src.agent.judge import judge_job, make_candidate_payload, parse_judgment
from src.llm.client import LLMClient, get_client


def _quiet_map(client, jobs):
    import asyncio

    if hasattr(client, "amap"):
        async def go():
            return await asyncio.gather(*[client.acall(**j) for j in jobs])
        return client._loop_run(go())
    return client.map(jobs)


def run_agent(model: str, queries: pd.DataFrame, cands: pd.DataFrame, products_by_id: pd.DataFrame,
              tag: str, override_text: dict[str, dict] | None = None,
              client: LLMClient | None = None, verbose: bool = True) -> pd.DataFrame:
    """queries: query_id, query. cands: query_id, product_id, shuffled_pos, in_prompt, esci_label.

    tag distinguishes cache namespaces (e.g. 'sample', 'diag', 'rewrite:<pid>').
    Returns one row per (query_id, product_id) with the agent's judgment.
    """
    client = client or get_client()
    cands = cands[cands["in_prompt"]].sort_values(["query_id", "shuffled_pos"])
    groups = {qid: g for qid, g in cands.groupby("query_id", sort=False)}
    jobs, payloads, qids = [], [], []
    for r in queries.itertuples(index=False):
        g = groups.get(r.query_id)
        if g is None or g.empty:
            continue
        payload = make_candidate_payload(g, products_by_id, override_text)
        jobs.append(judge_job(r.query, payload, model, tag))
        payloads.append(payload)
        qids.append(r.query_id)
    results = client.map(jobs, desc=f"agent {model} [{tag}]") if verbose else _quiet_map(client, jobs)
    if verbose:
        print(client.summary())
    rows = []
    for qid, payload, res in zip(qids, payloads, results):
        for row in parse_judgment(res, payload):
            rows.append({"query_id": qid, "model": model, "tag": tag, **row})
    out = pd.DataFrame(rows)
    out = out.merge(cands[["query_id", "product_id", "esci_label", "shuffled_pos"]], on=["query_id", "product_id"], how="left")
    return out
