"""Compare original and rewritten listings on the six demo shelves."""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from statistics import mean

from engine_cli import (
    STATIC_PROFILES, apply_rewrite, audit_listing, fields_of, get_client,
    load_buyers, load_shelves, rewrite, simulate, weakest_index,
)

from src.analysis.rewrite_eval import paired_bootstrap
from src.config import path
from src.llm.client import LLMClient

METRICS = ("pick_rate", "top3_rate", "avg_rank", "fair_share", "audit_score")


class CacheOnlyClient:
    def __init__(self, db):
        self.db = db

    def call(self, *, model, system, user, schema, tag, max_tokens):
        key = LLMClient.key(model, system, user, schema, tag)
        row = self.db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            raise RuntimeError(f"cache miss: {tag} ({model})")
        return json.loads(row[0])


def measure(client, model, competitors, listing, buyers, vocab, tag):
    result = simulate(client, model, competitors, listing, buyers, tag=tag)
    if any(row["error"] for row in result["per_buyer"]):
        raise RuntimeError(f"{tag}: buyer simulation failed")
    metrics = {key: result[key] for key in METRICS if key != "audit_score"}
    ranks = [row["rank"] for row in result["per_buyer"]]
    metrics["avg_rank"] = mean(result["n"] if rank is None else rank for rank in ranks)
    metrics["omitted_count"] = sum(
        row["rank"] is None or row.get("omitted", False) for row in result["per_buyer"]
    )
    metrics["audit_score"] = audit_listing(listing, vocab)["score"]
    if not result["n_ok"]:
        metrics["top3_rate"] = None
    return metrics


def summarize(client, model, shelves, buyers, n_buyers):
    rows = []
    for category, shelf in shelves.items():
        products = shelf["products"]
        i = weakest_index(shelf)
        product = products[i]
        if len(product["title"]) < 40 or len(product["bullets"]) < 3:
            raise ValueError(f"{category}: engine_cli selection {i} does not meet the listing requirements")
        mine = fields_of(product)
        competitors = [p for j, p in enumerate(products) if j != i]
        needs = buyers[category][:n_buyers]
        if len(needs) != n_buyers:
            raise ValueError(f"{category}: expected {n_buyers} buyer scenarios, found {len(needs)}")
        vocab = shelf["buyer_vocab"]
        before = measure(client, model, competitors, mine, needs, vocab, f"sim:{category}:{i}:orig")
        rw = rewrite(client, mine, STATIC_PROFILES.get(shelf["department"], ""), model, model, tag=f"rw:{category}:{i}")
        if all("error" in attempt for attempt in rw["history"]):
            raise RuntimeError(f"{category}: rewrite failed to produce a listing")
        after = measure(client, model, competitors, apply_rewrite(mine, rw), needs, vocab, f"sim:{category}:{i}:rw")
        rows.append({"category": category, "product_id": products[i]["product_id"], "n_buyers": len(needs),
                     "before": before, "after": after, "faithfulness_passed": rw["passed"]})
        print(f"{category}: pick rate {before['pick_rate']:.1%} → {after['pick_rate']:.1%}", flush=True)

    averages = {"before": {}, "after": {}, "paired_bootstrap": {}}
    for key in METRICS:
        before, after = zip(*[(row["before"][key], row["after"][key]) for row in rows])
        averages["before"][key] = mean(before) if None not in before else None
        averages["after"][key] = mean(after) if None not in after else None
        averages["paired_bootstrap"][key] = (
            paired_bootstrap(before, after) if None not in before + after else None
        )
    return {"model": model, "shelves": rows, "averages": averages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ollama:qwen2.5:7b")
    parser.add_argument("--buyers", type=int, default=6, help="number of buyer scenarios per shelf")
    parser.add_argument("--cache-only", action="store_true", help="stop at the first cache miss without calling Ollama")
    args = parser.parse_args()
    if args.buyers < 1:
        parser.error("--buyers must be positive")
    if not args.model.startswith("ollama:") or not args.model.removeprefix("ollama:"):
        parser.error("--model must be a local Ollama model (ollama:<name>)")
    inputs = args.model, load_shelves(), load_buyers(), args.buyers
    try:
        if args.cache_only:
            cache_uri = path("cache").as_uri() + "?mode=ro"
            with closing(sqlite3.connect(cache_uri, uri=True)) as db:
                summary = summarize(CacheOnlyClient(db), *inputs)
        else:
            summary = summarize(get_client(), *inputs)
    except (RuntimeError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")
    output = path("results") / "live_summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
