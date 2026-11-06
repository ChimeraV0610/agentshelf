"""Compare original and rewritten listings on the six demo shelves."""
from __future__ import annotations

import argparse
import json
from statistics import mean

from engine_cli import (
    STATIC_PROFILES, apply_rewrite, audit_listing, fields_of, get_client,
    load_buyers, load_shelves, rewrite, simulate, weakest_index,
)

from src.analysis.rewrite_eval import paired_bootstrap
from src.config import path

METRICS = ("pick_rate", "top3_rate", "avg_rank", "fair_share", "audit_score")


def measure(client, model, competitors, listing, buyers, vocab, tag):
    result = simulate(client, model, competitors, listing, buyers, tag=tag)
    if any(row["error"] for row in result["per_buyer"]):
        raise RuntimeError(f"{tag}: buyer simulation failed")
    metrics = {key: result[key] for key in METRICS if key != "audit_score"}
    ranks = [row["rank"] for row in result["per_buyer"]]
    metrics["avg_rank"] = mean(result["n"] if rank is None else rank for rank in ranks)
    metrics["omitted_count"] = ranks.count(None)
    metrics["audit_score"] = audit_listing(listing, vocab)["score"]
    if not result["n_ok"]:
        metrics["top3_rate"] = None
    return metrics


def summarize(client, model, shelves, buyers):
    rows = []
    for category, shelf in shelves.items():
        products = shelf["products"]
        eligible = [i for i, p in enumerate(products) if len(p["title"]) >= 40 and len(p["bullets"]) >= 3]
        if not eligible:
            raise ValueError(f"{category}: no eligible listing")
        i = eligible[weakest_index({**shelf, "products": [products[j] for j in eligible]})]
        mine = fields_of(products[i])
        competitors = [p for j, p in enumerate(products) if j != i]
        needs = buyers[category]
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
    args = parser.parse_args()
    if not args.model.startswith("ollama:") or not args.model.removeprefix("ollama:"):
        parser.error("--model must be a local Ollama model (ollama:<name>)")
    summary = summarize(get_client(), args.model, load_shelves(), load_buyers())
    output = path("results") / "live_summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
