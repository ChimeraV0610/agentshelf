"""Step 6: select products, run the agent on their queries, write per-product diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.agent.runner import run_agent  # noqa: E402
from src.config import CFG, path  # noqa: E402
from src.data.load import build_listing_fields, load_processed  # noqa: E402
from src.diagnose.select_products import build_query_sets, product_pool, select_products  # noqa: E402
from src.diagnose.shelf import diagnose_product  # noqa: E402
from src.llm.client import has_api_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["models"]["agent_primary"])
    ap.add_argument("--n", type=int, default=CFG["diagnosis"]["n_products"])
    ap.add_argument("--skip-agent", action="store_true", help="retrieval diagnostics only")
    args = ap.parse_args()
    seed = CFG["seed"]
    ex = load_processed("examples_us")
    ranks = pd.read_parquet(path("results") / "full_corpus_ranks.parquet")
    depts = pd.read_parquet(path("samples") / "query_departments.parquet")
    products = load_processed("products_us").set_index("product_id")

    pool = product_pool(ex, ranks, depts)
    print(f"pool: {len(pool):,} products ({int(pool['weak_baseline'].sum()):,} weak baseline)")
    selected = select_products(pool, args.n, seed)
    selected.to_parquet(path("samples") / "products_for_diagnosis.parquet", index=False)
    queries, cands, links = build_query_sets(selected, ex, seed)
    queries.to_parquet(path("samples") / "diag_queries.parquet", index=False)
    cands.to_parquet(path("samples") / "diag_candidates.parquet", index=False)
    links.to_parquet(path("samples") / "diag_links.parquet", index=False)
    print(f"selected {len(selected)} products, {len(queries):,} queries, {int(cands['in_prompt'].sum()):,} prompt candidates")
    print(selected["department"].value_counts().to_string())

    runs = pd.DataFrame(columns=["query_id", "product_id", "agent_label", "is_pick", "reason_code", "missing_attributes", "abstained"])
    if not args.skip_agent:
        if not has_api_key(args.model) and os.environ.get("AGENTSHELF_MOCK") != "1":
            sys.exit(f"no API key for {args.model}; use --skip-agent or set .env")
        runs = run_agent(args.model, queries, cands, products, tag="diag")
        d = path("results") / "agent_runs"
        d.mkdir(exist_ok=True)
        runs.to_parquet(d / f"diag_{args.model}.parquet", index=False)

    out_dir = path("results") / "diagnostics"
    out_dir.mkdir(exist_ok=True)
    summary = []
    for pid in selected["product_id"]:
        diag = diagnose_product(pid, links, ex, ranks, runs, cands)
        diag["listing"] = build_listing_fields(products.loc[pid], desc_chars=2000, bullet_chars=2000)
        diag["department"] = selected.set_index("product_id").loc[pid, "department"]
        (out_dir / f"{pid}.json").write_text(json.dumps(diag, indent=1, default=str))
        summary.append({"product_id": pid, "department": diag["department"], "title": diag["listing"]["title"],
                        "n_e_queries": diag["n_e_queries"], "n_non_e_queries": diag["n_non_e_queries"],
                        "hybrid_top10_rate": diag["retrieval"]["hybrid"]["top10_rate"],
                        "hybrid_median_rank": diag["retrieval"]["hybrid"]["median_rank"],
                        "bm25_median_rank": diag["retrieval"]["bm25"]["median_rank"],
                        "dense_median_rank": diag["retrieval"]["dense"]["median_rank"],
                        "pick_rate_E": diag["agent"]["pick_rate_E"], "fair_share": diag["agent"]["fair_share"],
                        "pick_rate_nonE": diag["agent"]["pick_rate_nonE"],
                        "top_reason": diag["agent"]["top_reasons"][0][0] if diag["agent"]["top_reasons"] else None})
    s = pd.DataFrame(summary)
    s.to_parquet(path("results") / "diagnostics_summary.parquet", index=False)
    print(s.describe().round(3).to_string())


if __name__ == "__main__":
    main()
