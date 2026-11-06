"""Step 4: run one agent model over the sampled queries (all calls cached)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.agent.runner import run_agent  # noqa: E402
from src.config import CFG, path  # noqa: E402
from src.data.load import load_processed  # noqa: E402
from src.llm.client import has_api_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["models"]["agent_primary"])
    ap.add_argument("--limit", type=int, default=0, help="first N queries only (sanity run)")
    args = ap.parse_args()
    if not has_api_key(args.model) and os.environ.get("AGENTSHELF_MOCK") != "1":
        sys.exit(f"no API key for {args.model}: set it in .env")
    queries = pd.read_parquet(path("samples") / "queries_1000.parquet")[["query_id", "query"]]
    cands = pd.read_parquet(path("samples") / "candidates_1000.parquet")
    if args.limit:
        queries = queries.head(args.limit)
        cands = cands[cands["query_id"].isin(queries["query_id"])]
    products = load_processed("products_us").set_index("product_id")
    out = run_agent(args.model, queries, cands, products, tag="sample")
    d = path("results") / "agent_runs"
    d.mkdir(exist_ok=True)
    f = d / f"{args.model}{'_limit' + str(args.limit) if args.limit else ''}.parquet"
    out.to_parquet(f, index=False)
    valid = out["agent_label"].notna().mean()
    print(f"rows={len(out):,} queries={out['query_id'].nunique():,} labelled={valid:.3f}")
    print(pd.crosstab(out["esci_label"], out["agent_label"].fillna("none")))
    print("saved", f)


if __name__ == "__main__":
    main()
