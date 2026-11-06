"""Step 3a: tag every query in the chosen split with one of 12 departments."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, path  # noqa: E402
from src.data.load import load_processed  # noqa: E402
from src.llm.client import has_api_key  # noqa: E402
from src.sampling.departments import tag_keyword, tag_llm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["llm", "keyword"], default="llm")
    ap.add_argument("--split", default="all", help="all | train | test")
    args = ap.parse_args()
    ex = load_processed("examples_us")
    if args.split != "all":
        ex = ex[ex["split"] == args.split]
    queries = ex.drop_duplicates("query_id")[["query_id", "query"]].reset_index(drop=True)
    model = CFG["models"]["tagger"]
    if args.method == "llm" and not has_api_key(model) and os.environ.get("AGENTSHELF_MOCK") != "1":
        print(f"no API key for {model}; falling back to keyword tagger")
        args.method = "keyword"
    tagged = tag_llm(queries, model) if args.method == "llm" else tag_keyword(queries)
    out = path("samples") / "query_departments.parquet"
    tagged.to_parquet(out, index=False)
    print(tagged["department"].value_counts())
    print(tagged["method"].value_counts())
    print("saved", out)


if __name__ == "__main__":
    main()
