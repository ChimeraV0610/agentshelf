"""Step 8: statistics + anti-gaming checks for the rewrite experiment -> results/rewrite_eval.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analysis.rewrite_eval import headline_table, summarize  # noqa: E402
from src.config import path  # noqa: E402


def main():
    d = path("results") / "rewrite"
    ba = pd.read_parquet(d / "before_after.parquet")
    listings = pd.read_parquet(d / "listings.parquet")
    summary = summarize(ba, listings)
    (path("results") / "rewrite_eval.json").write_text(json.dumps(summary, indent=1, default=str))
    table = headline_table(summary)
    table.to_csv(d / "headline_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(table.to_string(index=False))
    for v, pp in summary["per_product"].items():
        print(f"{v}: share of products with higher E pick rate = {pp['share_products_improved_pick']:.2f}")


if __name__ == "__main__":
    main()
