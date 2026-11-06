"""Command-line check of live mode (same code path as the Streamlit page)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live import (fields_from_rewrite, guess_department, load_resources, rewrite_live, run_live,  # noqa: E402
                      suggest_queries, user_fields)
from src.llm.client import get_client  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama:qwen2.5:7b")
    ap.add_argument("--title", default="Insulated Stainless Steel Water Bottle 32 oz with Straw Lid, Leak Proof, Keeps Drinks Cold 24 Hours")
    ap.add_argument("--bullets", default="Double wall vacuum insulation keeps drinks cold for 24 hours\nLeak proof straw lid with carry handle\n32 oz capacity fits most cup holders\nBPA free, 18/8 stainless steel")
    ap.add_argument("--brand", default="")
    ap.add_argument("--n-cands", type=int, default=9)
    ap.add_argument("--n-queries", type=int, default=3)
    ap.add_argument("--rewrite", action="store_true")
    args = ap.parse_args()
    res = load_resources()
    client = get_client()
    f = user_fields(args.title, args.brand, "", args.bullets, "")
    dept = guess_department(f)
    qs = suggest_queries(res, f, k=args.n_queries)
    print("department:", dept)
    print("suggested queries:", [q["query"] for q in qs])
    before = run_live(res, client, args.model, f, qs, n_cands=args.n_cands,
                      progress=lambda i, n, r: print(f"  {i}/{n} {r['query']!r}: rank {r['rank_hybrid']} label {r['agent_label']} picked {r['picked']} {r['seconds']}s err={r['error']}"))
    if args.rewrite:
        rw = rewrite_live(client, f, dept, res.dept_profiles, args.model, args.model)
        print(json.dumps({k: rw[k] for k in ["title", "bullets", "passed", "rounds", "attributes_not_available"]}, indent=1))
        after = run_live(res, client, args.model, fields_from_rewrite(f, rw), qs, n_cands=args.n_cands,
                         progress=lambda i, n, r: print(f"  {i}/{n} {r['query']!r}: rank {r['rank_hybrid']} label {r['agent_label']} picked {r['picked']} {r['seconds']}s"))
        print("picked before/after:", sum(r["picked"] for r in before), "/", sum(r["picked"] for r in after))
    print(client.summary())


if __name__ == "__main__":
    main()
