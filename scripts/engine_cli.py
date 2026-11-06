"""Run the 4-step engine from the command line (also pre-warms the cache for the live demo).

python scripts/engine_cli.py --model ollama:qwen2.5:7b --category "air purifier" --buyers 3 --rewrite
python scripts/engine_cli.py --model ollama:qwen2.5:7b --all --buyers 6 --rewrite     # pre-warm every shelf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.audit import audit_listing  # noqa: E402
from src.engine.rewrite import apply_rewrite, rewrite  # noqa: E402
from src.engine.simulate import simulate  # noqa: E402
from src.llm.client import get_client  # noqa: E402
from src.rewrite.dept_profile import STATIC_PROFILES  # noqa: E402
from src.shelves import load_buyers, load_shelves  # noqa: E402


def fields_of(p: dict) -> dict:
    return {"title": p["title"], "brand": p.get("brand", ""), "color": p.get("color", ""), "bullets_list": p["bullets"],
            "bullets_full": " | ".join(p["bullets"]), "bullets": " | ".join(p["bullets"])[:500],
            "description_full": p.get("description", ""), "description": p.get("description", "")[:600]}


def weakest_index(shelf: dict) -> int:
    scores = [audit_listing(fields_of(p), shelf["buyer_vocab"])["score"] for p in shelf["products"]]
    return int(min(range(len(scores)), key=lambda i: scores[i]))


def run_one(cat: str, model: str, n_buyers: int, do_rewrite: bool, mine_idx: int | None = None):
    shelves, buyers = load_shelves(), load_buyers()
    shelf = shelves[cat]
    client = get_client()
    i = weakest_index(shelf) if mine_idx is None else mine_idx
    mine = fields_of(shelf["products"][i])
    competitors = [p for j, p in enumerate(shelf["products"]) if j != i]
    needs = buyers[cat][:n_buyers]
    a0 = audit_listing(mine, shelf["buyer_vocab"])
    print(f"\n== {cat}: mine = {mine['title'][:80]!r}\n   audit {a0['score']}/100: " + "; ".join(a0["issues"][:3]))
    s0 = simulate(client, model, competitors, mine, needs, tag=f"sim:{cat}:{i}:orig",
                  progress=lambda k, n, r: print(f"   buyer {k}/{n}: rank {r['rank']}/{r['n']} picked={r['picked']} {r['seconds']}s err={r['error']} | {r['reason'][:90] if r['reason'] else ''}"))
    print(f"   BEFORE pick {s0['pick_rate']:.2f} (fair {s0['fair_share']:.2f}) top3 {s0['top3_rate']:.2f} avg rank {s0['avg_rank']:.1f} | missing: {s0['missing_facts'][:5]}")
    if not do_rewrite:
        print(client.summary())
        return
    rw = rewrite(client, mine, STATIC_PROFILES.get(shelf["department"], ""), model, model, tag=f"rw:{cat}:{i}")
    print(f"   REWRITE passed={rw['passed']} rounds={rw['rounds']} title={rw['title'][:100]!r}")
    for b in rw["bullets"]:
        print(f"      - {b}")
    print(f"      seller_to_confirm: {rw['seller_to_confirm']}")
    mine2 = apply_rewrite(mine, rw)
    a1 = audit_listing(mine2, shelf["buyer_vocab"])
    print(f"   audit after {a1['score']}/100")
    s1 = simulate(client, model, competitors, mine2, needs, tag=f"sim:{cat}:{i}:rw",
                  progress=lambda k, n, r: print(f"   buyer {k}/{n}: rank {r['rank']}/{r['n']} picked={r['picked']} {r['seconds']}s"))
    print(f"   AFTER  pick {s1['pick_rate']:.2f} (fair {s1['fair_share']:.2f}) top3 {s1['top3_rate']:.2f} avg rank {s1['avg_rank']:.1f}")
    print(client.summary())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama:qwen2.5:7b")
    ap.add_argument("--category", default="air purifier")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--buyers", type=int, default=3)
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--mine", type=int, default=None, help="index of the shelf product to treat as mine (default: weakest audit)")
    args = ap.parse_args()
    cats = list(load_shelves()) if args.all else [args.category]
    for c in cats:
        run_one(c, args.model, args.buyers, args.rewrite, args.mine)


if __name__ == "__main__":
    main()
