"""build_data: six real ESCI shelves -> data/shelves.json"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shelves import build_shelves, save_shelves  # noqa: E402


def main():
    shelves = build_shelves()
    save_shelves(shelves)
    for cat, s in shelves.items():
        print(f"\n== {cat} | query '{s['query']}' | {len(s['products'])} products | vocab: {', '.join(s['buyer_vocab'][:10])}")
        for p in s["products"]:
            print(f"   {p['product_id']} {p['title'][:80]!r} bullets={len(p['bullets'])} desc={len(p['description'])}")


if __name__ == "__main__":
    main()
