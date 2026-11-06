"""Dense index over the products referenced by the small US examples (~482k) - enough for
in-candidate nDCG, hybrid retrieval and competitor search, at a fraction of the 1.2M encode time."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import path  # noqa: E402
from src.data.load import load_processed  # noqa: E402
from src.retrieval.dense_index import DenseIndex  # noqa: E402


def main():
    prod = load_processed("products_us")
    corpus = load_processed("corpus")
    keep = set(prod.loc[prod["in_small"], "product_id"])
    corpus = corpus[corpus["product_id"].isin(keep)]
    print(f"encoding {len(corpus):,} products (small-version subset)")
    t = time.time()
    DenseIndex.encode_corpus(corpus["product_id"].tolist(), corpus["doc_text"].fillna("").tolist(), path("indices") / "dense")
    print(f"done in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
