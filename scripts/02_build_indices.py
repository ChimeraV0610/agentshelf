"""Step 2a: build BM25 and dense indices over the corpus."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import path  # noqa: E402
from src.data.load import load_processed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bm25", action="store_true")
    ap.add_argument("--skip-dense", action="store_true")
    args = ap.parse_args()
    corpus = load_processed("corpus")
    ids = corpus["product_id"].tolist()
    texts = corpus["doc_text"].fillna("").tolist()
    if not args.skip_bm25:
        from src.retrieval.bm25_index import BM25Index

        t = time.time()
        idx = BM25Index.build(ids, texts, path("indices") / "bm25")
        print(f"bm25 built in {time.time()-t:.0f}s, N={idx.N}, avgdl={idx.avgdl:.1f}, vocab={len(idx.vocab):,}")
        q = "wireless bluetooth headphones"
        sample = dict(zip(ids[:50], texts[:50]))
        print("single-doc scoring max abs diff vs index:", idx.check_single_doc_consistency(q, sample))
    if not args.skip_dense:
        from src.retrieval.dense_index import DenseIndex

        t = time.time()
        DenseIndex.encode_corpus(ids, texts, path("indices") / "dense")
        print(f"dense built in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
