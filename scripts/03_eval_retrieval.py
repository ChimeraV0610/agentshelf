"""Step 2b: (a) in-candidate nDCG for random / BM25 / dense / hybrid,
(b) full-corpus rank of every E-labelled product for its query."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import CFG, path  # noqa: E402
from src.data.load import load_processed  # noqa: E402
from src.eval.ndcg import ndcg_from_scores  # noqa: E402
from src.retrieval.bm25_index import BM25Index  # noqa: E402
from src.retrieval.dense_index import DenseIndex  # noqa: E402
from src.retrieval.hybrid import rrf, rrf_scores_for_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="number of queries (0 = all)")
    ap.add_argument("--split", default="", help="train/test or empty for all")
    args = ap.parse_args()
    seed = CFG["seed"]
    ex = load_processed("examples_us")
    if args.split:
        ex = ex[ex["split"] == args.split]
    qdf = ex.drop_duplicates("query_id")[["query_id", "query"]]
    if args.limit:
        qdf = qdf.sample(n=min(args.limit, len(qdf)), random_state=seed)
        ex = ex[ex["query_id"].isin(qdf["query_id"])]
    queries = qdf["query"].tolist()
    qids = qdf["query_id"].tolist()
    print(f"evaluating {len(queries):,} queries, {len(ex):,} judgements")

    bm25 = BM25Index.load(path("indices") / "bm25")
    dense = DenseIndex.load(path("indices") / "dense")
    t = time.time()
    qvecs = dense.encode_queries(queries)
    print(f"encoded queries in {time.time()-t:.0f}s")
    qvec_by_qid = dict(zip(qids, qvecs))

    # ---- (a) in-candidate nDCG ----
    rng = np.random.default_rng(seed)
    rows = []
    k_full = CFG["retrieval"]["k_full"]
    for (qid, query), g in ex.groupby(["query_id", "query"], sort=False):
        pids = g["product_id"].tolist()
        labels = g["esci_label"].tolist()
        s_bm = bm25.score_candidates(query, pids)
        s_de = dense.score_candidates(query, pids, q_vec=qvec_by_qid[qid])
        s_hy = rrf_scores_for_candidates([s_bm, s_de])
        rows.append({
            "query_id": qid,
            "random": ndcg_from_scores(rng.random(len(pids)), labels),
            "bm25": ndcg_from_scores(s_bm, labels),
            "dense": ndcg_from_scores(s_de, labels),
            "hybrid": ndcg_from_scores(s_hy, labels),
        })
    per_q = pd.DataFrame(rows)
    means = per_q.drop(columns="query_id").mean().round(4).to_dict()
    print("in-candidate nDCG:", means)
    per_q.to_parquet(path("results") / "ndcg_per_query.parquet", index=False)

    # ---- (b) full-corpus ranks of E products ----
    t = time.time()
    bm_ids, _ = bm25.search(queries, k=k_full)
    de_ids, _ = dense.search_vectors(qvecs, k=k_full)
    hy_ids = [[pid for pid, _ in rrf([b, d])][:k_full] for b, d in zip(bm_ids, de_ids)]
    print(f"full-corpus search in {time.time()-t:.0f}s")
    rank_rows = []
    pos = {}
    for qid, b, d, h in zip(qids, bm_ids, de_ids, hy_ids):
        pos[qid] = ({p: r + 1 for r, p in enumerate(b)}, {p: r + 1 for r, p in enumerate(d)},
                    {p: r + 1 for r, p in enumerate(h)})
    for r in ex[["query_id", "product_id", "esci_label"]].itertuples(index=False):
        pb, pd_, ph = pos[r.query_id]
        rank_rows.append({"query_id": r.query_id, "product_id": r.product_id, "esci_label": r.esci_label,
                          "rank_bm25": pb.get(r.product_id), "rank_dense": pd_.get(r.product_id),
                          "rank_hybrid": ph.get(r.product_id)})
    ranks = pd.DataFrame(rank_rows)
    ranks.to_parquet(path("results") / "full_corpus_ranks.parquet", index=False)
    e = ranks[ranks["esci_label"] == "E"]
    recall = {}
    for col in ["rank_bm25", "rank_dense", "rank_hybrid"]:
        rr = e[col].fillna(0)
        recall[col] = {"recall@10": round(float((rr.between(1, 10)).mean()), 4),
                       "recall@50": round(float((rr.between(1, 50)).mean()), 4),
                       f"recall@{k_full}": round(float((rr >= 1).mean()), 4),
                       "mrr@100": round(float((1.0 / rr.where(rr > 0)).fillna(0).mean()), 4)}
    print("full-corpus recall for E products:", json.dumps(recall, indent=1))
    summary = {"n_queries": len(queries), "n_judgements": len(ex), "in_candidate_ndcg": means,
               "full_corpus_E": recall, "esci_readme_baseline_ndcg": 0.83}
    (path("results") / "retrieval_eval.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
