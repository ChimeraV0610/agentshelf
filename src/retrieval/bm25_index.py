"""BM25 index over the full US product corpus (bm25s) plus single-document scoring.

bm25s cannot add a document after indexing, so for the rewrite experiment we
re-implement Lucene-style BM25 for a single document from stored corpus stats
(df per token, N, avgdl). `check_single_doc_consistency` verifies the two agree.
"""
from __future__ import annotations

import json
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from src.config import CFG

STEMMER = Stemmer.Stemmer("english")


def tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, return_ids=False, show_progress=False)


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, ids: np.ndarray, vocab: dict[str, int],
                 df: np.ndarray, doc_len: np.ndarray, k1: float, b: float):
        self.retriever = retriever
        self.ids = ids
        self.id_to_idx = {pid: i for i, pid in enumerate(ids)}
        self.vocab = vocab
        self.df = df
        self.doc_len = doc_len
        self.N = len(ids)
        self.avgdl = float(doc_len.mean())
        self.k1, self.b = k1, b

    # ---------- build / load ----------
    @classmethod
    def build(cls, ids: list[str], texts: list[str], save_dir: Path) -> "BM25Index":
        k1 = CFG["retrieval"]["bm25_k1"]
        b = CFG["retrieval"]["bm25_b"]
        tok = bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, show_progress=True)
        # document frequency per vocab id and document lengths
        vocab_size = len(tok.vocab)
        df = np.zeros(vocab_size, dtype=np.int64)
        doc_len = np.zeros(len(texts), dtype=np.int32)
        for i, doc_ids in enumerate(tok.ids):
            doc_len[i] = len(doc_ids)
            if doc_ids:
                df[np.unique(doc_ids)] += 1
        retriever = bm25s.BM25(method="lucene", k1=k1, b=b)
        retriever.index(tok, show_progress=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(save_dir / "bm25s"))
        np.save(save_dir / "ids.npy", np.asarray(ids))
        np.save(save_dir / "df.npy", df)
        np.save(save_dir / "doc_len.npy", doc_len)
        with open(save_dir / "vocab.json", "w") as f:
            json.dump(tok.vocab, f)
        return cls(retriever, np.asarray(ids), tok.vocab, df, doc_len, k1, b)

    @classmethod
    def load(cls, save_dir: Path) -> "BM25Index":
        retriever = bm25s.BM25.load(str(save_dir / "bm25s"))
        ids = np.load(save_dir / "ids.npy", allow_pickle=True)
        df = np.load(save_dir / "df.npy")
        doc_len = np.load(save_dir / "doc_len.npy")
        with open(save_dir / "vocab.json") as f:
            vocab = json.load(f)
        return cls(retriever, ids, vocab, df, doc_len, CFG["retrieval"]["bm25_k1"], CFG["retrieval"]["bm25_b"])

    # ---------- search ----------
    def search(self, queries: list[str], k: int = 100) -> tuple[list[list[str]], np.ndarray]:
        tok = tokenize(queries)
        idx, scores = self.retriever.retrieve(tok, k=k, show_progress=False, n_threads=-1)
        return [[self.ids[j] for j in row] for row in idx], scores

    def get_scores(self, query: str) -> np.ndarray:
        """BM25 score of every document in the corpus for one query."""
        tok = tokenize([query])[0]
        return self.retriever.get_scores(tok)

    def score_candidates(self, query: str, product_ids: list[str]) -> np.ndarray:
        all_scores = self.get_scores(query)
        return np.array([all_scores[self.id_to_idx[p]] if p in self.id_to_idx else 0.0 for p in product_ids])

    # ---------- single-document scoring (for rewritten listings) ----------
    def idf(self, token: str) -> float:
        vid = self.vocab.get(token)
        df = self.df[vid] if vid is not None else 0
        return float(np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)))

    def score_single_doc(self, query: str, doc_text: str) -> float:
        q_tokens = tokenize([query])[0]
        d_tokens = tokenize([doc_text])[0]
        dl = len(d_tokens)
        if dl == 0 or not q_tokens:
            return 0.0
        tf: dict[str, int] = {}
        for t in d_tokens:
            tf[t] = tf.get(t, 0) + 1
        norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
        score = 0.0
        for t in set(q_tokens):
            f = tf.get(t, 0)
            if f == 0:
                continue
            score += self.idf(t) * f / (f + norm)
        return float(score)

    def rank_of_score(self, query: str, score: float, exclude_pid: str | None = None) -> int:
        """1-based rank a document with `score` would get in the full corpus."""
        all_scores = self.get_scores(query)
        if exclude_pid is not None and exclude_pid in self.id_to_idx:
            all_scores = all_scores.copy()
            all_scores[self.id_to_idx[exclude_pid]] = -np.inf
        return int((all_scores > score).sum()) + 1

    def check_single_doc_consistency(self, query: str, texts_by_pid: dict[str, str], n: int = 20) -> float:
        """Max abs difference between index score and re-implemented score on n existing docs."""
        all_scores = self.get_scores(query)
        diffs = []
        for pid, text in list(texts_by_pid.items())[:n]:
            diffs.append(abs(all_scores[self.id_to_idx[pid]] - self.score_single_doc(query, text)))
        return float(max(diffs)) if diffs else 0.0
