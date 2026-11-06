"""Dense retrieval with bge-small-en-v1.5 + FAISS IndexFlatIP (exact cosine)."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.config import CFG


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DenseIndex:
    def __init__(self, ids: np.ndarray, emb: np.ndarray, model=None):
        self.ids = ids
        self.id_to_idx = {pid: i for i, pid in enumerate(ids)}
        self.emb = emb.astype(np.float32)  # normalized
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(CFG["retrieval"]["dense_model"], device=_device())
        return self._model

    # ---------- build / load ----------
    @classmethod
    def encode_corpus(cls, ids: list[str], texts: list[str], save_dir: Path) -> "DenseIndex":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(CFG["retrieval"]["dense_model"], device=_device())
        model.max_seq_length = 256
        emb = model.encode(
            texts,
            batch_size=CFG["retrieval"]["encode_batch_size"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "emb.npy", emb.astype(np.float16))
        np.save(save_dir / "ids.npy", np.asarray(ids))
        return cls(np.asarray(ids), emb, model)

    @classmethod
    def load(cls, save_dir: Path) -> "DenseIndex":
        emb = np.load(save_dir / "emb.npy").astype(np.float32)
        ids = np.load(save_dir / "ids.npy", allow_pickle=True)
        return cls(ids, emb)

    # ---------- encode ----------
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefix = CFG["retrieval"]["bge_query_prefix"]
        return self.model.encode(
            [prefix + q for q in queries], batch_size=256, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(queries) > 1000,
        ).astype(np.float32)

    def encode_single_doc(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    # ---------- search ----------
    def search(self, queries: list[str], k: int = 100) -> tuple[list[list[str]], np.ndarray]:
        q = self.encode_queries(queries)
        scores, idx = self.index.search(q, k)
        return [[self.ids[j] for j in row if j >= 0] for row in idx], scores

    def search_vectors(self, q: np.ndarray, k: int = 100):
        scores, idx = self.index.search(q.astype(np.float32), k)
        return [[self.ids[j] for j in row if j >= 0] for row in idx], scores

    def score_candidates(self, query: str, product_ids: list[str], q_vec: np.ndarray | None = None) -> np.ndarray:
        q = q_vec if q_vec is not None else self.encode_queries([query])[0]
        rows = [self.id_to_idx[p] for p in product_ids if p in self.id_to_idx]
        out = np.zeros(len(product_ids), dtype=np.float32)
        if rows:
            sims = self.emb[rows] @ q
            j = 0
            for i, p in enumerate(product_ids):
                if p in self.id_to_idx:
                    out[i] = sims[j]
                    j += 1
        return out

    def rank_of_score(self, q_vec: np.ndarray, score: float, exclude_pid: str | None = None) -> int:
        sims = self.emb @ q_vec.astype(np.float32)
        if exclude_pid is not None and exclude_pid in self.id_to_idx:
            sims = sims.copy()
            sims[self.id_to_idx[exclude_pid]] = -np.inf
        return int((sims > score).sum()) + 1
