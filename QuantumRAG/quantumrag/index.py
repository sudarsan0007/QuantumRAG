"""Classical vector index (FAISS with an exact NumPy fallback).

The classical baseline uses an exact inner-product index over L2-normalised
vectors, which equals cosine similarity. FAISS ``IndexFlatIP`` is used when the
``faiss`` package is installed; otherwise an equivalent NumPy brute-force index
is used. Both return identical rankings (exact search), so the fallback does not
change accuracy, only speed on large corpora.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on environment
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    HAVE_FAISS = False


class VectorIndex:
    def __init__(self, use_faiss: bool | None = None):
        self.use_faiss = HAVE_FAISS if use_faiss is None else (use_faiss and HAVE_FAISS)
        self._index = None
        self._matrix: np.ndarray | None = None

    @property
    def backend(self) -> str:
        return "faiss.IndexFlatIP" if self.use_faiss else "numpy exact inner product"

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._matrix = vectors
        if self.use_faiss:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
        return self

    def __len__(self) -> int:
        return 0 if self._matrix is None else self._matrix.shape[0]

    @property
    def nbytes(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.nbytes)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) of the top-k most similar vectors."""
        if self._matrix is None:
            raise RuntimeError("index is empty")
        k = int(min(k, len(self)))
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if self.use_faiss:
            scores, idx = self._index.search(q, k)
            return idx[0].astype(int), scores[0].astype(float)
        scores = self._matrix @ q[0]
        if k < len(scores):
            part = np.argpartition(-scores, k - 1)[:k]
        else:
            part = np.arange(len(scores))
        order = part[np.argsort(-scores[part], kind="stable")]
        return order.astype(int), scores[order].astype(float)
