"""The three retrieval pipelines, plus a PCA-cosine control.

* ClassicalRetriever - 384-d cosine similarity through the vector index.
* QuantumRetriever   - PCA -> feature map -> fidelity against every document.
* HybridRetriever    - classical top-Kc shortlist, quantum re-rank to top-Kq.
* PCACosineRetriever - cosine on the same PCA features the quantum encoders see.
  It is not one of the three pipelines; it is the ablation that separates the
  accuracy lost to qubit-driven dimensionality reduction from the effect of the
  quantum kernel itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import QuantumConfig
from .index import VectorIndex
from .quantum.feature_maps import encode_states
from .quantum.similarity import QuantumSimilarity
from .reduction import PCAReducer


@dataclass
class Ranked:
    indices: np.ndarray
    scores: np.ndarray
    timings: dict = field(default_factory=dict)       # milliseconds
    stats: dict = field(default_factory=dict)
    classical_scores: np.ndarray | None = None


class QuantumSpace:
    """PCA reducer plus cached document features and statevectors for one config."""

    def __init__(self, doc_embeddings: np.ndarray, cfg: QuantumConfig):
        t0 = time.perf_counter()
        self.cfg = cfg
        self.layers = 1 if cfg.encoding == "amplitude" else cfg.layers
        self.reducer = PCAReducer(cfg.feature_dim, angle_range=cfg.encoding != "amplitude",
                                  seed=cfg.seed)
        self.doc_features = self.reducer.fit_transform(doc_embeddings)
        t1 = time.perf_counter()
        self.doc_states = encode_states(self.doc_features, cfg.encoding, cfg.n_qubits,
                                        self.layers, cfg.bandwidth)
        t2 = time.perf_counter()
        self.build_ms = {"pca": (t1 - t0) * 1e3, "encode_corpus": (t2 - t1) * 1e3}

    def encode_query(self, q_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feat = self.reducer.transform(q_emb)
        state = encode_states(feat[None, :], self.cfg.encoding, self.cfg.n_qubits,
                              self.layers, self.cfg.bandwidth)[0]
        return feat, state

    @property
    def nbytes(self) -> int:
        return int(self.doc_states.nbytes + self.doc_features.nbytes)


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    # Stable sort: ties keep the incoming order (for the hybrid, the classical rank).
    return np.argsort(-scores, kind="stable")[:k]


class ClassicalRetriever:
    name = "classical"

    def __init__(self, index: VectorIndex):
        self.index = index

    def retrieve(self, q_emb: np.ndarray, k: int) -> Ranked:
        t0 = time.perf_counter()
        idx, scores = self.index.search(q_emb, k)
        return Ranked(idx, scores, {"search": (time.perf_counter() - t0) * 1e3},
                      classical_scores=scores)


class QuantumRetriever:
    name = "quantum"

    def __init__(self, space: QuantumSpace):
        self.space = space

    def retrieve(self, q_emb: np.ndarray, k: int, cfg: QuantumConfig) -> Ranked:
        t0 = time.perf_counter()
        q_feat, q_state = self.space.encode_query(q_emb)
        t1 = time.perf_counter()
        sim = QuantumSimilarity.from_config(cfg)
        scores = sim.scores(q_state, self.space.doc_states, q_feat, self.space.doc_features)
        t2 = time.perf_counter()
        order = _top(scores, k)
        t3 = time.perf_counter()
        return Ranked(order, scores[order],
                      {"encode_query": (t1 - t0) * 1e3, "quantum_similarity": (t2 - t1) * 1e3,
                       "rank": (t3 - t2) * 1e3, "search": (t3 - t0) * 1e3},
                      sim.stats.to_dict())


class HybridRetriever:
    name = "hybrid"

    def __init__(self, index: VectorIndex, space: QuantumSpace):
        self.index = index
        self.space = space

    def retrieve(self, q_emb: np.ndarray, k: int, cfg: QuantumConfig, candidates: int) -> Ranked:
        t0 = time.perf_counter()
        cand, cand_scores = self.index.search(q_emb, max(candidates, k))
        t1 = time.perf_counter()
        q_feat, q_state = self.space.encode_query(q_emb)
        sim = QuantumSimilarity.from_config(cfg)
        scores = sim.scores(q_state, self.space.doc_states[cand], q_feat, self.space.doc_features[cand])
        if cfg.fusion > 0:
            # Late fusion: blend quantum fidelity with cosine mapped to [0, 1].
            scores = (1 - cfg.fusion) * scores + cfg.fusion * (np.asarray(cand_scores) + 1) / 2
        t2 = time.perf_counter()
        order = _top(scores, k)
        t3 = time.perf_counter()
        return Ranked(cand[order], scores[order],
                      {"coarse_filter": (t1 - t0) * 1e3, "quantum_rerank": (t2 - t1) * 1e3,
                       "rank": (t3 - t2) * 1e3, "search": (t3 - t0) * 1e3},
                      {**sim.stats.to_dict(), "candidates": int(len(cand))},
                      classical_scores=cand_scores[order])


class PCACosineRetriever:
    name = "pca_cosine"

    def __init__(self, doc_embeddings: np.ndarray, n_components: int, seed: int = 42):
        self.reducer = PCAReducer(n_components, angle_range=False, seed=seed)
        z = self.reducer.fit_transform(doc_embeddings)
        self.docs = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)

    def retrieve(self, q_emb: np.ndarray, k: int) -> Ranked:
        t0 = time.perf_counter()
        q = self.reducer.transform(q_emb)
        q = q / max(np.linalg.norm(q), 1e-12)
        scores = self.docs @ q
        order = _top(scores, k)
        return Ranked(order, scores[order], {"search": (time.perf_counter() - t0) * 1e3})


class RBFKernelRetriever:
    """Classical control: Gaussian (RBF) kernel on exactly the features a quantum map receives.

    It answers the question "does a quantum kernel do better than a standard classical
    kernel given identical compressed inputs?" gamma follows the common 'scale'
    heuristic, 1 / (n_features * Var(X)), computed on the corpus features.
    """
    name = "rbf"

    def __init__(self, space: QuantumSpace):
        self.space = space
        x = space.doc_features
        self.gamma = 1.0 / (x.shape[1] * max(float(x.var()), 1e-12))

    def scores(self, q_feat: np.ndarray, doc_feat: np.ndarray) -> np.ndarray:
        d2 = ((doc_feat - q_feat[None, :]) ** 2).sum(1)
        return np.exp(-self.gamma * d2)

    def retrieve(self, q_emb: np.ndarray, k: int, index: VectorIndex | None = None,
                 candidates: int | None = None) -> Ranked:
        t0 = time.perf_counter()
        q_feat = self.space.reducer.transform(q_emb)
        if index is not None and candidates:
            cand, _ = index.search(q_emb, max(candidates, k))
            s = self.scores(q_feat, self.space.doc_features[cand])
            order = _top(s, k)
            idx, sc = cand[order], s[order]
        else:
            s = self.scores(q_feat, self.space.doc_features)
            idx = _top(s, k)
            sc = s[idx]
        return Ranked(idx, sc, {"search": (time.perf_counter() - t0) * 1e3})


class BM25Retriever:
    """Lexical baseline (Okapi BM25, k1 = 0.9, b = 0.4 as in Anserini defaults)."""
    name = "bm25"

    def __init__(self, texts: list[str], k1: float = 0.9, b: float = 0.4):
        import re
        from collections import Counter
        self._tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
        docs = [self._tok(t) for t in texts]
        self.k1, self.b = k1, b
        self.len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avg = float(self.len.mean())
        self.tf = [Counter(d) for d in docs]
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: np.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.post: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.tf):
            for w, c in tf.items():
                self.post.setdefault(w, []).append((i, c))

    def retrieve_text(self, query: str, k: int) -> Ranked:
        t0 = time.perf_counter()
        s = np.zeros(len(self.len))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for w in set(self._tok(query)):
            for i, c in self.post.get(w, ()):
                s[i] += self.idf[w] * c * (self.k1 + 1) / (c + norm[i])
        idx = _top(s, k)
        return Ranked(idx, s[idx], {"search": (time.perf_counter() - t0) * 1e3})
