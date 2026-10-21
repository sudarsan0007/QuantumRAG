"""QuantumRAG engine: owns the corpus, embeddings, index and quantum spaces."""
from __future__ import annotations

import platform
import threading
import time
from pathlib import Path

import numpy as np

from .config import DEFAULT_DATASET, QuantumConfig, SEED
from .data import Document, Query, chunk_text, load_dataset
from .embedding import Embedder
from .generator import LLMGenerator
from .index import VectorIndex
from .quantum.circuits import circuit_layout
from .retrievers import (BM25Retriever, ClassicalRetriever, HybridRetriever, PCACosineRetriever,
                         QuantumRetriever, QuantumSpace, Ranked, RBFKernelRetriever)


class QuantumRAG:
    def __init__(self, dataset: Path | str = DEFAULT_DATASET, max_docs: int | None = None,
                 embedder: Embedder | None = None):
        self.dataset_path = Path(dataset)
        self.embedder = embedder or Embedder()
        docs, queries, meta = load_dataset(self.dataset_path, max_docs)
        self.meta = meta
        self.queries: list[Query] = queries
        self.corpus_kind = "benchmark"
        self._lock = threading.RLock()
        self._spaces: dict[tuple, QuantumSpace] = {}
        self._pca_cos: dict[int, PCACosineRetriever] = {}
        self.generator = LLMGenerator(self.embedder)
        self._load_documents(docs, refit=True)

    # ------------------------------------------------------------- corpus
    def _load_documents(self, docs: list[Document], refit: bool) -> None:
        t0 = time.perf_counter()
        self.documents = docs
        texts = [f"{d.title}. {d.text}" if d.topic else d.text for d in docs]
        if refit:
            self.embedder.fit(texts)
        self.doc_embeddings = self.embedder.embed_corpus(texts)
        self.index = VectorIndex().build(self.doc_embeddings)
        self.classical = ClassicalRetriever(self.index)
        self._bm25 = None
        self._texts = texts
        self._spaces.clear()
        self._pca_cos.clear()
        self.index_build_ms = (time.perf_counter() - t0) * 1e3

    def add_documents(self, items: list[dict], replace: bool = False) -> int:
        """Add user documents (long texts are chunked) and rebuild every index.

        replace=True searches only the given documents. The benchmark's labelled
        queries no longer apply then, so they are cleared until reset() is called.
        """
        with self._lock:
            new = []
            base = [] if replace else self.documents
            for item in items:
                title = (item.get("title") or "Untitled document").strip()
                topic = (item.get("topic") or "User upload").strip()
                chunks = chunk_text(item["text"])
                for ci, chunk in enumerate(chunks):
                    did = f"user-{len(base) + len(new):05d}"
                    name = title if len(chunks) == 1 else f"{title}, part {ci + 1}"
                    new.append(Document(id=did, title=name, text=chunk, topic=topic))
            if new:
                if replace:
                    self.queries = []
                    self.corpus_kind = "custom"
                self._load_documents(base + new, refit=replace or self.embedder.backend == "lsa")
            return len(new)

    def reset(self) -> None:
        with self._lock:
            docs, queries, _ = load_dataset(self.dataset_path)
            self.queries = queries
            self.corpus_kind = "benchmark"
            self._load_documents(docs, refit=True)

    # ------------------------------------------------------- quantum space
    def space(self, cfg: QuantumConfig) -> QuantumSpace:
        key = cfg.cache_key()
        with self._lock:
            if key not in self._spaces:
                self._spaces[key] = QuantumSpace(self.doc_embeddings, cfg)
            return self._spaces[key]

    def pca_cosine(self, n_components: int) -> PCACosineRetriever:
        with self._lock:
            if n_components not in self._pca_cos:
                self._pca_cos[n_components] = PCACosineRetriever(self.doc_embeddings, n_components, SEED)
            return self._pca_cos[n_components]

    # -------------------------------------------------------------- search
    def embed_query(self, text: str) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        v = self.embedder.embed(text)
        return v, (time.perf_counter() - t0) * 1e3

    def retrieve(self, pipeline: str, q_emb: np.ndarray, k: int = 5,
                 cfg: QuantumConfig | None = None, candidates: int = 50) -> Ranked:
        cfg = (cfg or QuantumConfig()).validate()
        if pipeline == "classical":
            return self.classical.retrieve(q_emb, k)
        if pipeline == "quantum":
            return QuantumRetriever(self.space(cfg)).retrieve(q_emb, k, cfg)
        if pipeline == "hybrid":
            return HybridRetriever(self.index, self.space(cfg)).retrieve(q_emb, k, cfg, candidates)
        if pipeline == "pca_cosine":
            return self.pca_cosine(cfg.feature_dim).retrieve(q_emb, k)
        if pipeline == "rbf":
            return RBFKernelRetriever(self.space(cfg)).retrieve(q_emb, k)
        if pipeline == "hybrid_rbf":
            return RBFKernelRetriever(self.space(cfg)).retrieve(q_emb, k, self.index, candidates)
        raise ValueError(f"unknown pipeline '{pipeline}'")

    def bm25(self, query: str, k: int) -> Ranked:
        """Lexical baseline; needs the query text rather than its embedding."""
        with self._lock:
            if self._bm25 is None:
                self._bm25 = BM25Retriever(self._texts)
        return self._bm25.retrieve_text(query, k)

    def search(self, query: str, pipelines=("classical", "quantum", "hybrid"), k: int = 5,
               cfg: QuantumConfig | None = None, candidates: int = 50) -> dict:
        cfg = (cfg or QuantumConfig()).validate()
        q_emb, embed_ms = self.embed_query(query)
        out = {"query": query, "embedding_ms": embed_ms, "config": cfg.to_dict(),
               "candidates": candidates, "k": k, "results": {}}
        cls_rank = None
        for p in pipelines:
            r = self.retrieve(p, q_emb, k, cfg, candidates)
            if p == "classical":
                cls_rank = {int(i): n + 1 for n, i in enumerate(r.indices)}
            out["results"][p] = self._format(r)
        if cls_rank is None and ("quantum" in pipelines or "hybrid" in pipelines):
            r = self.classical.retrieve(q_emb, max(k, candidates))
            cls_rank = {int(i): n + 1 for n, i in enumerate(r.indices)}
        for p, res in out["results"].items():
            for item in res["items"]:
                item["classical_rank"] = cls_rank.get(item["index"]) if cls_rank else None
        if any(p in pipelines for p in ("quantum", "hybrid")):
            sp = self.space(cfg)
            out["quantum_space"] = {"explained_variance": sp.reducer.explained_variance,
                                    "feature_dim": cfg.feature_dim,
                                    "corpus_encode_ms": sp.build_ms["encode_corpus"],
                                    "state_bytes": sp.nbytes}
        return out

    def _format(self, r: Ranked) -> dict:
        items = []
        for n, (i, s) in enumerate(zip(r.indices, r.scores)):
            d = self.documents[int(i)]
            item = {"rank": n + 1, "index": int(i), "score": float(np.nan_to_num(s)), **d.to_dict(snippet=320)}
            if r.classical_scores is not None:
                item["cosine"] = float(r.classical_scores[n])
            items.append(item)
        return {"items": items, "timings_ms": r.timings, "stats": r.stats}

    def answer(self, query: str, pipeline: str = "hybrid", k: int = 5,
               cfg: QuantumConfig | None = None, candidates: int = 50,
               provider: str | None = None) -> dict:
        q_emb, embed_ms = self.embed_query(query)
        r = self.retrieve(pipeline, q_emb, k, cfg, candidates)
        docs = [self.documents[int(i)] for i in r.indices]
        gen = self.generator.generate(query, docs, [float(s) for s in r.scores], provider)
        return {"query": query, "pipeline": pipeline, **gen,
                "sources": [{"n": n + 1, **d.to_dict(snippet=240), "score": float(s)}
                            for n, (d, s) in enumerate(zip(docs, r.scores))],
                "retrieval_ms": r.timings.get("search"), "embedding_ms": embed_ms}

    # ------------------------------------------------------------- circuit
    def circuit(self, cfg: QuantumConfig, query: str | None = None, doc_index: int | None = None) -> dict:
        cfg = cfg.validate()
        sp = self.space(cfg)
        if query:
            q_emb, _ = self.embed_query(query)
            q_feat, q_state = sp.encode_query(q_emb)
            if doc_index is None:
                doc_index = int(self.classical.retrieve(q_emb, 1).indices[0])
        else:
            q_feat, q_state = sp.doc_features[0], sp.doc_states[0]
            doc_index = 1 if doc_index is None else doc_index
        d_feat = sp.doc_features[doc_index]
        layout = circuit_layout(cfg.method, cfg.encoding, cfg.n_qubits, cfg.layers,
                                cfg.bandwidth, q_feat, d_feat)
        fid = float(abs(np.vdot(sp.doc_states[doc_index], q_state)) ** 2)
        layout.update(document={"index": doc_index, "title": self.documents[doc_index].title},
                      fidelity=fid, query_features=[float(v) for v in q_feat[:16]],
                      doc_features=[float(v) for v in d_feat[:16]],
                      query_probabilities=[float(abs(a) ** 2) for a in q_state[:256]])
        return layout

    # -------------------------------------------------------------- status
    def status(self) -> dict:
        from .index import HAVE_FAISS
        from .quantum.qiskit_backend import HAVE_QISKIT, QISKIT_VERSION
        import pennylane as qml
        return {
            "documents": len(self.documents), "queries": len(self.queries),
            "corpus": self.corpus_kind,
            "dataset": self.meta.get("name", self.dataset_path.name),
            "embedder": self.embedder.description, "embedder_backend": self.embedder.backend,
            "embedding_dim": int(self.doc_embeddings.shape[1]),
            "index": self.index.backend, "faiss": HAVE_FAISS,
            "pennylane": qml.__version__, "device": "default.qubit",
            "qiskit": QISKIT_VERSION if HAVE_QISKIT else None,
            "llm": {"provider": self.generator.provider, "model": self.generator.model_name},
            "python": platform.python_version(),
            "cached_quantum_spaces": len(self._spaces),
        }
