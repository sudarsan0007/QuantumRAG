"""Dense text embeddings shared by all three pipelines.

Primary backend: Sentence-Transformers all-MiniLM-L6-v2 (384-d), as specified in
the methodology. If the package or model weights are unavailable (for example on
an offline machine), the embedder falls back to a deterministic latent semantic
analysis model (TF-IDF followed by truncated SVD to 384 dimensions) so that the
system still runs end to end. The active backend is always reported by the API,
the dashboard and every benchmark file, so results are never silently mixed.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, SEED

log = logging.getLogger(__name__)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def find_local_minilm() -> Path | None:
    """Locate all-MiniLM-L6-v2 weights on disk (QRAG_MINILM_DIR or the HF cache)."""
    cands = []
    if os.environ.get("QRAG_MINILM_DIR"):
        cands.append(Path(os.environ["QRAG_MINILM_DIR"]))
    cands.append(Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2")
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    cands.extend(sorted(hub.glob("models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*")))
    for c in cands:
        if (c / "model.safetensors").exists() and (c / "vocab.txt").exists() and (c / "config.json").exists():
            return c
    return None


class Embedder:
    """Maps text to L2-normalised 384-dimensional vectors."""

    def __init__(self, backend: str = "auto", model_name: str = EMBEDDING_MODEL,
                 dim: int = EMBEDDING_DIM, cache_dir: Path = CACHE_DIR):
        self.model_name = model_name
        self.dim = dim
        self.cache_dir = Path(cache_dir)
        self._model = None
        self._lsa = None
        self.backend = None
        self.runtime = None
        backend = os.environ.get("QRAG_EMBEDDER", backend)
        if backend in ("auto", "minilm"):
            try:
                from sentence_transformers import SentenceTransformer  # noqa: WPS433
                self._model = SentenceTransformer(model_name)
                self.dim = int(self._model.get_sentence_embedding_dimension())
                self.backend = "minilm"
                self.runtime = "sentence-transformers"
            except Exception as exc:  # pragma: no cover - depends on environment
                local = find_local_minilm()
                if local is not None:
                    # Same weights, run with NumPy (no PyTorch needed).
                    from .minilm_numpy import MiniLMNumpy
                    self._model = MiniLMNumpy(local)
                    self.dim = self._model.dim
                    self.backend = "minilm"
                    self.runtime = "numpy"
                elif backend == "minilm":
                    raise
                else:
                    log.warning("Sentence-Transformers unavailable (%s); using LSA fallback", exc)
        if self.backend is None:
            self.backend = "lsa"

    # ------------------------------------------------------------------ public
    @property
    def description(self) -> str:
        if self.backend == "minilm":
            return f"Sentence-Transformers {self.model_name.split('/')[-1]} ({self.dim}-d)"
        return f"LSA fallback: TF-IDF + truncated SVD ({self.dim}-d)"

    def fit(self, corpus: list[str]) -> "Embedder":
        """Only the LSA fallback needs fitting; MiniLM is pre-trained."""
        if self.backend == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                  ngram_range=(1, 2), min_df=1, max_features=60000)
            tfidf = vec.fit_transform(corpus)
            n_comp = min(self.dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
            svd = TruncatedSVD(n_components=n_comp, random_state=SEED, algorithm="arpack")
            svd.fit(tfidf)
            self._lsa = (vec, svd)
            self.dim = n_comp
        return self

    def embed(self, texts: list[str] | str, batch_size: int = 64) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if self.backend == "minilm":
            if self.runtime == "numpy":
                out = self._model.encode(texts, batch_size=32)
            else:
                out = self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                         normalize_embeddings=True, show_progress_bar=False)
        else:
            if self._lsa is None:
                raise RuntimeError("LSA embedder must be fitted on the corpus first")
            vec, svd = self._lsa
            out = svd.transform(vec.transform(texts))
        out = _l2_normalize(out)
        return out[0] if single else out

    def embed_corpus(self, texts: list[str]) -> np.ndarray:
        """Embed a corpus with an on-disk cache keyed by backend and content."""
        h = hashlib.sha1((self.backend + self.model_name + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / f"emb_{self.backend}_{h}.npy"
        if self.backend == "minilm" and path.exists():
            return np.load(path)
        emb = self.embed(texts)
        if self.backend == "minilm":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb
