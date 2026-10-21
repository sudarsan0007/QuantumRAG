"""PCA compression from 384-d embeddings to the qubit budget.

Angle and IQP encodings use one feature per qubit, so the embedding is reduced
to N components and min-max scaled to [0, pi] (rotation angles). Amplitude
encoding stores 2^N features in N qubits, so it uses 2^N components; those are
kept as signed values and L2-normalised by the state preparation itself.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .config import SEED


class PCAReducer:
    def __init__(self, n_components: int, angle_range: bool = True, seed: int = SEED):
        self.n_components = int(n_components)
        self.angle_range = angle_range
        self.seed = seed
        self._pca: PCA | None = None
        self._lo = None
        self._hi = None

    def fit(self, vectors: np.ndarray) -> "PCAReducer":
        # A small corpus (e.g. one uploaded PDF) has fewer documents than the encoding
        # wants components. Fit what the data supports and zero-pad the rest: padded
        # features are constant, so they add no information and do not change rankings.
        n = min(self.n_components, vectors.shape[0] - 1, vectors.shape[1])
        if n < 1:
            raise ValueError("at least two documents are needed for quantum retrieval")
        self.fitted_components = n
        self._pca = PCA(n_components=n, random_state=self.seed, svd_solver="full")
        z = self._pad(self._pca.fit_transform(vectors))
        # Robust range (1st-99th percentile) so outliers do not squash the angles.
        self._lo = np.percentile(z, 1, axis=0)
        self._hi = np.percentile(z, 99, axis=0)
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAReducer is not fitted")
        single = vectors.ndim == 1
        z = self._pad(self._pca.transform(vectors.reshape(1, -1) if single else vectors))
        if self.angle_range:
            span = np.where(self._hi - self._lo > 1e-9, self._hi - self._lo, 1.0)
            z = np.clip((z - self._lo) / span, 0.0, 1.0) * np.pi
        return z[0] if single else z

    def _pad(self, z: np.ndarray) -> np.ndarray:
        missing = self.n_components - z.shape[1]
        return np.hstack([z, np.zeros((z.shape[0], missing))]) if missing > 0 else z

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.fit(vectors).transform(vectors)

    @property
    def explained_variance(self) -> float:
        return float(np.nan_to_num(np.sum(self._pca.explained_variance_ratio_))) if self._pca is not None else 0.0

    @property
    def explained_variance_curve(self) -> list[float]:
        if self._pca is None:
            return []
        return [float(v) for v in np.nan_to_num(np.cumsum(self._pca.explained_variance_ratio_))]
