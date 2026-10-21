"""Central configuration and shared constants."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

DEFAULT_DATASET = DATA_DIR / "squad_subset.json"
EMBEDDING_MODEL = os.environ.get("QRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
SEED = int(os.environ.get("QRAG_SEED", "42"))

ENCODINGS = ("angle", "iqp", "amplitude")
METHODS = ("kernel", "swap", "iqae")
PIPELINES = ("classical", "quantum", "hybrid")


@dataclass
class QuantumConfig:
    """Settings shared by the pure-quantum and hybrid retrievers."""

    n_qubits: int = 8
    encoding: str = "amplitude"      # angle | iqp | amplitude
    method: str = "kernel"           # kernel | swap | iqae
    layers: int = 2                  # repetitions of the feature map
    bandwidth: float = 1.0           # feature scaling; <1 counters kernel concentration
    shots: int | None = None         # None = exact (analytic) fidelity
    execution: str = "statevector"   # statevector | circuit
    iqae_epsilon: float = 0.01
    iqae_alpha: float = 0.05
    iqae_shots: int = 100
    fusion: float = 0.0              # hybrid only: weight of the classical cosine in the final score
    seed: int = SEED

    def validate(self) -> "QuantumConfig":
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.execution not in ("statevector", "circuit"):
            raise ValueError("execution must be 'statevector' or 'circuit'")
        max_q = 8 if self.encoding == "amplitude" else 12
        if not 2 <= self.n_qubits <= max_q:
            raise ValueError(f"n_qubits must be between 2 and {max_q} for {self.encoding} encoding")
        if not 1 <= self.layers <= 6:
            raise ValueError("layers must be between 1 and 6")
        if not 0.05 <= self.bandwidth <= 4.0:
            raise ValueError("bandwidth must be between 0.05 and 4.0")
        if not 0.0 <= self.fusion <= 1.0:
            raise ValueError("fusion must be between 0 and 1")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive")
        return self

    @property
    def feature_dim(self) -> int:
        """Number of PCA components the encoding consumes."""
        return 2 ** self.n_qubits if self.encoding == "amplitude" else self.n_qubits

    def cache_key(self) -> tuple:
        return (self.n_qubits, self.encoding, self.layers, round(self.bandwidth, 4))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidates: int = 50            # Kc for the hybrid pipeline
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
