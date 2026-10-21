"""Quantum similarity estimators: kernel fidelity, SWAP test and IQAE.

All three estimate the same quantity, the state fidelity
    K(x, y) = |<psi(x)|psi(y)>|^2,
but with different circuits and resource costs:

* kernel  - compute-uncompute circuit U(y)^dagger U(x) on N qubits; K = P(|0...0>).
* swap    - SWAP test on 2N+1 qubits; P(ancilla = 0) = (1 + K) / 2.
* iqae    - Iterative Amplitude Estimation on A = U(y)^dagger U(x), good state |0...0>.

Two execution modes are provided:

* statevector - document states are simulated once by the PennyLane feature-map
                circuit and cached; the measurement statistics of the similarity
                circuit are then computed from those states. This is exact and is
                what makes full-corpus quantum retrieval tractable on a simulator.
* circuit     - the full similarity circuit is built and executed in PennyLane for
                every query/document pair. Slower, used for verification, for the
                circuit view and for the hybrid re-ranker on small candidate sets.

The unit tests check that both modes agree to numerical precision.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pennylane as qml

from .feature_maps import get_feature_map
from .iqae import iterative_amplitude_estimation, grover_good_probability


# ----------------------------------------------------------------- QNodes
@lru_cache(maxsize=32)
def kernel_qnode(encoding: str, n: int, layers: int, bandwidth: float):
    fmap = get_feature_map(encoding)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(x, y):
        wires = list(range(n))
        fmap(x, wires=wires, layers=layers, bandwidth=bandwidth)
        qml.adjoint(fmap)(y, wires=wires, layers=layers, bandwidth=bandwidth)
        return qml.probs(wires=wires)

    return circuit


@lru_cache(maxsize=32)
def swap_qnode(encoding: str, n: int, layers: int, bandwidth: float):
    fmap = get_feature_map(encoding)
    dev = qml.device("default.qubit", wires=2 * n + 1)

    @qml.qnode(dev)
    def circuit(x, y):
        reg_a = list(range(1, n + 1))
        reg_b = list(range(n + 1, 2 * n + 1))
        fmap(x, wires=reg_a, layers=layers, bandwidth=bandwidth)
        fmap(y, wires=reg_b, layers=layers, bandwidth=bandwidth)
        qml.Hadamard(wires=0)
        for a, b in zip(reg_a, reg_b):
            qml.CSWAP(wires=[0, a, b])
        qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    return circuit


def _a_operator(fmap, x, y, wires, layers, bandwidth):
    fmap(x, wires=wires, layers=layers, bandwidth=bandwidth)
    qml.adjoint(fmap)(y, wires=wires, layers=layers, bandwidth=bandwidth)


def grover_ops(fmap, x, y, wires, layers, bandwidth):
    """Q = A S_0 A^dagger S_chi with the good state |0...0> (global phase dropped)."""
    zeros = [0] * len(wires)
    qml.FlipSign(zeros, wires=wires)                       # S_chi
    qml.adjoint(_a_operator)(fmap, x, y, wires, layers, bandwidth)
    qml.FlipSign(zeros, wires=wires)                       # S_0
    _a_operator(fmap, x, y, wires, layers, bandwidth)


@lru_cache(maxsize=32)
def grover_qnode(encoding: str, n: int, layers: int, bandwidth: float):
    fmap = get_feature_map(encoding)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(x, y, k):
        wires = list(range(n))
        _a_operator(fmap, x, y, wires, layers, bandwidth)
        for _ in range(k):
            grover_ops(fmap, x, y, wires, layers, bandwidth)
        return qml.probs(wires=wires)

    return circuit


def grover_unitaries(encoding: str, n: int, layers: int, bandwidth: float, x, y):
    """Return (A, Q) as matrices extracted from the PennyLane circuits."""
    fmap = get_feature_map(encoding)
    wires = list(range(n))
    a_mat = qml.matrix(_a_operator, wire_order=wires)(fmap, x, y, wires, layers, bandwidth)
    q_mat = qml.matrix(grover_ops, wire_order=wires)(fmap, x, y, wires, layers, bandwidth)
    return np.asarray(a_mat), np.asarray(q_mat)


# ------------------------------------------------------------ estimator
@dataclass
class SimilarityStats:
    circuits_executed: int = 0
    shots_used: int = 0
    oracle_queries: int = 0
    qubits_per_circuit: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class QuantumSimilarity:
    """Estimate fidelity between one query state and many document states."""

    def __init__(self, method: str = "kernel", encoding: str = "angle", n_qubits: int = 4,
                 layers: int = 2, bandwidth: float = 1.0, shots: int | None = None,
                 execution: str = "statevector", iqae_epsilon: float = 0.01,
                 iqae_alpha: float = 0.05, iqae_shots: int = 100, seed: int = 42):
        self.method = method
        self.encoding = encoding
        self.n = n_qubits
        self.layers = 1 if encoding == "amplitude" else layers
        self.bandwidth = bandwidth
        self.shots = shots
        self.execution = execution
        self.iqae_epsilon = iqae_epsilon
        self.iqae_alpha = iqae_alpha
        self.iqae_shots = iqae_shots
        self.rng = np.random.default_rng(seed)
        self.stats = SimilarityStats()

    @classmethod
    def from_config(cls, cfg) -> "QuantumSimilarity":
        return cls(method=cfg.method, encoding=cfg.encoding, n_qubits=cfg.n_qubits,
                   layers=cfg.layers, bandwidth=cfg.bandwidth, shots=cfg.shots,
                   execution=cfg.execution, iqae_epsilon=cfg.iqae_epsilon,
                   iqae_alpha=cfg.iqae_alpha, iqae_shots=cfg.iqae_shots, seed=cfg.seed)

    @property
    def qubits_per_circuit(self) -> int:
        return 2 * self.n + 1 if self.method == "swap" else self.n

    # ---------------------------------------------------------------- API
    def scores(self, q_state: np.ndarray, d_states: np.ndarray,
               q_feat: np.ndarray | None = None, d_feats: np.ndarray | None = None) -> np.ndarray:
        """Fidelity estimates for one query against m documents."""
        self.stats = SimilarityStats(qubits_per_circuit=self.qubits_per_circuit)
        if self.execution == "circuit":
            if q_feat is None or d_feats is None:
                raise ValueError("circuit execution needs the classical feature vectors")
            return self._scores_circuit(q_feat, d_feats)
        exact = np.abs(d_states.conj() @ q_state) ** 2
        exact = np.clip(exact.real, 0.0, 1.0)
        return self._measure(exact)

    # ----------------------------------------------------- statevector mode
    def _measure(self, fidelity: np.ndarray) -> np.ndarray:
        m = len(fidelity)
        self.stats.circuits_executed = m
        if self.method == "iqae":
            return np.array([self._iqae(lambda k, f=f: grover_good_probability(f, k)) for f in fidelity])
        if self.shots is None:
            return fidelity
        self.stats.shots_used = m * self.shots
        if self.method == "kernel":
            return self.rng.binomial(self.shots, fidelity) / self.shots
        # SWAP test: ancilla reads 0 with probability (1 + F) / 2.
        p0 = self.rng.binomial(self.shots, (1.0 + fidelity) / 2.0) / self.shots
        return np.clip(2.0 * p0 - 1.0, 0.0, 1.0)

    def _iqae(self, prob_good) -> float:
        res = iterative_amplitude_estimation(prob_good, epsilon=self.iqae_epsilon,
                                             alpha=self.iqae_alpha, shots=self.iqae_shots,
                                             rng=self.rng)
        self.stats.oracle_queries += res.oracle_queries
        self.stats.shots_used += res.total_shots
        return res.estimation

    # --------------------------------------------------------- circuit mode
    def _scores_circuit(self, q_feat: np.ndarray, d_feats: np.ndarray) -> np.ndarray:
        args = (self.encoding, self.n, self.layers, float(self.bandwidth))
        out = np.empty(len(d_feats))
        for i, d in enumerate(d_feats):
            if self.method == "kernel":
                p = float(kernel_qnode(*args)(q_feat, d)[0])
                out[i] = p if self.shots is None else self.rng.binomial(self.shots, p) / self.shots
            elif self.method == "swap":
                p0 = float(swap_qnode(*args)(q_feat, d)[0])
                if self.shots is not None:
                    p0 = self.rng.binomial(self.shots, p0) / self.shots
                out[i] = float(np.clip(2 * p0 - 1, 0.0, 1.0))
            else:
                a_mat, q_mat = grover_unitaries(*args, q_feat, d)
                psi = a_mat[:, 0]
                cache = {0: psi}

                def prob_good(k, cache=cache, q_mat=q_mat):
                    kmax = max(cache)
                    v = cache[kmax]
                    for j in range(kmax, k):
                        v = q_mat @ v
                        cache[j + 1] = v
                    return float(abs(cache[k][0]) ** 2)

                out[i] = self._iqae(prob_good)
            if self.shots is not None and self.method != "iqae":
                self.stats.shots_used += self.shots
        self.stats.circuits_executed = len(d_feats)
        return out


def fidelity_matrix(states_a: np.ndarray, states_b: np.ndarray | None = None) -> np.ndarray:
    """Exact Gram matrix of fidelities between two sets of states."""
    states_b = states_a if states_b is None else states_b
    return np.clip(np.abs(states_a.conj() @ states_b.T) ** 2, 0.0, 1.0)
