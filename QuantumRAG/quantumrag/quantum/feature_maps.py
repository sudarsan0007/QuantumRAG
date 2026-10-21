"""Parameterised quantum feature maps (PennyLane).

Each map U(x) prepares |psi(x)> = U(x)|0...0> from a classical feature vector.

* Angle encoding  - per layer: RY(b*x_i) RZ(b*x_i) on every qubit, then a CNOT ring.
* IQP encoding    - per layer: H on every qubit, RZ(b*x_i), then IsingZZ(b*x_i*x_j/pi)
                    on nearest-neighbour pairs (Havlicek et al. style, non-linear in x).
* Amplitude       - 2^N real features normalised into the N-qubit amplitudes.

``b`` is the kernel bandwidth. Scaling features down (b < 1) is a known remedy for
exponential concentration of fidelity kernels as the qubit count grows.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pennylane as qml


def _pairs(n: int) -> list[tuple[int, int]]:
    """Nearest-neighbour ring pairs; a line for 2 qubits."""
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]
    return [(i, (i + 1) % n) for i in range(n)]


def angle_map(x, wires, layers: int = 2, bandwidth: float = 1.0) -> None:
    n = len(wires)
    for _ in range(layers):
        for i, w in enumerate(wires):
            qml.RY(bandwidth * x[..., i], wires=w)
            qml.RZ(bandwidth * x[..., i], wires=w)
        if n > 1:
            for a, b in _pairs(n):
                qml.CNOT(wires=[wires[a], wires[b]])


def iqp_map(x, wires, layers: int = 2, bandwidth: float = 1.0) -> None:
    n = len(wires)
    for _ in range(layers):
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            qml.RZ(bandwidth * x[..., i], wires=w)
        for a, b in _pairs(n):
            qml.IsingZZ(bandwidth * x[..., a] * x[..., b] / np.pi, wires=[wires[a], wires[b]])


def amplitude_map(x, wires, layers: int = 1, bandwidth: float = 1.0) -> None:  # noqa: ARG001
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)


FEATURE_MAPS = {"angle": angle_map, "iqp": iqp_map, "amplitude": amplitude_map}


def get_feature_map(name: str):
    try:
        return FEATURE_MAPS[name]
    except KeyError as exc:
        raise ValueError(f"unknown encoding '{name}'") from exc


@lru_cache(maxsize=64)
def _state_qnode(encoding: str, n_qubits: int, layers: int, bandwidth: float):
    dev = qml.device("default.qubit", wires=n_qubits)
    fmap = get_feature_map(encoding)

    @qml.qnode(dev)
    def circuit(x):
        fmap(x, wires=list(range(n_qubits)), layers=layers, bandwidth=bandwidth)
        return qml.state()

    return circuit


def encode_states(features: np.ndarray, encoding: str, n_qubits: int,
                  layers: int = 2, bandwidth: float = 1.0, batch: int = 256) -> np.ndarray:
    """Run the feature-map circuit and return complex statevectors, shape (m, 2^n).

    Uses PennyLane parameter broadcasting so a whole batch of documents is
    simulated in one circuit execution.
    """
    features = np.atleast_2d(np.asarray(features, dtype=np.float64))
    expected = 2 ** n_qubits if encoding == "amplitude" else n_qubits
    if features.shape[1] != expected:
        raise ValueError(f"{encoding} encoding on {n_qubits} qubits needs {expected} features, "
                         f"got {features.shape[1]}")
    if encoding == "amplitude":
        norms = np.linalg.norm(features, axis=1)
        features = features.copy()
        features[norms < 1e-12] = 1.0  # all-zero vector cannot be normalised
    qnode = _state_qnode(encoding, n_qubits, int(layers), float(bandwidth))
    out = []
    for start in range(0, features.shape[0], batch):
        chunk = features[start:start + batch]
        st = np.asarray(qnode(chunk if chunk.shape[0] > 1 else chunk[0]))
        out.append(st.reshape(-1, 2 ** n_qubits))
    return np.vstack(out).astype(np.complex128)


def record_operations(encoding: str, x, n_qubits: int, layers: int = 2,
                      bandwidth: float = 1.0, wire_offset: int = 0, adjoint: bool = False):
    """Record the gate list of U(x) (or U(x)^dagger) without executing it."""
    fmap = get_feature_map(encoding)
    wires = list(range(wire_offset, wire_offset + n_qubits))
    with qml.tape.QuantumTape() as tape:
        if adjoint:
            qml.adjoint(fmap)(np.asarray(x), wires=wires, layers=layers, bandwidth=bandwidth)
        else:
            fmap(np.asarray(x), wires=wires, layers=layers, bandwidth=bandwidth)
    return list(tape.operations)
