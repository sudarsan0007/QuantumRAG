"""Optional Qiskit implementation of the same circuits, for cross-validation.

PennyLane is the primary simulator. This module rebuilds the kernel and SWAP-test
circuits gate-for-gate in Qiskit so the fidelities can be independently checked
on a second framework, and exports OpenQASM 2 for use on IBM hardware later.

Gate conventions match PennyLane: RY, RZ and RZZ (= IsingZZ) are all
exp(-i * theta/2 * P). Qiskit orders qubits little-endian, which permutes basis
labels but leaves state overlaps (fidelities) unchanged.

Install with:  pip install qiskit
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAVE_QISKIT = True
    QISKIT_VERSION = qiskit.__version__
except Exception:  # pragma: no cover
    HAVE_QISKIT = False
    QISKIT_VERSION = None


def _require():
    if not HAVE_QISKIT:
        raise RuntimeError("Qiskit is not installed; run `pip install qiskit`")


def _pairs(n: int):
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]
    return [(i, (i + 1) % n) for i in range(n)]


def feature_map_circuit(x, n: int, encoding: str, layers: int = 2,
                        bandwidth: float = 1.0) -> "QuantumCircuit":
    _require()
    x = np.asarray(x, dtype=float)
    qc = QuantumCircuit(n, name=f"U_{encoding}")
    if encoding == "angle":
        for _ in range(layers):
            for i in range(n):
                qc.ry(bandwidth * x[i], i)
                qc.rz(bandwidth * x[i], i)
            if n > 1:
                for a, b in _pairs(n):
                    qc.cx(a, b)
    elif encoding == "iqp":
        for _ in range(layers):
            for i in range(n):
                qc.h(i)
            for i in range(n):
                qc.rz(bandwidth * x[i], i)
            for a, b in _pairs(n):
                qc.rzz(bandwidth * x[a] * x[b] / np.pi, a, b)
    elif encoding == "amplitude":
        from qiskit.circuit.library import StatePreparation
        vec = x / max(np.linalg.norm(x), 1e-12)
        qc.append(StatePreparation(vec), list(range(n)))
    else:
        raise ValueError(f"unknown encoding '{encoding}'")
    return qc


def kernel_circuit(x, y, n, encoding, layers=2, bandwidth=1.0) -> "QuantumCircuit":
    """U(y)^dagger U(x); the fidelity is the probability of |0...0>."""
    qc = QuantumCircuit(n)
    qc.compose(feature_map_circuit(x, n, encoding, layers, bandwidth), inplace=True)
    qc.compose(feature_map_circuit(y, n, encoding, layers, bandwidth).inverse(), inplace=True)
    return qc


def swap_test_circuit(x, y, n, encoding, layers=2, bandwidth=1.0) -> "QuantumCircuit":
    """Ancilla is qubit 0; P(ancilla = 0) = (1 + F) / 2."""
    qc = QuantumCircuit(2 * n + 1)
    reg_a = list(range(1, n + 1))
    reg_b = list(range(n + 1, 2 * n + 1))
    qc.compose(feature_map_circuit(x, n, encoding, layers, bandwidth), qubits=reg_a, inplace=True)
    qc.compose(feature_map_circuit(y, n, encoding, layers, bandwidth), qubits=reg_b, inplace=True)
    qc.h(0)
    for a, b in zip(reg_a, reg_b):
        qc.cswap(0, a, b)
    qc.h(0)
    return qc


def kernel_fidelity(x, y, n, encoding, layers=2, bandwidth=1.0) -> float:
    _require()
    sv = Statevector.from_instruction(kernel_circuit(x, y, n, encoding, layers, bandwidth))
    return float(np.abs(sv.data[0]) ** 2)


def swap_test_fidelity(x, y, n, encoding, layers=2, bandwidth=1.0) -> float:
    _require()
    sv = Statevector.from_instruction(swap_test_circuit(x, y, n, encoding, layers, bandwidth))
    p0 = float(sv.probabilities([0])[0])
    return float(np.clip(2 * p0 - 1, 0.0, 1.0))


def to_qasm(qc) -> str:
    _require()
    from qiskit import qasm2, transpile
    basis = ["u", "cx", "rz", "ry", "h", "rzz", "cswap", "x"]
    return qasm2.dumps(transpile(qc, basis_gates=basis, optimization_level=0))


def cross_check(features: np.ndarray, n: int, encoding: str, layers: int = 2,
                bandwidth: float = 1.0, pairs: int = 5, seed: int = 0) -> dict:
    """Compare PennyLane and Qiskit fidelities on random document pairs."""
    from .similarity import kernel_qnode, swap_qnode

    _require()
    rng = np.random.default_rng(seed)
    layers = 1 if encoding == "amplitude" else layers
    rows = []
    for _ in range(pairs):
        i, j = rng.choice(len(features), size=2, replace=False)
        x, y = features[i], features[j]
        pl_k = float(kernel_qnode(encoding, n, layers, float(bandwidth))(x, y)[0])
        pl_s = float(np.clip(2 * swap_qnode(encoding, n, layers, float(bandwidth))(x, y)[0] - 1, 0, 1))
        qk_k = kernel_fidelity(x, y, n, encoding, layers, bandwidth)
        qk_s = swap_test_fidelity(x, y, n, encoding, layers, bandwidth)
        rows.append({"pair": [int(i), int(j)], "pennylane_kernel": pl_k, "qiskit_kernel": qk_k,
                     "pennylane_swap": pl_s, "qiskit_swap": qk_s})
    max_dev = max(max(abs(r["pennylane_kernel"] - r["qiskit_kernel"]),
                      abs(r["pennylane_swap"] - r["qiskit_swap"])) for r in rows)
    return {"qiskit_version": QISKIT_VERSION, "rows": rows, "max_abs_deviation": max_dev}
