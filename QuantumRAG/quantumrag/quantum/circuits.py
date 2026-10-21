"""Export similarity circuits as a structured gate layout for the dashboard."""
from __future__ import annotations

import numpy as np
import pennylane as qml

from .feature_maps import record_operations
from .similarity import grover_ops, get_feature_map

_SHORT = {
    "Hadamard": "H", "CNOT": "CNOT", "IsingZZ": "ZZ", "CSWAP": "CSWAP",
    "AmplitudeEmbedding": "AmpEnc", "StatePrep": "AmpEnc", "FlipSign": "Flip0",
    "RY": "RY", "RZ": "RZ",
}


def _describe(op, group: str) -> dict:
    name = op.name
    dagger = False
    base = op
    if name.startswith("Adjoint(") and hasattr(op, "base"):
        base, dagger = op.base, True
    short = _SHORT.get(base.name, base.name)
    params = []
    if base.name not in ("AmplitudeEmbedding", "StatePrep", "FlipSign"):
        for p in base.parameters:
            arr = np.asarray(p, dtype=float).ravel()
            if arr.size == 1:
                params.append(round(float(arr[0]) * (-1 if dagger and base.name in ("RY", "RZ", "IsingZZ") else 1), 4))
    return {"gate": short + ("†" if dagger and short not in ("H", "CNOT", "CSWAP") else ""),
            "wires": [int(w) for w in op.wires], "params": params, "group": group}


def _layout(ops: list[dict], n_wires: int) -> int:
    """Greedy ASAP column assignment; multi-qubit gates block the wires they span."""
    frontier = [0] * n_wires
    for op in ops:
        if op.get("barrier"):
            frontier = [max(frontier)] * n_wires
            continue
        lo, hi = min(op["wires"]), max(op["wires"])
        col = max(frontier[lo:hi + 1])
        op["col"] = col
        for w in range(lo, hi + 1):
            frontier[w] = col + 1
    return max(frontier) if frontier else 0


def circuit_layout(method: str, encoding: str, n_qubits: int, layers: int,
                   bandwidth: float, x, y) -> dict:
    layers = 1 if encoding == "amplitude" else layers
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ops: list[dict] = []
    if method == "swap":
        n_wires = 2 * n_qubits + 1
        labels = ["anc"] + [f"q{i}" for i in range(n_qubits)] + [f"d{i}" for i in range(n_qubits)]
        ops += [_describe(o, "U(query)") for o in record_operations(encoding, x, n_qubits, layers, bandwidth, 1)]
        ops += [_describe(o, "U(doc)") for o in record_operations(encoding, y, n_qubits, layers, bandwidth, n_qubits + 1)]
        ops.append({"barrier": True})
        ops.append({"gate": "H", "wires": [0], "params": [], "group": "SWAP test"})
        for i in range(n_qubits):
            ops.append({"gate": "CSWAP", "wires": [0, 1 + i, n_qubits + 1 + i], "params": [], "group": "SWAP test"})
        ops.append({"gate": "H", "wires": [0], "params": [], "group": "SWAP test"})
        measured = [0]
    else:
        n_wires = n_qubits
        labels = [f"q{i}" for i in range(n_qubits)]
        ops += [_describe(o, "U(query)") for o in record_operations(encoding, x, n_qubits, layers, bandwidth)]
        ops += [_describe(o, "U(doc)†") for o in record_operations(encoding, y, n_qubits, layers, bandwidth, adjoint=True)]
        if method == "iqae":
            fmap = get_feature_map(encoding)
            with qml.tape.QuantumTape() as tape:
                grover_ops(fmap, x, y, list(range(n_qubits)), layers, bandwidth)
            ops.append({"barrier": True})
            ops += [_describe(o, "Grover Q (one power)") for o in tape.operations]
        measured = list(range(n_qubits))
    depth = _layout(ops, n_wires)
    ops = [o for o in ops if not o.get("barrier")]
    groups = []
    for op in ops:
        if not groups or groups[-1]["name"] != op["group"]:
            groups.append({"name": op["group"], "start": op["col"], "end": op["col"]})
        groups[-1]["start"] = min(groups[-1]["start"], op["col"])
        groups[-1]["end"] = max(groups[-1]["end"], op["col"])
    merged = []
    for g in groups:
        if merged and g["start"] <= merged[-1]["end"] and g["name"] != merged[-1]["name"]:
            merged[-1]["name"] += " and " + g["name"]
            merged[-1]["end"] = max(merged[-1]["end"], g["end"])
        else:
            merged.append(g)
    groups = merged
    two_q = sum(1 for o in ops if len(o["wires"]) > 1)
    return {"method": method, "encoding": encoding, "n_wires": n_wires, "wire_labels": labels,
            "ops": ops, "depth": depth, "gate_count": len(ops), "two_qubit_gates": two_q,
            "groups": groups, "measured": measured}
