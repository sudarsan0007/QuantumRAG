"""Cross-validation against Qiskit. Skipped automatically when Qiskit is not installed."""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from quantumrag.quantum import qiskit_backend as qb  # noqa: E402
from quantumrag.quantum.feature_maps import encode_states  # noqa: E402


@pytest.mark.parametrize("enc,dim", [("angle", 3), ("iqp", 3), ("amplitude", 8)])
def test_qiskit_kernel_matches_pennylane(rng, enc, dim):
    x = rng.normal(size=(2, dim)) if enc == "amplitude" else rng.uniform(0, np.pi, size=(2, dim))
    layers = 1 if enc == "amplitude" else 2
    st = encode_states(x, enc, 3, layers)
    expected = abs(np.vdot(st[0], st[1])) ** 2
    assert np.isclose(qb.kernel_fidelity(x[0], x[1], 3, enc, layers), expected, atol=1e-8)
    assert np.isclose(qb.swap_test_fidelity(x[0], x[1], 3, enc, layers), expected, atol=1e-8)


def test_qasm_export(rng):
    x = rng.uniform(0, np.pi, size=(2, 3))
    text = qb.to_qasm(qb.swap_test_circuit(x[0], x[1], 3, "angle"))
    assert text.startswith("OPENQASM 2.0")
