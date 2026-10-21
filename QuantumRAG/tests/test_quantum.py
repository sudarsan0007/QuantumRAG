"""Feature maps, fidelity estimators and IQAE."""
import numpy as np
import pennylane as qml
import pytest

from quantumrag.quantum.feature_maps import encode_states, record_operations
from quantumrag.quantum.iqae import grover_good_probability, iterative_amplitude_estimation
from quantumrag.quantum.similarity import (QuantumSimilarity, fidelity_matrix, grover_qnode,
                                           grover_unitaries, kernel_qnode, swap_qnode)

ENCODINGS = [("angle", 4), ("iqp", 4), ("amplitude", 16)]


def _features(rng, enc, dim, m=6):
    return rng.normal(size=(m, dim)) if enc == "amplitude" else rng.uniform(0, np.pi, size=(m, dim))


@pytest.mark.parametrize("enc,dim", ENCODINGS)
def test_states_are_normalised(rng, enc, dim):
    st = encode_states(_features(rng, enc, dim), enc, 4)
    assert st.shape == (6, 16)
    assert np.allclose(np.linalg.norm(st, axis=1), 1.0)


@pytest.mark.parametrize("enc,dim", ENCODINGS)
def test_self_fidelity_is_one(rng, enc, dim):
    st = encode_states(_features(rng, enc, dim), enc, 4)
    assert np.allclose(np.diag(fidelity_matrix(st)), 1.0)


@pytest.mark.parametrize("enc,dim", ENCODINGS)
def test_fidelity_is_symmetric_and_bounded(rng, enc, dim):
    g = fidelity_matrix(encode_states(_features(rng, enc, dim), enc, 4))
    assert np.allclose(g, g.T)
    assert g.min() >= 0 and g.max() <= 1 + 1e-12


@pytest.mark.parametrize("enc,dim", ENCODINGS)
def test_kernel_circuit_matches_statevector_overlap(rng, enc, dim):
    x = _features(rng, enc, dim, 2)
    st = encode_states(x, enc, 4)
    layers = 1 if enc == "amplitude" else 2
    p0 = kernel_qnode(enc, 4, layers, 1.0)(x[0], x[1])[0]
    assert np.isclose(p0, abs(np.vdot(st[1], st[0])) ** 2, atol=1e-10)


@pytest.mark.parametrize("enc,dim", [("angle", 3), ("iqp", 3), ("amplitude", 8)])
def test_swap_test_matches_fidelity(rng, enc, dim):
    x = _features(rng, enc, dim, 2)
    st = encode_states(x, enc, 3)
    layers = 1 if enc == "amplitude" else 2
    p0 = swap_qnode(enc, 3, layers, 1.0)(x[0], x[1])[0]
    assert np.isclose(2 * p0 - 1, abs(np.vdot(st[1], st[0])) ** 2, atol=1e-10)


def test_amplitude_fidelity_equals_squared_cosine(rng):
    x = rng.normal(size=(2, 16))
    st = encode_states(x, "amplitude", 4)
    cos = x[0] @ x[1] / np.linalg.norm(x[0]) / np.linalg.norm(x[1])
    assert np.isclose(abs(np.vdot(st[0], st[1])) ** 2, cos ** 2)


def test_feature_dimension_is_checked(rng):
    with pytest.raises(ValueError):
        encode_states(rng.uniform(size=(2, 5)), "angle", 4)
    with pytest.raises(ValueError):
        encode_states(rng.uniform(size=(2, 4)), "amplitude", 4)


def test_bandwidth_zero_limit_gives_identical_states(rng):
    x = rng.uniform(0, np.pi, size=(2, 4))
    st = encode_states(x * 1e-6, "angle", 4)
    assert abs(np.vdot(st[0], st[1])) ** 2 > 0.999999


@pytest.mark.parametrize("k", [1, 2, 4])
def test_grover_circuit_follows_amplification_law(rng, k):
    x = rng.uniform(0, np.pi, size=(2, 3))
    circuit = grover_qnode("angle", 3, 1, 1.0)
    a = circuit(x[0], x[1], 0)[0]
    assert np.isclose(circuit(x[0], x[1], k)[0], grover_good_probability(a, k), atol=1e-9)


def test_grover_unitaries_match_qnode(rng):
    x = rng.uniform(0, np.pi, size=(2, 3))
    a_mat, q_mat = grover_unitaries("iqp", 3, 1, 1.0, x[0], x[1])
    psi = np.linalg.matrix_power(q_mat, 3) @ a_mat[:, 0]
    assert np.isclose(abs(psi[0]) ** 2, grover_qnode("iqp", 3, 1, 1.0)(x[0], x[1], 3)[0], atol=1e-9)
    assert np.allclose(a_mat.conj().T @ a_mat, np.eye(8), atol=1e-9)


@pytest.mark.parametrize("a", [0.03, 0.27, 0.5, 0.81, 0.97])
def test_iqae_within_epsilon(a):
    rng = np.random.default_rng(0)
    res = iterative_amplitude_estimation(lambda k: grover_good_probability(a, k), epsilon=0.01,
                                         alpha=0.05, shots=100, rng=rng)
    assert abs(res.estimation - a) <= 0.01


def test_iqae_confidence_interval_coverage():
    """The interval must contain the true amplitude in at least 1 - alpha of runs."""
    misses, runs = 0, 0
    for a in np.linspace(0.05, 0.95, 10):
        for seed in range(20):
            res = iterative_amplitude_estimation(lambda k, a=a: grover_good_probability(a, k),
                                                 epsilon=0.01, alpha=0.05, shots=100,
                                                 rng=np.random.default_rng(seed))
            lo, hi = res.confidence_interval
            misses += not (lo - 1e-9 <= a <= hi + 1e-9)
            runs += 1
    assert misses / runs <= 0.05


def test_iqae_uses_fewer_calls_than_sampling_for_same_error():
    rng = np.random.default_rng(3)
    a = 0.4
    res = iterative_amplitude_estimation(lambda k: grover_good_probability(a, k), epsilon=0.001,
                                         alpha=0.05, shots=100, rng=rng)
    # Plain sampling needs ~a(1-a)/eps^2 = 240,000 shots for a standard error of 0.001.
    assert res.a_calls < a * (1 - a) / 0.001 ** 2
    assert max(res.powers) > 1


def test_iqae_rejects_bad_parameters():
    with pytest.raises(ValueError):
        iterative_amplitude_estimation(lambda k: 0.5, epsilon=0.0)
    with pytest.raises(ValueError):
        iterative_amplitude_estimation(lambda k: 0.5, alpha=1.5)


@pytest.mark.parametrize("method", ["kernel", "swap", "iqae"])
def test_statevector_and_circuit_modes_agree(rng, method):
    x = rng.uniform(0, np.pi, size=(5, 3))
    st = encode_states(x, "angle", 3, 2, 1.0)
    kw = dict(method=method, encoding="angle", n_qubits=3, layers=2, seed=7)
    sv = QuantumSimilarity(**kw).scores(st[0], st[1:])
    ci = QuantumSimilarity(execution="circuit", **kw).scores(st[0], st[1:], x[0], x[1:])
    tol = 0.011 if method == "iqae" else 1e-9
    assert np.allclose(sv, ci, atol=tol)


def test_finite_shots_converge_to_exact(rng):
    x = rng.uniform(0, np.pi, size=(30, 4))
    st = encode_states(x, "iqp", 4)
    exact = QuantumSimilarity("kernel", "iqp", 4).scores(st[0], st[1:])
    noisy = QuantumSimilarity("kernel", "iqp", 4, shots=20000, seed=1).scores(st[0], st[1:])
    swap = QuantumSimilarity("swap", "iqp", 4, shots=20000, seed=1).scores(st[0], st[1:])
    assert np.max(np.abs(noisy - exact)) < 0.02
    assert np.max(np.abs(swap - exact)) < 0.04


def test_shot_noise_is_reproducible_with_seed(rng):
    st = encode_states(rng.uniform(0, np.pi, size=(10, 4)), "angle", 4)
    a = QuantumSimilarity("kernel", "angle", 4, shots=128, seed=5).scores(st[0], st[1:])
    b = QuantumSimilarity("kernel", "angle", 4, shots=128, seed=5).scores(st[0], st[1:])
    assert np.array_equal(a, b)


def test_swap_stats_report_2n_plus_1_qubits(rng):
    st = encode_states(rng.uniform(0, np.pi, size=(4, 4)), "angle", 4)
    sim = QuantumSimilarity("swap", "angle", 4)
    sim.scores(st[0], st[1:])
    assert sim.stats.qubits_per_circuit == 9 and sim.stats.circuits_executed == 3


def test_adjoint_recording_reverses_gate_order():
    x = np.array([0.1, 0.2, 0.3])
    fwd = record_operations("angle", x, 3, layers=1)
    adj = record_operations("angle", x, 3, layers=1, adjoint=True)
    assert len(fwd) == len(adj)
    with qml.tape.QuantumTape() as tape:
        for op in fwd + adj:
            qml.apply(op)
    assert np.allclose(qml.matrix(tape, wire_order=[0, 1, 2]), np.eye(8), atol=1e-9)
