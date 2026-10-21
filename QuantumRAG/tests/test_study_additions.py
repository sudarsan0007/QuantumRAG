"""Tests for the NumPy MiniLM runtime, the classical controls and the statistics."""
import numpy as np
import pytest

from quantumrag.embedding import find_local_minilm
from quantumrag.stats import bootstrap_ci, holm, paired_permutation_test


@pytest.mark.skipif(find_local_minilm() is None, reason="MiniLM weights not available")
def test_numpy_minilm_matches_sentence_transformers_reference():
    """Cosine scores published in the Sentence-Transformers documentation for this model."""
    from quantumrag.minilm_numpy import MiniLMNumpy
    m = MiniLMNumpy(find_local_minilm())
    pairs = [("The new movie is awesome", "The new movie is so great", 0.8939),
             ("The cat sits outside", "The cat plays in the garden", 0.6788),
             ("I love pasta", "Do you like pizza?", 0.5096),
             ("A man is playing guitar", "A woman watches TV", -0.0327)]
    for a, b, ref in pairs:
        e = m.encode([a, b])
        assert abs(float(e[0] @ e[1]) - ref) < 1e-3


def test_bert_tokenizer_basic_rules(tmp_path):
    from quantumrag.minilm_numpy import BertTokenizer
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", ",", "world", "!", "play", "##ing", "naive"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = BertTokenizer(tmp_path / "vocab.txt")
    ids = tok.encode("Hello, WORLD! Playing Naïve xyz")
    names = [vocab[i] for i in ids]
    assert names == ["[CLS]", "hello", ",", "world", "!", "play", "##ing", "naive", "[UNK]", "[SEP]"]


def test_permutation_test_detects_shift_and_null():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.2, 400)
    assert paired_permutation_test(a + 0.1, a, n_perm=2000) < 0.01
    assert paired_permutation_test(a, a.copy(), n_perm=500) == 1.0


def test_bootstrap_ci_covers_mean_and_holm_is_monotone():
    x = np.random.default_rng(1).uniform(0, 1, 500)
    lo, hi = bootstrap_ci(x, n_boot=1000)
    assert lo < x.mean() < hi and hi - lo < 0.1
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))


def test_classical_controls_run(engine):
    from quantumrag.config import QuantumConfig
    cfg = QuantumConfig(n_qubits=4, encoding="angle")
    q = engine.embedder.embed(engine.queries[0].text)
    for p in ("rbf", "hybrid_rbf"):
        r = engine.retrieve(p, q, 5, cfg, 10)
        assert len(r.indices) == 5 and np.all(np.isfinite(r.scores))
    r = engine.bm25(engine.queries[0].text, 5)
    assert len(r.indices) == 5 and r.scores[0] >= r.scores[-1]
