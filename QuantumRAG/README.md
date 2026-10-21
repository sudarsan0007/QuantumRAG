# QuantumRAG: Quantum Similarity Search for Retrieval-Augmented Generation

MCA Dissertation (PAAML698), VIT School of Computer Science Engineering and Information Systems.
Sudarsan M (25MAG0139), guided by Dr. Vanitha M.

QuantumRAG replaces or augments the retrieval stage of a RAG system with quantum similarity
search, and benchmarks it honestly against classical dense retrieval. It implements:

* **Three retrieval pipelines**: classical cosine (baseline), pure quantum, hybrid two-stage.
* **Three quantum feature maps** (PennyLane): angle, IQP and amplitude encoding.
* **Three fidelity estimators**: compute-uncompute kernel, SWAP test, Iterative Quantum Amplitude Estimation.
* **Answer generation** grounded in the retrieved passages (offline extractive, local Ollama, or any OpenAI-compatible API).
* **A benchmark suite**: Recall@K, NDCG@K, MRR, latency, memory, plus analyses that explain the gaps
  (PCA information loss, kernel concentration, estimator efficiency).
* **A web dashboard**: side-by-side comparison, live circuit diagrams, benchmark charts, corpus manager.
* **Qiskit cross-validation**: the same circuits rebuilt in Qiskit, with OpenQASM export.

---

## 1. Setup

Python 3.10 to 3.12 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` model (about 90 MB) from Hugging Face.

## 2. Run

```bash
python run.py                         # dashboard at http://127.0.0.1:8000
python scripts/run_benchmark.py       # full benchmark, about 3 to 5 minutes on a laptop
python -m pytest                      # test suite
```

The dashboard has five pages:

| Page | What it does |
|---|---|
| Retrieve | Runs one question through all three pipelines side by side, shows latency breakdown, rank shifts versus classical, and a grounded answer with citations. Sample questions come from the benchmark, and their expected source is highlighted. |
| Circuit | Draws the exact circuit PennyLane simulates for the query and its nearest document: kernel, SWAP test or IQAE (with one Grover operator), with width, depth, gate counts and the encoded query state. |
| Benchmarks | Results table, auto-generated findings, and charts for quality, latency, memory, PCA information, kernel concentration and estimator efficiency. Can re-run the suite. |
| Corpus | Upload PDF, Word (.docx), .txt or .md files, or paste text. Choose "Search only these files" to ask questions about your own documents, or add them to the benchmark corpus. Restore the benchmark corpus at any time. |
| Method | Plain-language description of the method and the active environment. Runs the Qiskit cross-check when Qiskit is installed. |

### Optional: a local LLM for generation

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.2
set QRAG_LLM=ollama          # macOS/Linux: export QRAG_LLM=ollama
python run.py
```

Or choose "Ollama, local LLM" in the Answer panel. For an OpenAI-compatible endpoint set
`QRAG_OPENAI_API_KEY`, optionally `QRAG_OPENAI_BASE_URL` and `QRAG_OPENAI_MODEL`. The default
extractive generator is fully offline, so no corpus data leaves the machine (security requirement).

---

### Asking questions about your own files

On the Corpus page, choose or drag in PDF, DOCX, TXT or MD files (up to 25 MB each), pick
"Search only these files", and click "Upload and index". Text is extracted page by page (PDF
passages are titled "file, page N"), split into overlapping passages of about 160 words, and all
three pipelines are rebuilt on it. Then ask questions on the Retrieve page. Scanned PDFs contain
images rather than text and need OCR first. "Restore benchmark corpus" brings back the SQuAD data
and its labelled questions, which the Benchmarks page needs.

## 3. Project layout

```
QuantumRAG/
  run.py                       start the dashboard
  requirements.txt
  api/server.py                FastAPI JSON API + static dashboard
  web/                         index.html, styles.css, app.js (no build step)
  quantumrag/
    config.py                  QuantumConfig, RetrievalConfig
    data.py                    Document, Query, dataset loading, chunking
    embedding.py               Embedder (MiniLM, LSA fallback)
    index.py                   VectorIndex (FAISS IndexFlatIP, NumPy fallback)
    reduction.py               PCAReducer
    retrievers.py              ClassicalRetriever, QuantumRetriever, HybridRetriever, PCACosineRetriever
    generator.py               ContextBuilder, LLMGenerator
    ingest.py                  text extraction from PDF, DOCX, TXT, MD uploads
    evaluation.py              metrics, benchmark runner, analyses
    engine.py                  QuantumRAG engine tying it together
    quantum/
      feature_maps.py          angle, IQP, amplitude encodings (PennyLane)
      similarity.py            kernel, SWAP test, IQAE estimators; QNodes
      iqae.py                  Iterative Amplitude Estimation (Grinko et al. 2021)
      circuits.py              circuit layout export for the dashboard
      qiskit_backend.py        Qiskit re-implementation, cross-check, QASM export
  scripts/
    build_dataset.py           builds data/squad_subset.json from SQuAD v1.1 dev
    run_benchmark.py           command-line benchmark
  data/squad_subset.json       600 SQuAD paragraphs, 3,082 labelled questions
  results/benchmark_latest.json
  tests/                       66 core tests + 4 Qiskit cross-validation tests
```

The modules map one-to-one onto the components in the report's UML and class diagrams:
Query Processor and Embedder (`embedding.py`), PCA Reducer (`reduction.py`), Quantum Encoder
(`feature_maps.py`), Similarity Engine (`similarity.py`, `iqae.py`), the three retrievers
(`retrievers.py`), Context Builder and LLM Generator (`generator.py`), Evaluation and Logging
(`evaluation.py`), Web/API Interface (`api/server.py`, `web/`).

---

## 4. Method in brief

**Dataset.** SQuAD v1.1 dev set, CC BY-SA 4.0. Each paragraph is a document; each question's
single relevant document is the paragraph it was written for. 600 paragraphs are sampled evenly
across all 48 articles (seed 42). Rebuild with a different size:
`python scripts/build_dataset.py --squad dev-v1.1.json --docs 1000`.

**Classical baseline.** all-MiniLM-L6-v2 embeddings (384-d, L2-normalised), exact inner-product
search (FAISS `IndexFlatIP`), which equals cosine similarity.

**Quantum encoding.** Embeddings are reduced by PCA. Angle and IQP use N components (one per
qubit), min-max scaled to [0, pi]. Amplitude encoding uses 2^N components.

* Angle: per layer RY(b x_i) RZ(b x_i) on each qubit, then a CNOT ring.
* IQP: per layer H on each qubit, RZ(b x_i), then IsingZZ(b x_i x_j / pi) on neighbouring pairs.
* Amplitude: the 2^N features, normalised, become the state amplitudes.

`b` is the kernel bandwidth (default 1).

**Fidelity estimators.** All estimate K(x, y) = |<psi(x)|psi(y)>|^2.

* Kernel: U(x) then U(y)^dagger on N qubits; K = P(|0...0>).
* SWAP test: 2N + 1 qubits; P(ancilla = 0) = (1 + K) / 2.
* IQAE: A = U(y)^dagger U(x), good state |0...0>; Grover powers Q^k A shrink the confidence interval
  on K with about 1/epsilon circuit calls instead of 1/epsilon^2 shots.

**Execution.** Document states are simulated once by the feature-map circuit on PennyLane
`default.qubit` and cached; each query is compared against them using the measurement
statistics of the chosen estimator (finite shots are sampled from that distribution when set).
The tests verify that this equals executing the full kernel, SWAP-test and Grover circuits pair
by pair. Full per-pair circuit execution is also selectable (use it with the hybrid pipeline).

**Hybrid.** Classical top-Kc shortlist (Kc = 50), quantum re-rank to top-Kq (5). Optional late
fusion `(1 - w) * fidelity + w * (cos + 1) / 2` is included as an extension.

**Control.** "PCA cosine" runs classical cosine on exactly the features the quantum encoder sees.
The difference between classical and PCA cosine is the cost of compression; the difference
between PCA cosine and the quantum pipeline is the effect of the quantum kernel itself.

---

## 5. Results (paper version)

All numbers below use **all-MiniLM-L6-v2** embeddings, five random corpora of 600 SQuAD v1.1
dev paragraphs (seeds 42, 7, 13, 21, 99) and 300 questions per seed (1,500 in total). They are
the numbers reported in the paper. Raw outputs: `results/study_latest.json`,
`results/study_per_query.json` and `results/extras_latest.json`.

| Configuration | R@1 | R@5 | MRR@10 | NDCG@10 [95% CI] |
|---|---|---|---|---|
| Dense cosine (384-d) | 0.758 | 0.941 | 0.836 | 0.869 [0.857, 0.882] |
| BM25 | 0.875 | 0.967 | 0.916 | 0.932 [0.922, 0.942] |
| PCA cosine, 8-d | 0.239 | 0.556 | 0.374 | 0.450 [0.431, 0.469] |
| PCA cosine, 256-d | 0.761 | 0.941 | 0.837 | 0.869 [0.856, 0.882] |
| Classical RBF kernel, 8-d | 0.221 | 0.519 | 0.348 | 0.422 [0.403, 0.441] |
| Quantum angle, 8q | 0.218 | 0.510 | 0.343 | 0.414 [0.395, 0.433] |
| Quantum IQP, 8q | 0.209 | 0.489 | 0.328 | 0.400 [0.382, 0.420] |
| Quantum amplitude, 8q | 0.760 | 0.940 | 0.836 | 0.867 [0.855, 0.881] |
| Hybrid angle, 8q | 0.251 | 0.587 | 0.391 | 0.475 [0.457, 0.494] |
| Hybrid angle 8q, fusion 0.5 | 0.413 | 0.760 | 0.559 | 0.636 [0.619, 0.654] |
| Hybrid amplitude 8q, IQAE | 0.761 | 0.943 | 0.837 | 0.869 [0.856, 0.882] |

The full 20-row table, the paired tests (Holm-corrected) and the latency table are in the paper
and in the JSON files.

### Findings

1. **Compression is the main bottleneck.** Eight PCA components keep 23% of the embedding
   variance; cosine on them falls from 0.869 to 0.450 NDCG@10.
2. **The quantum kernels do not beat classical kernels on the same inputs.** Angle and IQP are
   slightly below 8-d cosine (-0.036, -0.050, p < 0.001) and no better than an RBF kernel.
   A bandwidth sweep (b = 0.25 to 2) does not change this.
3. **Amplitude encoding matches dense cosine for a classical reason.** Its fidelity is the squared
   cosine of 256 PCA features, so its ranking equals the 256-d PCA cosine control on 99.5% of
   questions.
4. **Hybrid re-ranking bounds quantum work but lowers quality with a weak kernel.**
5. **IQAE preserves exact rankings; SWAP with 1,024 shots costs 0.019 NDCG@10.** IQAE needs fewer
   circuit calls than sampling for the same error.
6. **Timings are simulator-relative** and make no claim about quantum hardware.

### Reproducing the paper

```bash
# 1. SQuAD v1.1 dev set
#    https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
# 2. MiniLM: installs via sentence-transformers, or point to a local copy of the model files
#    (model.safetensors, vocab.txt, config.json); the NumPy runtime in
#    quantumrag/minilm_numpy.py then runs the same weights without PyTorch.
export QRAG_MINILM_DIR=/path/to/all-MiniLM-L6-v2      # optional
python scripts/run_study.py  --squad dev-v1.1.json   # Tables I-II, Figs. 2, 3, 5, 6 (~25 min, 2 CPU cores)
python scripts/run_extras.py --squad dev-v1.1.json   # bandwidth sweep (Fig. 4) and Table IV
python scripts/make_figures.py                       # writes results/figures/fig2-6.png
```

Everything is seeded; a rerun gives identical quality metrics (timings vary with the machine).

---

## 6. Tests

```bash
python -m pytest -q
```

The MiniLM test runs when the model files are available (it checks the NumPy runtime against
the reference similarity scores published for the model).

Covered: state normalisation, symmetry and bounds of the kernel; kernel circuit, SWAP test and
Grover circuits against statevector fidelity for every encoding; amplitude fidelity equals squared
cosine; IQAE accuracy, confidence-interval coverage and shot efficiency; statevector versus
full-circuit agreement for all three estimators; seeded reproducibility; FAISS/NumPy index
exactness; PCA; metrics against hand-computed values; every retriever; generator fallback; and the
HTTP API. The Qiskit tests run automatically when Qiskit is installed.

## 7. Notes and limits

* Circuits run on a classical simulator; real hardware noise is not modelled.
* PCA is fitted on the corpus; adding documents refits it (the dashboard does this for you).
* Full-circuit execution is slow by design; use it with the hybrid pipeline and small qubit counts.

## References

Key method sources: Havlicek et al., *Supervised learning with quantum-enhanced feature spaces*,
Nature 567 (2019) for IQP feature maps; Grinko, Gacon, Zoufal and Woerner, *Iterative quantum
amplitude estimation*, npj Quantum Information 7 (2021); Buhrman et al., *Quantum fingerprinting*,
PRL 87 (2001) for the SWAP test; Thanasilp et al., *Exponential concentration in quantum kernel
methods*, Nature Communications 15 (2024); Rajpurkar et al., *SQuAD*, EMNLP (2016).
