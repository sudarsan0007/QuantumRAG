"""Retrieval metrics and the reproducible benchmark suite."""
from __future__ import annotations

import json
import platform
import time
import tracemalloc
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .config import RESULTS_DIR, SEED, QuantumConfig
from .quantum.feature_maps import encode_states
from .quantum.iqae import iterative_amplitude_estimation, grover_good_probability
from .quantum.similarity import fidelity_matrix
from .reduction import PCAReducer


# --------------------------------------------------------------- metrics
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int | None = None) -> float:
    for i, d in enumerate(ranked[:k] if k else ranked):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k if k else 0.0


# ------------------------------------------------------------- benchmark
DEFAULT_RUNS = [
    {"label": "Classical (cosine, 384-d)", "pipeline": "classical"},
    {"label": "PCA cosine control, 4-d", "pipeline": "pca_cosine", "n_qubits": 4, "encoding": "angle"},
    {"label": "PCA cosine control, 8-d", "pipeline": "pca_cosine", "n_qubits": 8, "encoding": "angle"},
    {"label": "PCA cosine control, 256-d", "pipeline": "pca_cosine", "n_qubits": 8, "encoding": "amplitude"},
    {"label": "Quantum angle, 4q", "pipeline": "quantum", "n_qubits": 4, "encoding": "angle"},
    {"label": "Quantum angle, 8q", "pipeline": "quantum", "n_qubits": 8, "encoding": "angle"},
    {"label": "Quantum IQP, 4q", "pipeline": "quantum", "n_qubits": 4, "encoding": "iqp"},
    {"label": "Quantum IQP, 8q", "pipeline": "quantum", "n_qubits": 8, "encoding": "iqp"},
    {"label": "Quantum amplitude, 4q", "pipeline": "quantum", "n_qubits": 4, "encoding": "amplitude"},
    {"label": "Quantum amplitude, 6q", "pipeline": "quantum", "n_qubits": 6, "encoding": "amplitude"},
    {"label": "Quantum amplitude, 8q", "pipeline": "quantum", "n_qubits": 8, "encoding": "amplitude"},
    {"label": "Hybrid angle, 8q", "pipeline": "hybrid", "n_qubits": 8, "encoding": "angle"},
    {"label": "Hybrid angle 8q, fusion 0.5", "pipeline": "hybrid", "n_qubits": 8, "encoding": "angle",
     "fusion": 0.5},
    {"label": "Hybrid IQP, 8q", "pipeline": "hybrid", "n_qubits": 8, "encoding": "iqp"},
    {"label": "Hybrid amplitude, 8q", "pipeline": "hybrid", "n_qubits": 8, "encoding": "amplitude"},
    {"label": "Hybrid amplitude 8q, SWAP 1024 shots", "pipeline": "hybrid", "n_qubits": 8,
     "encoding": "amplitude", "method": "swap", "shots": 1024},
    {"label": "Hybrid amplitude 8q, IQAE eps 0.01", "pipeline": "hybrid", "n_qubits": 8,
     "encoding": "amplitude", "method": "iqae"},
]


def _cfg_from_run(run: dict, base: QuantumConfig) -> QuantumConfig:
    fields = {k: run[k] for k in ("n_qubits", "encoding", "method", "shots", "layers",
                                  "bandwidth", "execution", "fusion") if k in run}
    return replace(base, **fields).validate()


def evaluate_run(engine, run: dict, queries, k_values=(1, 3, 5, 10), candidates: int = 50,
                 base: QuantumConfig | None = None, q_embs: np.ndarray | None = None,
                 per_query: bool = False) -> dict:
    base = base or QuantumConfig()
    cfg = _cfg_from_run(run, base)
    pipeline = run["pipeline"]
    kmax = max(k_values)
    cand = run.get("candidates", candidates)

    # Warm-up: build the PCA / statevector cache outside the timed loop.
    build_ms = None
    if pipeline in ("quantum", "hybrid", "rbf", "hybrid_rbf"):
        t0 = time.perf_counter()
        space = engine.space(cfg)
        build_ms = (time.perf_counter() - t0) * 1e3
        mem_index = space.nbytes + (engine.index.nbytes if pipeline.startswith("hybrid") else 0)
        explained = space.reducer.explained_variance
    elif pipeline == "pca_cosine":
        r = engine.pca_cosine(cfg.feature_dim)
        mem_index = r.docs.nbytes
        explained = r.reducer.explained_variance
    else:
        mem_index = engine.index.nbytes
        explained = 1.0

    if q_embs is None:
        q_embs = engine.embedder.embed([q.text for q in queries])

    def do(q, emb):
        if pipeline == "bm25":
            return engine.bm25(q.text, kmax)
        return engine.retrieve(pipeline, emb, kmax, cfg, cand)

    do(queries[0], q_embs[0])   # JIT / cache warm-up
    pq = {"ndcg@10": [], "recall@1": [], "recall@5": [], "rr": []}

    sums = {f"recall@{k}": 0.0 for k in k_values}
    sums.update({f"ndcg@{k}": 0.0 for k in k_values})
    mrr, lat, peaks, qstats = 0.0, [], [], {}
    for q, emb in zip(queries, q_embs):
        rel = set(q.relevant)
        tracemalloc.start()
        r = do(q, emb)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        lat.append(r.timings["search"])
        ids = [engine.documents[int(i)].id for i in r.indices]
        for k in k_values:
            sums[f"recall@{k}"] += recall_at_k(ids, rel, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(ids, rel, k)
        mrr += reciprocal_rank(ids, rel, kmax)
        pq["ndcg@10"].append(ndcg_at_k(ids, rel, 10))
        pq["recall@1"].append(recall_at_k(ids, rel, 1))
        pq["recall@5"].append(recall_at_k(ids, rel, 5))
        pq["rr"].append(reciprocal_rank(ids, rel, kmax))
        for key, v in r.stats.items():
            if isinstance(v, (int, float)):
                qstats[key] = qstats.get(key, 0) + v

    n = len(queries)
    lat = np.array(lat)
    out = {
        "label": run.get("label", pipeline), "pipeline": pipeline,
        "config": cfg.to_dict() if pipeline not in ("classical", "bm25") else None,
        "candidates": cand if pipeline.startswith("hybrid") else None,
        "metrics": {**{k: v / n for k, v in sums.items()}, f"mrr@{kmax}": mrr / n},
        "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                       "p95": float(np.percentile(lat, 95))},
        "memory": {"index_bytes": int(mem_index), "peak_query_bytes": int(np.max(peaks)),
                   "mean_query_bytes": int(np.mean(peaks))},
        "explained_variance": explained,
        "build_ms": build_ms,
        "quantum_per_query": {k: v / n for k, v in qstats.items()},
        "queries": n,
    }
    if per_query:
        out["per_query"] = pq
    return out


def kernel_concentration(doc_embeddings: np.ndarray, qubits=(2, 4, 6, 8, 12, 16),
                         encodings=("angle", "iqp"), bandwidths=(2.0, 1.0, 0.5),
                         sample: int = 120, seed: int = SEED) -> list[dict]:
    """Spread of off-diagonal fidelities as qubits grow (exponential concentration).

    Rows with source="corpus" use PCA features of the real documents. Rows with
    source="random" use i.i.d. uniform angles, the setting in which concentration
    theorems apply, as a reference for how quickly an unstructured kernel collapses.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(doc_embeddings), size=min(sample, len(doc_embeddings)), replace=False)
    rows = []

    def stats(states):
        gram = fidelity_matrix(states)
        off = gram[~np.eye(len(gram), dtype=bool)]
        return float(off.mean()), float(off.var())

    for enc in encodings:
        for n in qubits:
            feats = PCAReducer(n, angle_range=True, seed=seed).fit_transform(doc_embeddings)[idx]
            for bw in bandwidths:
                mean, var = stats(encode_states(feats, enc, n, 2, bw))
                rows.append({"source": "corpus", "encoding": enc, "n_qubits": n, "bandwidth": bw,
                             "mean": mean, "variance": var})
    for n in qubits:
        feats = rng.uniform(0, np.pi, size=(len(idx), n))
        for bw in bandwidths:
            mean, var = stats(encode_states(feats, "iqp", n, 2, bw))
            rows.append({"source": "random", "encoding": "iqp", "n_qubits": n, "bandwidth": bw,
                         "mean": mean, "variance": var})
    return rows


def explained_variance_curve(doc_embeddings: np.ndarray, max_dim: int = 256) -> list[dict]:
    dims = [d for d in (2, 4, 6, 8, 10, 16, 32, 64, 128, 256) if d <= min(max_dim, *doc_embeddings.shape)]
    pca = PCAReducer(dims[-1], angle_range=False).fit(doc_embeddings)
    curve = pca.explained_variance_curve
    return [{"dim": d, "explained_variance": curve[d - 1]} for d in dims]


def estimator_efficiency(trials: int = 40, seed: int = SEED) -> dict:
    """Estimation error versus oracle cost: plain sampling against IQAE."""
    rng = np.random.default_rng(seed)
    truths = rng.uniform(0.05, 0.95, size=trials)
    sampling = []
    for shots in (32, 100, 316, 1000, 3162, 10000, 31623):
        err = [abs(rng.binomial(shots, a) / shots - a) for a in truths]
        sampling.append({"oracle_calls": shots, "mean_abs_error": float(np.mean(err))})
    iqae = []
    for eps in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        errs, calls = [], []
        for a in truths:
            r = iterative_amplitude_estimation(lambda k, a=a: grover_good_probability(a, k),
                                               epsilon=eps, alpha=0.05, shots=100, rng=rng)
            errs.append(abs(r.estimation - a))
            calls.append(r.a_calls)
        iqae.append({"epsilon": eps, "oracle_calls": float(np.mean(calls)),
                     "mean_abs_error": float(np.mean(errs))})
    return {"sampling": sampling, "iqae": iqae, "trials": trials}


def run_benchmark(engine, runs: list[dict] | None = None, n_queries: int = 300,
                  candidates: int = 50, k_values=(1, 3, 5, 10), seed: int = SEED,
                  progress: Callable[[int, int, str], None] | None = None,
                  include_analysis: bool = True, out_path: Path | None = None) -> dict:
    runs = runs or DEFAULT_RUNS
    rng = np.random.default_rng(seed)
    qs = list(engine.queries)
    sel = rng.choice(len(qs), size=min(n_queries, len(qs)), replace=False)
    queries = [qs[i] for i in sorted(sel)]
    total = len(runs) + (3 if include_analysis else 0)
    results = []
    t_start = time.perf_counter()
    for i, run in enumerate(runs):
        if progress:
            progress(i, total, run.get("label", run["pipeline"]))
        results.append(evaluate_run(engine, run, queries, k_values, candidates))
    analysis = {}
    if include_analysis:
        if progress:
            progress(len(runs), total, "PCA explained variance")
        analysis["explained_variance"] = explained_variance_curve(engine.doc_embeddings)
        if progress:
            progress(len(runs) + 1, total, "Kernel concentration")
        analysis["kernel_concentration"] = kernel_concentration(engine.doc_embeddings)
        if progress:
            progress(len(runs) + 2, total, "Estimator efficiency")
        analysis["estimator_efficiency"] = estimator_efficiency()
    if progress:
        progress(total, total, "done")
    import pennylane as qml
    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": time.perf_counter() - t_start,
        "dataset": engine.meta.get("name"), "documents": len(engine.documents),
        "queries_evaluated": len(queries), "seed": seed, "candidates": candidates,
        "k_values": list(k_values),
        "environment": {"embedder": engine.embedder.description, "index": engine.index.backend,
                        "pennylane": qml.__version__, "device": "default.qubit (simulator)",
                        "python": platform.python_version(), "machine": platform.machine(),
                        "processor": platform.processor() or platform.machine()},
        "runs": results, "analysis": analysis,
    }
    out_path = out_path or RESULTS_DIR / "benchmark_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_path.parent / f"benchmark_{stamp}.json").write_text(json.dumps(report, indent=2))
    return report
