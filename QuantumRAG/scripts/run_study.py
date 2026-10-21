"""Multi-seed benchmark with confidence intervals and paired significance tests.

For each seed the script draws a fresh 600-passage corpus from the SQuAD v1.1
dev set (stratified over articles), samples 300 evaluation questions, and runs
every configuration with that seed (shot noise included). It then reports, per
configuration, the mean over seeds, the standard deviation across seeds, a
seed-stratified bootstrap 95% CI over all evaluated questions, and paired tests
for the comparisons that answer RQ1-RQ3 (Holm-corrected).

    python scripts/run_study.py --squad path/to/dev-v1.1.json --seeds 42 7 13 21 99
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_dataset import build  # noqa: E402
from quantumrag.config import QuantumConfig  # noqa: E402
from quantumrag.embedding import Embedder  # noqa: E402
from quantumrag.engine import QuantumRAG  # noqa: E402
from quantumrag.evaluation import (DEFAULT_RUNS, estimator_efficiency, evaluate_run,  # noqa: E402
                                   explained_variance_curve, kernel_concentration)
from quantumrag.stats import bootstrap_ci, holm, paired_permutation_test, wilcoxon_p  # noqa: E402

EXTRA_RUNS = [
    {"label": "BM25 (lexical)", "pipeline": "bm25"},
    {"label": "Classical RBF kernel, 8-d", "pipeline": "rbf", "n_qubits": 8, "encoding": "angle"},
    {"label": "Hybrid classical RBF, 8-d", "pipeline": "hybrid_rbf", "n_qubits": 8, "encoding": "angle"},
]

# (label A, label B, question) - paired tests on per-question NDCG@10
COMPARISONS = [
    ("Quantum angle, 8q", "PCA cosine control, 8-d", "RQ2: angle kernel vs. cosine on identical 8-d features"),
    ("Quantum IQP, 8q", "PCA cosine control, 8-d", "RQ2: IQP kernel vs. cosine on identical 8-d features"),
    ("Quantum angle, 8q", "Classical RBF kernel, 8-d", "RQ2: angle kernel vs. classical RBF kernel"),
    ("Quantum IQP, 8q", "Classical RBF kernel, 8-d", "RQ2: IQP kernel vs. classical RBF kernel"),
    ("PCA cosine control, 8-d", "Classical (cosine, 384-d)", "RQ2: cost of compression to 8-d"),
    ("Quantum amplitude, 8q", "PCA cosine control, 256-d", "RQ2: amplitude fidelity vs. 256-d cosine (expected equal)"),
    ("Quantum amplitude, 8q", "Classical (cosine, 384-d)", "RQ1: best quantum vs. classical"),
    ("Hybrid angle, 8q", "Classical (cosine, 384-d)", "RQ3: hybrid angle vs. classical"),
    ("Hybrid angle, 8q", "Hybrid classical RBF, 8-d", "RQ3: quantum vs. RBF re-ranking"),
    ("Hybrid angle 8q, fusion 0.5", "Classical (cosine, 384-d)", "RQ3: fused hybrid vs. classical"),
    ("Hybrid amplitude 8q, SWAP 1024 shots", "Hybrid amplitude, 8q", "Estimators: SWAP 1024 shots vs. exact"),
    ("Hybrid amplitude 8q, IQAE eps 0.01", "Hybrid amplitude, 8q", "Estimators: IQAE vs. exact"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 21, 99])
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "study_latest.json")
    args = ap.parse_args()

    runs = DEFAULT_RUNS + EXTRA_RUNS
    embedder = Embedder()
    if embedder.backend != "minilm":
        print("WARNING: MiniLM not available, results will use the LSA fallback")
    print(f"Embedder: {embedder.description} [{embedder.runtime}]")
    t_start = time.perf_counter()
    per_seed: dict[int, dict] = {}
    analysis = {}
    for seed in args.seeds:
        ds = build(args.squad, args.docs, seed)
        dpath = ROOT / "results" / "study_data" / f"squad_seed{seed}.json"
        dpath.parent.mkdir(parents=True, exist_ok=True)
        dpath.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")
        engine = QuantumRAG(dataset=dpath, embedder=embedder)
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(len(engine.queries), size=min(args.queries, len(engine.queries)),
                                replace=False))
        queries = [engine.queries[i] for i in sel]
        q_embs = embedder.embed([q.text for q in queries])
        base = replace(QuantumConfig(), seed=seed)
        res = {}
        for run in runs:
            r = evaluate_run(engine, run, queries, candidates=args.candidates, base=base,
                             q_embs=q_embs, per_query=True)
            res[run["label"]] = r
            print(f"  seed {seed:3d}  {run['label'][:44]:44s} NDCG@10 {r['metrics']['ndcg@10']:.3f}"
                  f"  {r['latency_ms']['mean']:7.2f} ms", flush=True)
        per_seed[seed] = {"documents": len(engine.documents), "questions_available": len(engine.queries),
                          "runs": res}
        if seed == args.seeds[0]:
            analysis["explained_variance"] = explained_variance_curve(engine.doc_embeddings)
            analysis["kernel_concentration"] = kernel_concentration(engine.doc_embeddings)
            analysis["estimator_efficiency"] = estimator_efficiency()

    # ---------------------------------------------------------- aggregate
    seeds = args.seeds
    summary = []
    for run in runs:
        lab = run["label"]
        row = {"label": lab, "pipeline": run["pipeline"]}
        for metric, key in (("ndcg@10", "ndcg@10"), ("recall@1", "recall@1"),
                            ("recall@5", "recall@5"), ("mrr@10", "rr")):
            per = [np.mean(per_seed[s]["runs"][lab]["per_query"][key]) for s in seeds]
            pooled = np.concatenate([per_seed[s]["runs"][lab]["per_query"][key] for s in seeds])
            groups = np.concatenate([[i] * len(per_seed[s]["runs"][lab]["per_query"][key])
                                     for i, s in enumerate(seeds)])
            lo, hi = bootstrap_ci(pooled, groups, seed=1)
            row[metric] = {"mean": float(np.mean(pooled)), "seed_sd": float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
                           "ci95": [lo, hi], "per_seed": [float(v) for v in per]}
        lat = [per_seed[s]["runs"][lab]["latency_ms"]["mean"] for s in seeds]
        p95 = [per_seed[s]["runs"][lab]["latency_ms"]["p95"] for s in seeds]
        row["latency_ms"] = {"mean": float(np.mean(lat)), "p95": float(np.mean(p95))}
        row["memory_mb"] = per_seed[seeds[0]]["runs"][lab]["memory"]["index_bytes"] / 1e6
        row["explained_variance"] = float(np.mean([per_seed[s]["runs"][lab]["explained_variance"] for s in seeds]))
        summary.append(row)

    tests = []
    for a, b, q in COMPARISONS:
        da = np.concatenate([per_seed[s]["runs"][a]["per_query"]["ndcg@10"] for s in seeds])
        db = np.concatenate([per_seed[s]["runs"][b]["per_query"]["ndcg@10"] for s in seeds])
        groups = np.concatenate([[i] * len(per_seed[s]["runs"][a]["per_query"]["ndcg@10"])
                                 for i, s in enumerate(seeds)])
        diff = da - db
        lo, hi = bootstrap_ci(diff, groups, seed=2)
        per_seed_diff = [float(np.mean(per_seed[s]["runs"][a]["per_query"]["ndcg@10"])
                               - np.mean(per_seed[s]["runs"][b]["per_query"]["ndcg@10"])) for s in seeds]
        tests.append({"a": a, "b": b, "question": q, "mean_diff": float(diff.mean()), "ci95": [lo, hi],
                      "p_permutation": paired_permutation_test(da, db, seed=3),
                      "p_wilcoxon": wilcoxon_p(da, db), "n": int(len(diff)),
                      "identical_rankings_share": float(np.mean(np.isclose(diff, 0))),
                      "per_seed_diff": per_seed_diff})
    adj = holm([t["p_permutation"] for t in tests])
    for t, p in zip(tests, adj):
        t["p_holm"] = p

    import pennylane as qml
    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": time.perf_counter() - t_start,
        "design": {"seeds": seeds, "documents_per_seed": args.docs, "questions_per_seed": args.queries,
                   "candidates": args.candidates, "bootstrap": "5000 resamples, stratified by seed",
                   "test": "paired sign-flip permutation (20000), Holm-corrected; Wilcoxon also reported"},
        "environment": {"embedder": embedder.description, "embedder_runtime": embedder.runtime,
                        "pennylane": qml.__version__, "device": "default.qubit (simulator)",
                        "python": platform.python_version(), "machine": platform.machine()},
        "summary": summary, "tests": tests, "analysis": analysis,
        "per_seed": {str(s): {"documents": v["documents"], "questions_available": v["questions_available"],
                              "runs": {k: {kk: vv for kk, vv in r.items() if kk != "per_query"}
                                       for k, r in v["runs"].items()}}
                     for s, v in per_seed.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    pq = {str(s): {k: r["per_query"] for k, r in v["runs"].items()} for s, v in per_seed.items()}
    (args.out.parent / "study_per_query.json").write_text(json.dumps(pq))

    print(f"\n{'Configuration':44s} {'NDCG@10':>8s} {'95% CI':>17s} {'seed sd':>8s}")
    for r in summary:
        m = r["ndcg@10"]
        print(f"{r['label'][:44]:44s} {m['mean']:8.3f}  [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] {m['seed_sd']:8.3f}")
    print("\nPaired tests (NDCG@10, A - B):")
    for t in tests:
        print(f"  {t['question'][:58]:58s} {t['mean_diff']:+.3f} [{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] "
              f"p_holm={t['p_holm']:.2g}")
    print(f"\nSaved {args.out} ({report['duration_s']:.0f} s)")


if __name__ == "__main__":
    main()
