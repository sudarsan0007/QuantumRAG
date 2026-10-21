"""Sensitivity to kernel bandwidth and a clean, single-process latency measurement.

1. Bandwidth sweep: pure quantum angle and IQP retrieval on 8 qubits for
   b in {0.25, 0.5, 1, 2}, and the classical RBF control with gamma scaled by the
   matching factors, over the same seeds and questions as run_study.py. This is a
   sensitivity analysis on the evaluation questions (not a tuned result), so it
   shows the best case each kernel family could reach.
2. Latency: the Table III configurations on the first seed, measured with nothing
   else running.

    python scripts/run_extras.py --squad path/to/dev-v1.1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_dataset import build  # noqa: E402
from quantumrag.config import QuantumConfig  # noqa: E402
from quantumrag.embedding import Embedder  # noqa: E402
from quantumrag.engine import QuantumRAG  # noqa: E402
from quantumrag.evaluation import evaluate_run, ndcg_at_k  # noqa: E402
from quantumrag.retrievers import RBFKernelRetriever, _top  # noqa: E402
from quantumrag.stats import bootstrap_ci  # noqa: E402

BANDWIDTHS = [0.25, 0.5, 1.0, 2.0]
GAMMA_FACTORS = [0.0625, 0.25, 1.0, 4.0]   # b scales distances, so gamma scales as b^2

LATENCY_RUNS = [
    {"label": "Classical (cosine, 384-d)", "pipeline": "classical"},
    {"label": "BM25 (lexical)", "pipeline": "bm25"},
    {"label": "Quantum angle, 8q", "pipeline": "quantum", "n_qubits": 8, "encoding": "angle"},
    {"label": "Quantum IQP, 8q", "pipeline": "quantum", "n_qubits": 8, "encoding": "iqp"},
    {"label": "Quantum amplitude, 8q", "pipeline": "quantum", "n_qubits": 8, "encoding": "amplitude"},
    {"label": "Hybrid angle, 8q", "pipeline": "hybrid", "n_qubits": 8, "encoding": "angle"},
    {"label": "Hybrid amplitude, 8q", "pipeline": "hybrid", "n_qubits": 8, "encoding": "amplitude"},
    {"label": "Hybrid amplitude 8q, SWAP 1024 shots", "pipeline": "hybrid", "n_qubits": 8,
     "encoding": "amplitude", "method": "swap", "shots": 1024},
    {"label": "Hybrid amplitude 8q, IQAE eps 0.01", "pipeline": "hybrid", "n_qubits": 8,
     "encoding": "amplitude", "method": "iqae"},
    {"label": "Classical RBF kernel, 8-d", "pipeline": "rbf", "n_qubits": 8, "encoding": "angle"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 21, 99])
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "extras_latest.json")
    args = ap.parse_args()
    emb = Embedder()
    print(f"Embedder: {emb.description} [{emb.runtime}]")
    sweep: dict[str, dict[str, list]] = {}
    latency = []
    for si, seed in enumerate(args.seeds):
        dpath = ROOT / "results" / "study_data" / f"squad_seed{seed}.json"
        if not dpath.exists():
            dpath.parent.mkdir(parents=True, exist_ok=True)
            dpath.write_text(json.dumps(build(args.squad, 600, seed), ensure_ascii=False), encoding="utf-8")
        eng = QuantumRAG(dataset=dpath, embedder=emb)
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(len(eng.queries), size=min(args.queries, len(eng.queries)), replace=False))
        queries = [eng.queries[i] for i in sel]
        q_embs = emb.embed([q.text for q in queries])
        base = replace(QuantumConfig(), seed=seed)
        for enc in ("angle", "iqp"):
            for b in BANDWIDTHS:
                run = {"label": f"{enc} b={b}", "pipeline": "quantum", "n_qubits": 8, "encoding": enc,
                       "bandwidth": b}
                r = evaluate_run(eng, run, queries, base=base, q_embs=q_embs, per_query=True)
                sweep.setdefault(run["label"], {}).setdefault("per_query", []).extend(r["per_query"]["ndcg@10"])
                sweep[run["label"]].setdefault("per_seed", []).append(r["metrics"]["ndcg@10"])
                print(f"  seed {seed} {run['label']:16s} {r['metrics']['ndcg@10']:.3f}", flush=True)
        space = eng.space(replace(base, n_qubits=8, encoding="angle").validate())
        for f in GAMMA_FACTORS:
            ret = RBFKernelRetriever(space)
            ret.gamma *= f
            vals = []
            for q, e in zip(queries, q_embs):
                feat = space.reducer.transform(e)
                s = ret.scores(feat, space.doc_features)
                ids = [eng.documents[int(i)].id for i in _top(s, 10)]
                vals.append(ndcg_at_k(ids, set(q.relevant), 10))
            lab = f"rbf gamma x{f:g}"
            sweep.setdefault(lab, {}).setdefault("per_query", []).extend(vals)
            sweep[lab].setdefault("per_seed", []).append(float(np.mean(vals)))
            print(f"  seed {seed} {lab:16s} {np.mean(vals):.3f}", flush=True)
        if si == 0:
            for run in LATENCY_RUNS:
                r = evaluate_run(eng, run, queries, base=base, q_embs=q_embs)
                latency.append({"label": run["label"], **r["latency_ms"],
                                "memory_mb": r["memory"]["index_bytes"] / 1e6})
                print(f"  latency {run['label']:40s} {r['latency_ms']['mean']:.2f} ms", flush=True)
    out = {"bandwidths": BANDWIDTHS, "gamma_factors": GAMMA_FACTORS, "sweep": {}, "latency": latency,
           "seeds": args.seeds, "latency_seed": args.seeds[0], "embedder": emb.description}
    n_per = len(next(iter(sweep.values()))["per_query"]) // len(args.seeds)
    groups = np.repeat(np.arange(len(args.seeds)), n_per)
    for lab, v in sweep.items():
        x = np.array(v["per_query"])
        lo, hi = bootstrap_ci(x, groups[: len(x)], seed=1)
        out["sweep"][lab] = {"mean": float(x.mean()), "ci95": [lo, hi],
                             "seed_sd": float(np.std(v["per_seed"], ddof=1)), "per_seed": v["per_seed"]}
    args.out.write_text(json.dumps(out, indent=2))
    for lab, v in out["sweep"].items():
        print(f"{lab:18s} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print("saved", args.out)


if __name__ == "__main__":
    t = time.perf_counter()
    main()
    print(f"{time.perf_counter() - t:.0f} s")
