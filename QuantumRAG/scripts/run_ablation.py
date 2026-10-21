"""Ablations: number of qubits (2-12) and number of feature-map layers (1-3).

For each seed and question set of run_study.py, evaluates pure quantum angle and
IQP retrieval and the PCA-cosine control at the same feature dimension.

    python scripts/run_ablation.py --squad path/to/dev-v1.1.json
"""
from __future__ import annotations

import argparse
import json
import sys
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
from quantumrag.evaluation import evaluate_run  # noqa: E402
from quantumrag.stats import bootstrap_ci  # noqa: E402

QUBITS = [2, 4, 6, 8, 10, 12]
LAYERS = [1, 2, 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 21, 99])
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablation_latest.json")
    args = ap.parse_args()
    emb = Embedder()
    print(f"Embedder: {emb.description} [{emb.runtime}]", flush=True)
    acc: dict[str, dict] = {}
    for seed in args.seeds:
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
        runs = []
        for n in QUBITS:
            runs.append({"label": f"pca {n}", "pipeline": "pca_cosine", "n_qubits": n, "encoding": "angle"})
            for enc in ("angle", "iqp"):
                runs.append({"label": f"{enc} {n}q L2", "pipeline": "quantum", "n_qubits": n, "encoding": enc})
        for L in LAYERS:
            if L == 2:
                continue
            for enc in ("angle", "iqp"):
                runs.append({"label": f"{enc} 8q L{L}", "pipeline": "quantum", "n_qubits": 8, "encoding": enc, "layers": L})
        for run in runs:
            r = evaluate_run(eng, run, queries, base=base, q_embs=q_embs, per_query=True)
            a = acc.setdefault(run["label"], {"per_query": [], "per_seed": [], "latency": []})
            a["per_query"].extend(r["per_query"]["ndcg@10"])
            a["per_seed"].append(r["metrics"]["ndcg@10"])
            a["latency"].append(r["latency_ms"]["mean"])
            print(f"  seed {seed} {run['label']:12s} {r['metrics']['ndcg@10']:.3f}", flush=True)
    groups = np.repeat(np.arange(len(args.seeds)), args.queries)
    out = {"qubits": QUBITS, "layers": LAYERS, "seeds": args.seeds, "embedder": emb.description, "results": {}}
    for lab, a in acc.items():
        x = np.array(a["per_query"])
        lo, hi = bootstrap_ci(x, groups[: len(x)], seed=1)
        out["results"][lab] = {"mean": float(x.mean()), "ci95": [lo, hi], "per_seed": a["per_seed"],
                               "seed_sd": float(np.std(a["per_seed"], ddof=1)), "latency_ms": float(np.mean(a["latency"]))}
    args.out.write_text(json.dumps(out, indent=2))
    for lab, v in out["results"].items():
        print(f"{lab:14s} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  {v['latency_ms']:.1f} ms")


if __name__ == "__main__":
    main()
