"""Run the full, seeded benchmark suite from the command line.

    python scripts/run_benchmark.py --queries 300

Writes results/benchmark_latest.json (shown on the dashboard's Benchmarks page)
plus a timestamped copy, and prints a summary table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantumrag.engine import QuantumRAG  # noqa: E402
from quantumrag.evaluation import run_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--no-analysis", action="store_true")
    args = ap.parse_args()

    engine = QuantumRAG()
    print(f"Embedder: {engine.embedder.description}")
    print(f"Index:    {engine.index.backend}")
    print(f"Corpus:   {len(engine.documents)} documents, {len(engine.queries)} labelled queries\n")

    def progress(i, total, label):
        print(f"  [{i + 1 if i < total else total}/{total}] {label}", flush=True)

    rep = run_benchmark(engine, n_queries=args.queries, candidates=args.candidates,
                        include_analysis=not args.no_analysis, progress=progress)
    print(f"\n{'Configuration':42s} {'R@1':>6s} {'R@5':>6s} {'NDCG@10':>8s} {'MRR':>6s} {'ms/query':>9s}")
    for r in rep["runs"]:
        m = r["metrics"]
        print(f"{r['label'][:42]:42s} {m['recall@1']:6.3f} {m['recall@5']:6.3f} {m['ndcg@10']:8.3f} "
              f"{m['mrr@10']:6.3f} {r['latency_ms']['mean']:9.2f}")
    print(f"\nSaved to results/benchmark_latest.json ({rep['duration_s']:.0f} s)")


if __name__ == "__main__":
    main()
