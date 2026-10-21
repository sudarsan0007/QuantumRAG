"""Paper figures (numbered as in the paper; Fig. 1 is the architecture diagram) from results/study_latest.json and results/extras_latest.json.

    python scripts/make_figures.py --out paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MID, LIGHT = "#000000", "#555555", "#999999"


def fig_variance(an, out):
    ev = an["explained_variance"]
    d = [e["dim"] for e in ev]
    v = [e["explained_variance"] for e in ev]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(d, v, "-o", color=INK, ms=4, lw=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(d, [str(x) for x in d])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Variance retained")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    val = dict(zip(d, v))
    for dim, txt, xy, ha in ((8, "8 components\n(angle/IQP, 8 qubits)", (2.0, 0.78), "left"),
                             (256, "256 components\n(amplitude, 8 qubits)", (240, 0.10), "right")):
        ax.axvline(dim, color=LIGHT, ls="--", lw=0.8)
        ax.annotate(f"{txt}: {val[dim]:.1%}", (dim, val[dim]), xytext=xy, ha=ha, fontsize=9.5,
                    arrowprops={"arrowstyle": "-", "color": MID, "lw": 0.7})
    fig.savefig(out / "fig3.png")
    plt.close(fig)


def fig_concentration(an, out):
    rows = an["kernel_concentration"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    styles = [("corpus", "angle", 1.0, "-o", INK, "Angle, corpus, b = 1"),
              ("corpus", "iqp", 1.0, "-s", MID, "IQP, corpus, b = 1"),
              ("corpus", "angle", 0.5, ":o", INK, "Angle, corpus, b = 0.5"),
              ("random", "iqp", 1.0, "--^", LIGHT, "IQP, random inputs, b = 1")]
    for src, enc, bw, st, col, lab in styles:
        pts = sorted((r["n_qubits"], r["variance"]) for r in rows
                     if r["source"] == src and r["encoding"] == enc and r["bandwidth"] == bw)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], st, color=col, ms=4, lw=1.4, label=lab,
                mfc="white" if ":" in st else col)
    ax.set_yscale("log")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Variance of off-diagonal fidelity")
    ax.set_xticks([2, 4, 6, 8, 12, 16])
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    fig.savefig(out / "fig6.png")
    plt.close(fig)


def fig_estimators(an, out):
    e = an["estimator_efficiency"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    s = e["sampling"]
    ax.loglog([r["oracle_calls"] for r in s], [r["mean_abs_error"] for r in s], "-o", color=MID, ms=4,
              lw=1.4, label="Direct sampling")
    q = e["iqae"]
    ax.loglog([r["oracle_calls"] for r in q], [r["mean_abs_error"] for r in q], "-s", color=INK, ms=4,
              lw=1.6, label="IQAE")
    ax.set_xlabel("Circuit calls (oracle applications)")
    ax.set_ylabel("Mean absolute error")
    ax.legend(frameon=False)
    ax.grid(color="#e5e5e5", lw=0.5, which="major")
    fig.savefig(out / "fig5.png")
    plt.close(fig)


def fig_ndcg(study, out):
    keep = [("Classical (cosine, 384-d)", "Classical cosine, 384-d"), ("BM25 (lexical)", "BM25"),
            ("PCA cosine control, 256-d", "PCA cosine, 256-d"), ("Quantum amplitude, 8q", "Quantum amplitude, 8q"),
            ("Hybrid angle 8q, fusion 0.5", "Hybrid angle 8q + fusion"),
            ("Hybrid classical RBF, 8-d", "Hybrid RBF, 8-d"), ("Hybrid angle, 8q", "Hybrid angle, 8q"),
            ("PCA cosine control, 8-d", "PCA cosine, 8-d"), ("Classical RBF kernel, 8-d", "RBF kernel, 8-d"),
            ("Quantum angle, 8q", "Quantum angle, 8q"), ("Quantum IQP, 8q", "Quantum IQP, 8q")]
    s = {r["label"]: r for r in study["summary"]}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (lab, name) in enumerate(keep):
        m = s[lab]["ndcg@10"]
        quantum = lab.startswith(("Quantum", "Hybrid angle", "Hybrid IQP", "Hybrid amp"))
        y = len(keep) - 1 - i
        ax.errorbar(m["mean"], y, xerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]],
                    fmt="o" if quantum else "s", color=INK if quantum else MID,
                    mfc=INK if quantum else "white", ms=5, capsize=2.5, lw=1.2)
    ax.set_yticks(range(len(keep)), [n for _, n in keep][::-1], fontsize=9)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("NDCG@10 (mean and 95% CI, 5 seeds x 300 questions)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    fig.savefig(out / "fig2.png")
    plt.close(fig)


def fig_bandwidth(extras, out, study=None):
    sw = extras["sweep"]
    bws = extras["bandwidths"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for enc, st, col, name in (("angle", "-o", INK, "Quantum angle, 8q"), ("iqp", "-s", MID, "Quantum IQP, 8q")):
        m = [sw[f"{enc} b={b}"]["mean"] for b in bws]
        lo = [sw[f"{enc} b={b}"]["ci95"][0] for b in bws]
        hi = [sw[f"{enc} b={b}"]["ci95"][1] for b in bws]
        ax.plot(bws, m, st, color=col, ms=4, lw=1.4, label=name)
        ax.fill_between(bws, lo, hi, color=col, alpha=0.12, lw=0)
    g = [sw[f"rbf gamma x{f:g}"]["mean"] for f in extras["gamma_factors"]]
    ax.plot(bws, g, "--^", color=LIGHT, ms=4, lw=1.4, label="Classical RBF, 8-d (any gamma)")
    if study is not None:
        pc = {r["label"]: r for r in study["summary"]}["PCA cosine control, 8-d"]["ndcg@10"]["mean"]
        ax.axhline(pc, color=MID, ls=":", lw=1.4, label="PCA cosine, 8-d")
        ax.set_ylim(top=pc + 0.02)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bws, [f"{b:g}" for b in bws])
    ax.set_xlabel("Feature-map bandwidth b")
    ax.set_ylabel("NDCG@10")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    fig.savefig(out / "fig4.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    study = json.loads((ROOT / "results" / "study_latest.json").read_text())
    fig_variance(study["analysis"], args.out)
    fig_concentration(study["analysis"], args.out)
    fig_estimators(study["analysis"], args.out)
    fig_ndcg(study, args.out)
    ex = ROOT / "results" / "extras_latest.json"
    if ex.exists():
        fig_bandwidth(json.loads(ex.read_text()), args.out, study)
    print("figures in", args.out)


if __name__ == "__main__":
    main()
