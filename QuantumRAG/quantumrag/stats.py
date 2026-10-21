"""Statistics for the multi-seed study: bootstrap intervals and paired tests."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling queries within each seed group."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if groups is None:
        groups = np.zeros(len(x), dtype=int)
    parts = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(p, size=len(p), replace=True) for p in parts])
        means[b] = x[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
                            seed: int = 0) -> float:
    """Two-sided paired randomisation (sign-flip) test of mean(a - b) = 0."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    obs = abs(d.mean())
    if np.allclose(d, 0):
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-15)) / (n_perm + 1))


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return [float(v) for v in adj]
