"""Iterative Quantum Amplitude Estimation (Grinko, Gacon, Zoufal, Woerner, 2021).

Estimates the amplitude a = P(good) of a state-preparation circuit A without
quantum phase estimation. Each round runs Q^k A, where Q is the Grover operator,
and measures the good-state probability sin^2((2k+1) theta) with a = sin^2(theta).
The confidence interval for theta shrinks geometrically by choosing larger k,
giving O(1/epsilon) oracle queries instead of the O(1/epsilon^2) shots needed by
plain sampling.

Here A = U(y)^dagger U(x) and the good state is |0...0>, so a is exactly the
fidelity kernel K(x, y) = |<psi(y)|psi(x)>|^2.

The estimator only needs a callable ``prob_good(k)`` that returns the exact
probability of the good state after Q^k A. Measurement shots are drawn from
that distribution with a seeded generator, which is how a shot-based simulator
produces counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class IQAEResult:
    estimation: float
    confidence_interval: tuple[float, float]
    rounds: int
    oracle_queries: int          # total applications of Q across all shots
    total_shots: int
    a_calls: int = 0             # total applications of A (each shot of Q^k A costs 2k+1)
    powers: list[int] = field(default_factory=list)


def _find_next_k(k: int, upper_half_circle: bool, theta_interval: tuple[float, float],
                 min_ratio: float = 2.0) -> tuple[int, bool]:
    theta_l, theta_u = theta_interval
    old_scaling = 4 * k + 2
    width = theta_u - theta_l
    if width <= 0:
        return k, upper_half_circle
    max_scaling = int(1 / (2 * width))
    scaling = max_scaling - (max_scaling - 2) % 4
    while scaling >= min_ratio * old_scaling:
        theta_min = scaling * theta_l - int(scaling * theta_l)
        theta_max = scaling * theta_u - int(scaling * theta_u)
        if theta_min <= theta_max <= 0.5 and theta_min <= 0.5:
            return int((scaling - 2) / 4), True
        if theta_max >= 0.5 and theta_max >= theta_min >= 0.5:
            return int((scaling - 2) / 4), False
        scaling -= 4
    return int(k), upper_half_circle


def iterative_amplitude_estimation(prob_good: Callable[[int], float], epsilon: float = 0.01,
                                   alpha: float = 0.05, shots: int = 100,
                                   rng: np.random.Generator | None = None,
                                   max_rounds: int = 64) -> IQAEResult:
    """Run IQAE with Chernoff-Hoeffding confidence intervals.

    epsilon: target half-width of the confidence interval on the amplitude a.
    alpha:   overall failure probability.
    """
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rng = rng or np.random.default_rng()

    # Theta is expressed in units of 2*pi: a = sin^2(2*pi*theta), theta in [0, 1/4].
    theta_l, theta_u = 0.0, 0.25
    k, upper = 0, True
    powers: list[int] = []
    max_iters = int(np.ceil(np.log2(np.pi / (8 * epsilon)))) + 1
    ones_for_k, shots_for_k = 0, 0
    queries, total_shots, rounds, a_calls = 0, 0, 0, 0

    while theta_u - theta_l > epsilon / np.pi and rounds < max_rounds:
        rounds += 1
        new_k, upper = _find_next_k(k, upper, (theta_l, theta_u))
        if new_k != k or not powers:
            ones_for_k, shots_for_k = 0, 0
        k = new_k
        powers.append(k)

        p = float(np.clip(prob_good(k), 0.0, 1.0))
        ones = int(rng.binomial(shots, p))
        ones_for_k += ones
        shots_for_k += shots
        queries += shots * k
        a_calls += shots * (2 * k + 1)
        total_shots += shots

        prob = ones_for_k / shots_for_k
        eps_a = np.sqrt(np.log(2 * max_iters / alpha) / (2 * shots_for_k))
        a_min, a_max = max(0.0, prob - eps_a), min(1.0, prob + eps_a)

        if upper:
            th_min = np.arccos(1 - 2 * a_min) / (2 * np.pi)
            th_max = np.arccos(1 - 2 * a_max) / (2 * np.pi)
        else:
            th_min = 1 - np.arccos(1 - 2 * a_max) / (2 * np.pi)
            th_max = 1 - np.arccos(1 - 2 * a_min) / (2 * np.pi)

        scaling = 4 * k + 2
        new_u = (int(scaling * theta_u) + th_max) / scaling
        new_l = (int(scaling * theta_l) + th_min) / scaling
        theta_l, theta_u = max(theta_l, new_l), min(theta_u, new_u)
        if theta_l > theta_u:  # numerical guard
            theta_l, theta_u = theta_u, theta_l

    a_l = float(np.sin(2 * np.pi * theta_l) ** 2)
    a_u = float(np.sin(2 * np.pi * theta_u) ** 2)
    return IQAEResult(estimation=(a_l + a_u) / 2, confidence_interval=(a_l, a_u),
                      rounds=rounds, oracle_queries=queries, total_shots=total_shots,
                      a_calls=a_calls, powers=powers)


def grover_good_probability(amplitude: float, k: int) -> float:
    """P(good) after Q^k A for a state with initial good-probability ``amplitude``."""
    theta = np.arcsin(np.sqrt(np.clip(amplitude, 0.0, 1.0)))
    return float(np.sin((2 * k + 1) * theta) ** 2)
