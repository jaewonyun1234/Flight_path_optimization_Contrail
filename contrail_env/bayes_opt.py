"""
bayes_opt.py — Minimal Gaussian-process Bayesian optimizer (RBF + EI).

A dependency-free stand-in for skopt.gp_minimize at the scale this project
needs (<= 4 dimensions, <= ~50 evaluations), used to tune the Pasqal analog
pulse schedule (T, Omega_max, delta_init, delta_final) following Tibaldi
et al. (arXiv:2501.16229).

Method, in one paragraph: evaluations are normalized into the unit cube and
standardized in y; a Gaussian process with an RBF kernel is fit by Cholesky
factorization; the next point maximizes Expected Improvement over a cloud of
random candidates plus local perturbations of the incumbent. Deterministic
given the rng.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# Kernel hyperparameters: fixed, not fit. With <= 50 points in a unit cube,
# marginal-likelihood optimization adds noise for no measurable gain.
_LENGTH_SCALE = 0.25
_NOISE = 1e-8
_N_RANDOM_CANDIDATES = 256
_N_LOCAL_CANDIDATES = 64
_LOCAL_SIGMA = 0.08


@dataclass
class BOResult:
    """Outcome of gp_minimize: incumbent plus the full evaluation trace."""

    x_best: np.ndarray
    y_best: float
    x_iters: list[np.ndarray]
    y_iters: list[float]


def _rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """k(a, b) = exp(-||a - b||^2 / (2 l^2)) for row-stacked points."""
    sq = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.exp(-0.5 * np.maximum(sq, 0.0) / _LENGTH_SCALE**2)


def _gp_posterior(
    x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """GP posterior mean and standard deviation at the query points."""
    k_tt = _rbf_kernel(x_train, x_train) + _NOISE * np.eye(len(x_train))
    k_tq = _rbf_kernel(x_train, x_query)
    chol = np.linalg.cholesky(k_tt)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    mean = k_tq.T @ alpha
    v = np.linalg.solve(chol, k_tq)
    var = np.maximum(1.0 - np.sum(v**2, axis=0), 1e-12)
    return mean, np.sqrt(var)


def _expected_improvement(
    mean: np.ndarray, std: np.ndarray, y_best: float
) -> np.ndarray:
    """EI for minimization: E[max(y_best - Y, 0)]."""
    z = (y_best - mean) / std
    cdf = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return (y_best - mean) * cdf + std * pdf


def gp_minimize(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    *,
    n_calls: int = 20,
    n_initial: int = 6,
    rng: np.random.Generator | None = None,
    on_step: Callable[[int, np.ndarray, float], None] | None = None,
) -> BOResult:
    """Minimize `func` over a box via GP-EI Bayesian optimization.

    Args:
        func:      objective; receives a point in the ORIGINAL units.
        bounds:    [(lo, hi), ...] per dimension.
        n_calls:   total evaluations (initial random + BO-guided).
        n_initial: random warm-up evaluations before the GP takes over.
        rng:       numpy Generator for full determinism.
        on_step:   optional callback(iteration, x, y) after each evaluation.
    """
    rng = rng or np.random.default_rng()
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    n_dim = len(bounds)
    n_initial = min(n_initial, n_calls)

    def to_unit(x: np.ndarray) -> np.ndarray:
        return (x - lo) / (hi - lo)

    def from_unit(u: np.ndarray) -> np.ndarray:
        return lo + u * (hi - lo)

    x_unit: list[np.ndarray] = []
    y_vals: list[float] = []

    def evaluate(u: np.ndarray, iteration: int) -> None:
        x = from_unit(np.clip(u, 0.0, 1.0))
        y = float(func(x))
        x_unit.append(np.clip(u, 0.0, 1.0))
        y_vals.append(y)
        if on_step is not None:
            on_step(iteration, x, y)

    for i in range(n_initial):
        evaluate(rng.uniform(0.0, 1.0, size=n_dim), i)

    for i in range(n_initial, n_calls):
        x_train = np.vstack(x_unit)
        y_arr = np.array(y_vals)
        y_mean, y_std = float(y_arr.mean()), float(y_arr.std() + 1e-12)
        y_norm = (y_arr - y_mean) / y_std

        u_best = x_unit[int(np.argmin(y_arr))]
        candidates = np.vstack([
            rng.uniform(0.0, 1.0, size=(_N_RANDOM_CANDIDATES, n_dim)),
            np.clip(
                u_best + rng.normal(0.0, _LOCAL_SIGMA, size=(_N_LOCAL_CANDIDATES, n_dim)),
                0.0,
                1.0,
            ),
        ])
        mean, std = _gp_posterior(x_train, y_norm, candidates)
        ei = _expected_improvement(mean, std, float(y_norm.min()))
        evaluate(candidates[int(np.argmax(ei))], i)

    best = int(np.argmin(y_vals))
    return BOResult(
        x_best=from_unit(x_unit[best]),
        y_best=y_vals[best],
        x_iters=[from_unit(u) for u in x_unit],
        y_iters=list(y_vals),
    )
