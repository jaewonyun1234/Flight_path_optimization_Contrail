"""
qubo.py — QUBO matrix assembly for the flight-option selection problem.

WHAT THIS MODULE DOES
=====================
Given a Scenario (costs, flight membership, conflict pairs — see
problem.py), this module produces the QUBO MATRIX Q: the upper-triangular
n x n matrix whose quadratic form z^T Q z encodes the entire problem
(cost + all penalties).

This matrix Q is the SINGLE OBJECT handed off to any optimizer: the
brute-force enumerator, the random baseline, or the analog-QAOA solver.
The environment is done at this point.

QUBO STRUCTURE
==============

    z = (x_{1,1}, x_{1,2}, ..., x_{F,K})     one binary per (flight, level)

    Q[i,i] = linear cost on variable i (cost + penalty diagonals)
    Q[i,j] = pairwise coupling for i < j (one-hot, conflict)
    The total energy is z^T Q z (NOT 0.5 z^T Q z; we use the convention
    that the off-diagonal entry Q[i,j] = 2 * J_{ij} directly).

KEY FORMULAS
============

    H = sum_i c_i x_i                                     [cost]
      + A * sum_f (sum_k x_{f,k} - 1)^2                   [one-hot]
      + B * sum_{(i,j) in conflicts} x_i x_j              [conflict]

The penalty constants A, B are AUTO-COMPUTED to satisfy the bound
    A, B > c_max - c_min
with a configurable safety factor (default 2x). WHY: breaking a
constraint must never pay. The most a violation could ever "save" is the
full cost spread c_max - c_min, so any penalty above that spread makes
every infeasible state strictly worse than the best feasible one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import Scenario

# =============================================================================
# QUBO MATRIX
# =============================================================================

@dataclass
class QUBOInstance:
    """
    The final QUBO instance: matrix Q, flight groups, and the constant
    energy offset.

    Energy formula:
        E(z) = z^T Q z + constant
    where z is a binary vector of length n = n_flights * n_options.

    Attributes:
        Q:              upper-triangular n x n numpy array
        constant:       additive constant (from expanding the squares)
        penalty_A, B:   the penalty constants used
        flight_groups:  for each flight, the list of variable ids
                        belonging to it (used by the one-hot constraint)
        conflicts:      the cross-flight conflict pairs (i, j), i < j
        costs:          the raw per-option costs (no penalties)
    """
    Q: np.ndarray
    constant: float
    penalty_A: float
    penalty_B: float
    flight_groups: list[list[int]]
    conflicts: list[tuple[int, int]]
    costs: np.ndarray

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def energy(self, z: np.ndarray) -> float:
        """Evaluate E(z) = z^T Q z + constant for a binary vector z."""
        z = np.asarray(z, dtype=float).reshape(-1)
        assert z.shape == (self.n,), \
            f"Expected z of length {self.n}, got {z.shape}"
        return float(z @ self.Q @ z + self.constant)


def _add_quadratic(Q: np.ndarray, i: int, j: int, coef: float) -> None:
    """
    Add `coef * z_i * z_j` to the upper-triangular QUBO matrix.

    Diagonal (i == j) uses x^2 = x for binary, so adding `coef * x_i^2`
    is the same as adding `coef` to Q[i, i].

    Off-diagonal (i != j) stores `coef` in the upper triangle Q[min, max].
    """
    if i == j:
        Q[i, i] += coef
    elif i < j:
        Q[i, j] += coef
    else:
        Q[j, i] += coef


def _add_linear(Q: np.ndarray, i: int, coef: float) -> None:
    """Add `coef * z_i` to Q (goes on the diagonal because x^2 = x)."""
    Q[i, i] += coef


def assemble_qubo(scenario: Scenario, safety_factor: float = 2.0) -> QUBOInstance:
    """
    Assemble the full QUBO matrix from a Scenario.

    Variable ordering is the scenario's: var i = option (i mod K) of
    flight (i div K).

    Penalty constants:
        Let delta = max(cost) - min(cost).
        A = B = safety_factor * (1 + delta)
    which satisfies the A, B > c_max - c_min bound (the +1 keeps the
    penalty strictly positive even when all costs are equal).
    """
    n = scenario.n_vars
    costs = np.asarray(scenario.costs, dtype=float)

    # ----- 1. Compute penalty constants -----
    delta = float(costs.max() - costs.min())
    A_penalty = safety_factor * (1.0 + delta)
    B_penalty = safety_factor * (1.0 + delta)

    # ----- 2. Initialize Q and constant -----
    Q = np.zeros((n, n), dtype=float)
    constant = 0.0

    # ----- 3. Linear cost: sum_i c_i x_i -----
    for i in range(n):
        _add_linear(Q, i, float(costs[i]))

    # ----- 4. One-hot penalty: A * (sum_k x_{f,k} - 1)^2 for each flight -----
    # Expand (sum_k x_k - 1)^2:
    #   = (sum_k x_k)^2 - 2 (sum_k x_k) + 1
    #   = sum_k x_k^2 + 2 sum_{k<l} x_k x_l - 2 sum_k x_k + 1
    #   = sum_k x_k       (since x_k^2 = x_k for binary)
    #     + 2 sum_{k<l} x_k x_l
    #     - 2 sum_k x_k
    #     + 1
    #   = -sum_k x_k + 2 sum_{k<l} x_k x_l + 1
    flight_groups = scenario.groups()
    for group in flight_groups:
        for k in group:
            _add_linear(Q, k, -A_penalty)
        for ki, a in enumerate(group):
            for b in group[ki + 1:]:
                _add_quadratic(Q, a, b, 2.0 * A_penalty)
        constant += A_penalty   # +1 per flight

    # ----- 5. Conflict penalty: B * sum x_i x_j -----
    for (i, j) in scenario.conflicts:
        _add_quadratic(Q, i, j, B_penalty)

    return QUBOInstance(
        Q=Q,
        constant=constant,
        penalty_A=A_penalty,
        penalty_B=B_penalty,
        flight_groups=flight_groups,
        conflicts=list(scenario.conflicts),
        costs=costs,
    )


# =============================================================================
# VERIFICATION HELPERS
# =============================================================================

def is_feasible(z: np.ndarray, qubo: QUBOInstance) -> tuple[bool, list[str]]:
    """
    Check whether a binary vector z satisfies all constraints.

    Returns (feasible, violations) where violations is a list of
    human-readable strings describing each broken constraint.
    """
    z = np.asarray(z, dtype=int).reshape(-1)
    viols: list[str] = []

    # One-hot
    for f, group in enumerate(qubo.flight_groups):
        s = sum(int(z[v]) for v in group)
        if s != 1:
            viols.append(f"one-hot violated for flight {f}: sum = {s}")

    # Conflicts
    for (i, j) in qubo.conflicts:
        if z[i] and z[j]:
            viols.append(f"conflict edge {i}-{j} both = 1")

    return (len(viols) == 0, viols)


def cost_of_assignment(z: np.ndarray, qubo: QUBOInstance) -> float:
    """
    Compute the actual COST (sum c_i x_i, with no penalties) for the
    chosen assignment.

    Use this to score a candidate AFTER checking feasibility.
    """
    z = np.asarray(z, dtype=int).reshape(-1)
    return float(np.dot(qubo.costs, z))
