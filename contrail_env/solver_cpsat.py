"""
solver_cpsat.py — Classical ground-truth solver using Google OR-Tools CP-SAT.

ROLE
====
This is the CLASSICAL baseline / ground-truth verifier for the contrail
flight-option selection problem. It consumes exactly the structures the
rest of `contrail_env` already produces:

    evals     : list[EvaluatedOption]   (one per (flight, option) pair)
    conflicts : list[ConflictEdge]      (pairs that cannot both be chosen)
    buckets   : list[CapacityBucket]    (sector x time-bucket capacity caps)

and models the assignment problem DIRECTLY in CP-SAT. Unlike the QUBO
formulation (qubo.py), CP-SAT handles the one-hot and inequality
constraints natively, so there are NO penalty constants and NO slack
bits — the model is exact and small.

WHY CP-SAT AND NOT THE QUBO?
============================
The QUBO matrix exists so the *quantum* backends (Pasqal, Xanadu) have a
single object to embed. CP-SAT does not need it: it is a constraint
solver, so we give it the constraints verbatim. For instances of this
size (<= ~60 binary variables) CP-SAT finds the proven optimum in
milliseconds, which is exactly what makes it a trustworthy verifier.

OBJECTIVE SCALING
=================
CP-SAT optimises integers. Combined costs are floats (kg-fuel-equivalent),
so we scale by 100 (one cent of resolution) and round to the nearest
integer before handing them to the solver, then divide the reported
objective back by 100. The brute-force oracle below uses the *identical*
scaling so the two agree to the cent.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from typing import Callable

from ortools.sat.python import cp_model

from .flight import EvaluatedOption
from .qubo import CapacityBucket, ConflictEdge

# One unit = one "cent" of combined cost. CP-SAT works in these integers.
_COST_SCALE = 100


def _scaled_cost(ev: EvaluatedOption) -> int:
    """Integer objective coefficient for one option (combined cost x 100)."""
    return int(round(_COST_SCALE * ev.cost_combined))


def _group_by_flight(evals: list[EvaluatedOption]) -> "OrderedDict[str, list[int]]":
    """Map flight_name -> ordered list of eval indices belonging to it.

    Insertion order is preserved so the one-hot groups (and the brute-force
    enumeration) are deterministic across runs.
    """
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for idx, ev in enumerate(evals):
        groups.setdefault(ev.flight_name, []).append(idx)
    return groups


# =============================================================================
# RESULT TYPE
# =============================================================================

@dataclass
class CPSATResult:
    """Outcome of a CP-SAT solve.

    Attributes:
        chosen:              flight_name -> chosen option_index
        chosen_eval_indices: indices into `evals` whose variable is 1
        objective:           combined cost of the optimum (already /100)
        status:              "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN"
        wall_clock_s:        wall-clock seconds spent in solve()
        n_improvements:      number of improved incumbent solutions seen
    """
    chosen: dict[str, int]
    chosen_eval_indices: list[int]
    objective: float
    status: str
    wall_clock_s: float
    n_improvements: int = 0


# =============================================================================
# SOLUTION CALLBACK — counts improvements and forwards progress
# =============================================================================

class _ProgressCallback(cp_model.CpSolverSolutionCallback):
    """Fires once per improved incumbent; forwards (index, objective) out.

    CP-SAT calls `on_solution_callback` every time it finds a new, strictly
    better feasible solution. We use that to (a) count improvements and
    (b) stream the convergence curve to whoever passed `on_progress`
    (the gRPC server wires this to ZMQ).
    """

    def __init__(self, on_progress: Callable[[int, float], None] | None) -> None:
        super().__init__()
        self._on_progress = on_progress
        self.n_improvements = 0

    def on_solution_callback(self) -> None:  # noqa: N802 (OR-Tools API name)
        self.n_improvements += 1
        if self._on_progress is not None:
            # ObjectiveValue() is the scaled integer objective as a float.
            self._on_progress(self.n_improvements, self.ObjectiveValue() / _COST_SCALE)


# =============================================================================
# THE SOLVER
# =============================================================================

def solve_cpsat(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
    *,
    time_limit_s: float = 30.0,
    on_progress: Callable[[int, float], None] | None = None,
) -> CPSATResult:
    """Solve the flight-option selection problem to optimality with CP-SAT.

    Model:
        * one BoolVar x[i] per eval index i
        * one-hot: exactly one option chosen per flight
        * conflict: x[i] + x[j] <= 1 for every ConflictEdge
        * capacity: sum(x[m] for m in bucket.members) <= bucket.capacity
        * minimise sum(round(100 * cost_combined) * x[i])

    The optional `on_progress(improvement_index, current_objective)` callback
    is invoked once per improved incumbent solution (objective already /100).
    """
    model = cp_model.CpModel()

    # One boolean per (flight, option).
    x = [model.new_bool_var(f"x_{i}") for i in range(len(evals))]

    # One-hot: exactly one option per flight.
    groups = _group_by_flight(evals)
    for members in groups.values():
        model.add(sum(x[i] for i in members) == 1)

    # Pairwise conflicts: at most one of a conflicting pair.
    for e in conflicts:
        model.add(x[e.i] + x[e.j] <= 1)

    # Sector capacity: occupancy of each (sector, time-bucket) <= cap.
    for b in buckets:
        model.add(sum(x[m] for m in b.members) <= b.capacity)

    # Objective: minimise total combined cost (scaled to integers).
    model.minimize(sum(_scaled_cost(ev) * x[i] for i, ev in enumerate(evals)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)

    callback = _ProgressCallback(on_progress)

    t0 = time.perf_counter()
    status = solver.solve(model, callback)
    wall_clock_s = time.perf_counter() - t0

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
    }.get(status, "UNKNOWN")

    chosen: dict[str, int] = {}
    chosen_eval_indices: list[int] = []
    objective = float("inf")

    if status_name in ("OPTIMAL", "FEASIBLE"):
        for i, ev in enumerate(evals):
            if solver.value(x[i]) == 1:
                chosen[ev.flight_name] = ev.option_index
                chosen_eval_indices.append(i)
        objective = solver.objective_value / _COST_SCALE

    return CPSATResult(
        chosen=chosen,
        chosen_eval_indices=chosen_eval_indices,
        objective=objective,
        status=status_name,
        wall_clock_s=wall_clock_s,
        n_improvements=callback.n_improvements,
    )


# =============================================================================
# INDEPENDENT VERIFICATION ORACLE — do NOT use CP-SAT to check CP-SAT
# =============================================================================

def enumerate_optimum(
    evals: list[EvaluatedOption],
    conflicts: list[ConflictEdge],
    buckets: list[CapacityBucket],
) -> tuple[dict[str, int], float]:
    """Brute-force the optimum over all one-hot assignments (K^F of them).

    Returns (best_chosen, best_objective) among the assignments that satisfy
    every conflict and capacity constraint, using the SAME integer cost
    scaling as `solve_cpsat`, so the two objectives agree to the cent.

    For small (F, K) only — this is the test oracle, independent of CP-SAT.
    """
    groups = _group_by_flight(evals)
    flight_names = list(groups.keys())
    choice_lists = [groups[name] for name in flight_names]

    # Safety rail: this is exponential. Refuse obviously oversized instances.
    n_combos = 1
    for cl in choice_lists:
        n_combos *= max(1, len(cl))
    assert n_combos <= 2_000_000, (
        f"enumerate_optimum is a small-instance oracle; {n_combos} "
        "combinations is too many — use solve_cpsat instead."
    )

    best_scaled = None  # int or None
    best_chosen: dict[str, int] = {}

    for combo in product(*choice_lists):
        selected = set(combo)

        # Conflicts: no edge may have both endpoints selected.
        if any(e.i in selected and e.j in selected for e in conflicts):
            continue
        # Capacity: occupancy of each bucket within its cap.
        if any(
            sum(1 for m in b.members if m in selected) > b.capacity
            for b in buckets
        ):
            continue

        scaled = sum(_scaled_cost(evals[i]) for i in combo)
        if best_scaled is None or scaled < best_scaled:
            best_scaled = scaled
            best_chosen = {
                name: evals[i].option_index
                for name, i in zip(flight_names, combo)
            }

    if best_scaled is None:
        return {}, float("inf")
    return best_chosen, best_scaled / _COST_SCALE
