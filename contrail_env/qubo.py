"""
qubo.py — Conflict-graph construction + QUBO matrix assembly.

WHAT THIS MODULE DOES
=====================
Given a list of EvaluatedOptions (one per (flight, option_idx) pair),
this module produces:

    1. The CONFLICT GRAPH: which pairs of options can't both be chosen.
       Two options conflict iff they pass through the same ISSR cell
       at the same time (= contrail amplification risk).

    2. The CAPACITY BUCKETS: which (sector, time-bucket) pairs are
       capacity-limited, and which options charge to each one.

    3. The QUBO MATRIX Q: the upper-triangular n x n matrix whose
       quadratic form z^T Q z encodes the entire problem (cost + all
       penalties).

This matrix Q is the SINGLE OBJECT handed off to any optimizer:
CP-SAT, Pasqal's analog-QAOA, Xanadu's GBS, etc. The environment is
done at this point.

QUBO STRUCTURE
==============

    z = (x_{1,1}, x_{1,2}, ..., x_{F,K}, s_{1,0}, s_{1,1}, ...)
        |_________ option variables ____________|  |_ slack bits _|

    Q[i,i] = linear cost on variable i (cost + penalty diagonals)
    Q[i,j] = pairwise coupling for i < j (one-hot, conflict, capacity)
    The total energy is z^T Q z (NOT 0.5 z^T Q z; we use the convention
    that the off-diagonal entry Q[i,j] = 2 * J_{ij} directly).

KEY FORMULAS (matching the project plan §3)
============================================

    H = sum_i c_i x_i                                     [cost]
      + A * sum_f (sum_k x_{f,k} - 1)^2                   [one-hot]
      + B * sum_{(i,j) in conflicts} x_i x_j              [conflict]
      + C * sum_b (sum_{(f,k)} a_{f,k,b} x_{f,k}          [capacity]
                   + sum_t 2^t s_{b,t} - cap_b)^2

The penalty constants A, B, C are AUTO-COMPUTED to satisfy the bound
    A, B, C > c_max - c_min
with a configurable safety factor (default 2x).

ROAD-TRIP ANALOGY
=================
This module takes "here are all the possible trips each driver could
take, with costs, and here are the highway-patrol speed limits and
road-closure warnings" and converts it into one big number — the
"badness score" of any joint assignment of trip choices — that the
optimizer can then minimize.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import math
import numpy as np

from .flight import EvaluatedOption


# =============================================================================
# CONFLICT GRAPH
# =============================================================================

@dataclass
class ConflictEdge:
    """
    A conflict between two options. Both options can't be chosen at once.

    Attributes:
        i, j: indices into the flat list of all EvaluatedOptions.
        reason: human-readable explanation (e.g., "share ISSR cell (5,3,1,2)")
        shared_cells: the 4D cells that triggered the conflict.
    """
    i: int
    j: int
    reason: str
    shared_cells: list[tuple[int, int, int, int]] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ConflictEdge({self.i}<->{self.j}, {self.reason})"


def build_conflict_graph(
    evals: list[EvaluatedOption],
    world,
) -> list[ConflictEdge]:
    """
    Build the conflict graph: edges between option pairs that share at
    least one ISSR cell.

    Rules:
        - Options of the SAME flight are NOT in conflict (they're
          handled by the one-hot constraint instead).
        - Options of DIFFERENT flights conflict iff they share at least
          one 4D cell that lies in an ISSR.
        - Sharing a non-ISSR cell is NOT a conflict — separation in
          non-ISSR airspace is handled by capacity buckets, not pairwise
          conflicts.

    Complexity: O(n^2) pairwise comparison. For n <= 50 options this
    is microseconds. For larger n, switch to a per-cell inverted index.
    """
    edges: list[ConflictEdge] = []

    # Pre-compute the ISSR-cells set for each option for fast intersection.
    issr_cells_per_option: list[set] = []
    for ev in evals:
        s = set(c for c in ev.cells_visited if world.is_issr_cell(c))
        issr_cells_per_option.append(s)

    # Map each option's flight name -> for filtering same-flight pairs
    flight_of_option = [ev.flight_name for ev in evals]

    n = len(evals)
    for i in range(n):
        for j in range(i + 1, n):
            if flight_of_option[i] == flight_of_option[j]:
                continue  # Same flight; one-hot will handle it.
            shared = issr_cells_per_option[i] & issr_cells_per_option[j]
            if shared:
                edges.append(ConflictEdge(
                    i=i, j=j,
                    reason=f"share {len(shared)} ISSR cell(s)",
                    shared_cells=sorted(shared),
                ))
    return edges


# =============================================================================
# CAPACITY BUCKETS
# =============================================================================

@dataclass
class CapacityBucket:
    """
    One (sector, time-bucket) pair with a capacity constraint.

    Attributes:
        bucket_id:   (sector_idx, time_bucket_idx) — the b in v2 §2.6.3
        capacity:    cap_b
        members:     list of (option_index, weight) where option_index
                     is an index into evals; weight = a_{f,k,b}
                     (here always 1 since we don't allow partial occupancy)
    """
    bucket_id: tuple[int, int]
    capacity: int
    members: list[int] = field(default_factory=list)

    @property
    def n_slack_bits(self) -> int:
        """
        Number of binary slack bits needed for the inequality
            sum members <= cap
        rewritten as
            sum members + s = cap
        with s in {0, 1, ..., cap} encoded in binary as
            s = sum_t 2^t s_t.
        We need ceil(log2(cap+1)) bits.
        """
        return max(1, math.ceil(math.log2(self.capacity + 1)))


def build_capacity_buckets(
    evals: list[EvaluatedOption],
    world,
) -> list[CapacityBucket]:
    """
    Collect all (sector, time-bucket) pairs that have at least one
    option charging to them, and look up their capacities.

    Returns a list of CapacityBuckets, each carrying its capacity and
    the indices of options that occupy it.
    """
    # Build inverse: bucket -> list of option_indices that visit it
    inverse: dict[tuple[int, int], list[int]] = {}
    for opt_idx, ev in enumerate(evals):
        for b in ev.buckets:
            inverse.setdefault(b, []).append(opt_idx)

    buckets: list[CapacityBucket] = []
    for b, members in inverse.items():
        sec_idx, _t_idx = b
        cap = world.sectors.capacity_of(sec_idx, _t_idx)
        # Only keep buckets where capacity could actually be hit.
        # If at most `cap` options visit this bucket, the constraint
        # is automatically satisfied and we can skip it to save QUBO
        # variables.
        if len(members) > cap:
            buckets.append(CapacityBucket(
                bucket_id=b,
                capacity=cap,
                members=sorted(members),
            ))
    return buckets


# =============================================================================
# QUBO MATRIX
# =============================================================================

@dataclass
class QUBOInstance:
    """
    The final QUBO instance: matrix Q, variable map, and the constant
    energy offset.

    Energy formula:
        E(z) = z^T Q z + constant
    where z is a binary vector of length n = n_options + n_slack.

    Attributes:
        Q:                upper-triangular n x n numpy array
        n_options:        number of option variables
        n_slack:          number of slack-bit variables (n - n_options)
        option_index:     map from option_index in evals to QUBO variable id
                          (in this implementation they're the same:
                          var_id == evals index for the first n_options).
        slack_index:      map (bucket_id, bit_index) -> QUBO variable id
        constant:         additive constant (from expanding the squares)
        penalty_A, B, C:  the penalty constants used
        flight_groups:    for each flight, the list of option-variable
                          ids belonging to it (used by one-hot constraint)
    """
    Q: np.ndarray
    n_options: int
    n_slack: int
    option_index: dict[int, int]       # eval_index -> qubo_var
    slack_index: dict[tuple, int]       # (bucket_id, bit_idx) -> qubo_var
    constant: float
    penalty_A: float
    penalty_B: float
    penalty_C: float
    flight_groups: dict[str, list[int]]  # flight_name -> [var_ids]
    conflict_edges: list[ConflictEdge]
    capacity_buckets: list[CapacityBucket]

    @property
    def n(self) -> int:
        return self.n_options + self.n_slack

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


def assemble_qubo(
    evals: list[EvaluatedOption],
    conflict_edges: list[ConflictEdge],
    capacity_buckets: list[CapacityBucket],
    safety_factor: float = 2.0,
) -> QUBOInstance:
    """
    Assemble the full QUBO matrix.

    Variable ordering:
        var 0, 1, ..., n_options-1  =  option variables x_{f,k}
                                       (ordered the same as `evals`)
        var n_options, ...           =  slack bits for capacity buckets

    Penalty constants:
        Let delta = max(combined cost) - min(combined cost).
        A = B = C = safety_factor * (1 + delta).

    Returns: QUBOInstance with Q, the var maps, and metadata.
    """
    n_options = len(evals)

    # ----- 1. Index the variables -----
    option_index = {i: i for i in range(n_options)}

    # Slack bits: assign IDs after the options
    slack_index: dict[tuple, int] = {}
    next_var = n_options
    for bkt in capacity_buckets:
        for bit in range(bkt.n_slack_bits):
            slack_index[(bkt.bucket_id, bit)] = next_var
            next_var += 1
    n_slack = next_var - n_options
    n = n_options + n_slack

    # ----- 2. Compute penalty constants -----
    costs = [ev.cost_combined for ev in evals]
    c_max, c_min = max(costs), min(costs)
    delta = c_max - c_min
    A_penalty = safety_factor * (1.0 + delta)
    B_penalty = safety_factor * (1.0 + delta)
    C_penalty = safety_factor * (1.0 + delta)

    # ----- 3. Initialize Q and constant -----
    Q = np.zeros((n, n), dtype=float)
    constant = 0.0

    # ----- 4. Linear cost: sum_i c_i x_i -----
    for i, ev in enumerate(evals):
        _add_linear(Q, i, ev.cost_combined)

    # ----- 5. Group options by flight (for one-hot penalty) -----
    flight_groups: dict[str, list[int]] = {}
    for i, ev in enumerate(evals):
        flight_groups.setdefault(ev.flight_name, []).append(i)

    # ----- 6. One-hot penalty: A * (sum_k x_{f,k} - 1)^2 for each flight -----
    # Expand (sum_k x_k - 1)^2:
    #   = (sum_k x_k)^2 - 2 (sum_k x_k) + 1
    #   = sum_k x_k^2 + 2 sum_{k<l} x_k x_l - 2 sum_k x_k + 1
    #   = sum_k x_k       (since x_k^2 = x_k for binary)
    #     + 2 sum_{k<l} x_k x_l
    #     - 2 sum_k x_k
    #     + 1
    #   = -sum_k x_k + 2 sum_{k<l} x_k x_l + 1
    for f_name, group in flight_groups.items():
        for k in group:
            _add_linear(Q, k, -A_penalty)
        for ki, a in enumerate(group):
            for b in group[ki+1:]:
                _add_quadratic(Q, a, b, 2.0 * A_penalty)
        constant += A_penalty   # +1 per flight

    # ----- 7. Conflict penalty: B * sum x_i x_j -----
    for edge in conflict_edges:
        _add_quadratic(Q, edge.i, edge.j, B_penalty)

    # ----- 8. Capacity penalty: C * (sum a_{f,k,b} x + sum 2^t s - cap)^2 -----
    # Expand (S - cap)^2 where S = sum members + sum 2^t s_t:
    #   = S^2 - 2 cap S + cap^2
    # We need S^2 = sum_p (a_p)^2 x_p^2 + 2 sum_{p<q} a_p a_q x_p x_q
    # plus cross-terms with slack bits.
    for bkt in capacity_buckets:
        cap = bkt.capacity
        members = bkt.members  # all coefs a_{f,k,b} are 1 in this model
        slack_vars = [slack_index[(bkt.bucket_id, t)]
                      for t in range(bkt.n_slack_bits)]
        slack_weights = [2 ** t for t in range(bkt.n_slack_bits)]

        # Combine members and slacks into one list of (var, weight)
        terms = [(m, 1) for m in members] + list(zip(slack_vars, slack_weights))

        # (sum_i a_i z_i - cap)^2 = sum_i a_i^2 z_i + 2 sum_{i<j} a_i a_j z_i z_j
        #                          - 2 cap sum_i a_i z_i + cap^2
        # Diagonal: a_i^2 - 2 cap a_i
        for (v, w) in terms:
            coef = w*w - 2 * cap * w
            _add_linear(Q, v, C_penalty * coef)
        # Off-diagonal: 2 a_i a_j
        for i_idx in range(len(terms)):
            v1, w1 = terms[i_idx]
            for j_idx in range(i_idx + 1, len(terms)):
                v2, w2 = terms[j_idx]
                _add_quadratic(Q, v1, v2, C_penalty * 2 * w1 * w2)
        # Constant: cap^2
        constant += C_penalty * cap * cap

    return QUBOInstance(
        Q=Q,
        n_options=n_options,
        n_slack=n_slack,
        option_index=option_index,
        slack_index=slack_index,
        constant=constant,
        penalty_A=A_penalty,
        penalty_B=B_penalty,
        penalty_C=C_penalty,
        flight_groups=flight_groups,
        conflict_edges=conflict_edges,
        capacity_buckets=capacity_buckets,
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
    for f_name, group in qubo.flight_groups.items():
        s = sum(int(z[v]) for v in group)
        if s != 1:
            viols.append(f"one-hot violated for {f_name}: sum = {s}")

    # Conflicts
    for edge in qubo.conflict_edges:
        if z[edge.i] and z[edge.j]:
            viols.append(f"conflict edge {edge.i}-{edge.j} both = 1 "
                          f"({edge.reason})")

    # Capacity
    for bkt in qubo.capacity_buckets:
        s = sum(int(z[m]) for m in bkt.members)
        if s > bkt.capacity:
            viols.append(f"capacity bucket {bkt.bucket_id}: "
                          f"sum = {s} > cap = {bkt.capacity}")

    return (len(viols) == 0, viols)


def cost_of_assignment(z: np.ndarray, evals: list[EvaluatedOption]) -> float:
    """
    Compute the actual COMBINED COST (sum c_i x_i, with no penalties)
    for the chosen assignment.

    Use this to score a candidate AFTER checking feasibility.
    """
    z = np.asarray(z, dtype=int).reshape(-1)
    return sum(ev.cost_combined * int(z[i]) for i, ev in enumerate(evals))


def brute_force_optimum(qubo: QUBOInstance,
                         evals: list[EvaluatedOption]) -> tuple[np.ndarray, float, float]:
    """
    Find the ground state by brute force. ONLY USE for small n (< 25).

    Returns: (best_z, best_cost, best_energy).

    `best_cost` is the combined cost ignoring penalties; `best_energy`
    is z^T Q z + constant (should equal best_cost if feasible).
    """
    n = qubo.n
    assert n <= 24, f"Brute force won't fit: 2^{n} = {2**n} > 16M"
    best_z = None
    best_e = float("inf")
    best_c = float("inf")
    for k in range(2 ** n):
        z = np.array([(k >> b) & 1 for b in range(n)], dtype=int)
        e = qubo.energy(z)
        if e < best_e:
            feasible, _ = is_feasible(z, qubo)
            if feasible:
                c = cost_of_assignment(z, evals)
                if c < best_c:
                    best_e = e
                    best_c = c
                    best_z = z
    return best_z, best_c, best_e


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    from .world import default_european_world
    from .options import build_random_flights, build_and_evaluate_flight

    # Build a small world and 3 flights, evaluate options, build QUBO
    world = default_european_world(seed=7, n_issr_blobs=8)
    flights = build_random_flights(n_flights=3, world=world, seed=7)

    all_evals: list[EvaluatedOption] = []
    for f in flights:
        evs = build_and_evaluate_flight(f, world)
        all_evals.extend(evs)
    print(f"Total options: {len(all_evals)}")

    # Build conflicts + capacity buckets
    conflicts = build_conflict_graph(all_evals, world)
    buckets   = build_capacity_buckets(all_evals, world)
    print(f"Conflict edges: {len(conflicts)}")
    print(f"Capacity buckets (binding): {len(buckets)}")

    # Assemble QUBO
    qubo = assemble_qubo(all_evals, conflicts, buckets)
    print(f"QUBO size: n = {qubo.n} "
          f"({qubo.n_options} options + {qubo.n_slack} slack bits)")
    print(f"Penalty A = B = C = {qubo.penalty_A:.1f}")
    print(f"Constant = {qubo.constant:.1f}")

    # Brute-force the optimum (only if small enough)
    if qubo.n <= 20:
        best_z, best_c, best_e = brute_force_optimum(qubo, all_evals)
        print(f"\nBrute-force optimum:")
        print(f"  energy = {best_e:.1f}")
        print(f"  cost   = {best_c:.1f}")
        print(f"  z      = {best_z}")
        # Decode which option each flight picked
        for f_name, group in qubo.flight_groups.items():
            picked = [k for k in group if best_z[k]]
            opt = all_evals[picked[0]] if picked else None
            print(f"  {f_name} -> opt{opt.option_index} "
                  f"(fuel={opt.fuel_kg:.0f}kg, "
                  f"contrail={opt.contrail_cells}, "
                  f"disrupt={opt.disruption_FLmin:.1f})")
    else:
        print(f"\nSkipping brute force (n = {qubo.n} too large)")

    print("\nAll self-tests passed.")
