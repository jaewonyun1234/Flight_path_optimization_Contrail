"""CP-SAT solver tested against an independent brute-force oracle.

We do NOT use CP-SAT to check CP-SAT: the oracle (`enumerate_optimum`) is a
plain exhaustive search over one-hot assignments, so agreement between the two
is real evidence of correctness, not a tautology.
"""

from contrail_env import (
    build_and_evaluate_flight,
    build_capacity_buckets,
    build_conflict_graph,
    build_random_flights,
    default_european_world,
    enumerate_optimum,
    solve_cpsat,
)


def _build_small_instance():
    """Seed/size chosen so conflicts actually appear (tight corridor)."""
    world = default_european_world(seed=1, n_issr_blobs=8)
    flights = build_random_flights(
        n_flights=3,
        world=world,
        seed=1,
        corridor_frac=0.04,
        snapshot_window_s=(0.0, 300.0),
    )
    evals = []
    for flight in flights:
        evals.extend(build_and_evaluate_flight(flight, world))
    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)
    return evals, conflicts, buckets


def _assert_feasible(chosen_eval_indices, evals, conflicts, buckets, n_flights):
    selected = set(chosen_eval_indices)

    # One-hot: exactly one chosen option per flight.
    assert len(chosen_eval_indices) == n_flights
    assert len({evals[i].flight_name for i in chosen_eval_indices}) == n_flights

    # No conflict edge has both endpoints chosen.
    for e in conflicts:
        assert not (e.i in selected and e.j in selected)

    # No capacity bucket is exceeded.
    for b in buckets:
        assert sum(1 for m in b.members if m in selected) <= b.capacity


def test_cpsat_matches_brute_force_oracle():
    evals, conflicts, buckets = _build_small_instance()

    # The instance must be non-trivial (otherwise the test proves little).
    assert len(conflicts) > 0

    result = solve_cpsat(evals, conflicts, buckets, time_limit_s=10.0)
    _, oracle_obj = enumerate_optimum(evals, conflicts, buckets)

    assert result.status == "OPTIMAL"
    assert abs(result.objective - oracle_obj) < 1e-6


def test_cpsat_solution_is_feasible():
    evals, conflicts, buckets = _build_small_instance()
    result = solve_cpsat(evals, conflicts, buckets, time_limit_s=10.0)
    _assert_feasible(result.chosen_eval_indices, evals, conflicts, buckets, n_flights=3)
