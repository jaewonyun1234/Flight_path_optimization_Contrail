"""Smoke test: the contrail_env scenario pipeline builds end to end."""

from contrail_env import (
    assemble_qubo,
    build_and_evaluate_flight,
    build_capacity_buckets,
    build_conflict_graph,
    build_random_flights,
    default_european_world,
)


def test_scenario_pipeline_builds():
    world = default_european_world(seed=1, n_issr_blobs=8)
    flights = build_random_flights(
        n_flights=3,
        world=world,
        seed=1,
        corridor_frac=0.05,
        snapshot_window_s=(0.0, 300.0),
    )

    evals = []
    for flight in flights:
        evals.extend(build_and_evaluate_flight(flight, world))

    assert len(evals) > 0

    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)

    # Assembling the QUBO must not raise and must size consistently.
    qubo = assemble_qubo(evals, conflicts, buckets)
    assert qubo.n_options == len(evals)
    assert qubo.n == qubo.n_options + qubo.n_slack
    assert qubo.Q.shape == (qubo.n, qubo.n)
