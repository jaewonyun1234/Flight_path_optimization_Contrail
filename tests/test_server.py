"""gRPC round-trip: the served objective equals a direct in-process solve.

The generated stubs only exist after `scripts/gen_proto.sh` has run (they are
gitignored), so we importorskip them — locally `pytest` skips this file if you
have not generated stubs; in CI generation runs first, so it executes.
"""

import asyncio

import pytest

pytest.importorskip("service.generated.solver_pb2_grpc")

import grpc  # noqa: E402  (imported after the stub gate above)

from contrail_env import solve_cpsat  # noqa: E402
from service.generated import solver_pb2, solver_pb2_grpc  # noqa: E402
from service.server import build_scenario, start_server  # noqa: E402


def _config() -> "solver_pb2.ScenarioConfig":
    return solver_pb2.ScenarioConfig(
        seed=1,
        n_flights=3,
        n_issr_blobs=8,
        alpha_fuel=1.0,
        beta_contrail=5.0,
        gamma_disruption=0.5,
        corridor_frac=0.04,
        snapshot_window_s=300.0,
        time_limit_s=10.0,
        progress_topic="solve/test",
    )


async def _roundtrip(cfg: "solver_pb2.ScenarioConfig") -> "solver_pb2.SolveResponse":
    # No publisher in tests -> no ZMQ socket bound, no port conflicts.
    server, port = await start_server("localhost", 0)
    try:
        async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
            stub = solver_pb2_grpc.SolverStub(channel)
            return await stub.Solve(cfg)
    finally:
        await server.stop(0)


def test_server_objective_matches_direct_solve():
    cfg = _config()
    resp = asyncio.run(_roundtrip(cfg))

    # Independent direct solve on the same (fully seeded) scenario.
    evals, conflicts, buckets = build_scenario(cfg)
    direct = solve_cpsat(evals, conflicts, buckets, time_limit_s=10.0)

    assert resp.status == "OPTIMAL"
    assert abs(resp.objective - direct.objective) < 1e-6
    assert len(resp.choices) == cfg.n_flights
    assert resp.n_options_total == len(evals)
