"""
server.py — Async gRPC server exposing the CP-SAT solver.

The wire contract is a SCENARIO CONFIG (see proto/solver.proto): the client
sends sizes/weights/seed, and the server reconstructs the entire contrail_env
scenario from them, solves it with CP-SAT, and returns the chosen option per
flight. Because the scenario is fully seeded, the same config always yields
the same problem — which is what makes the round-trip test exact.

The blocking CP-SAT solve runs in a thread-pool executor so it never stalls
the asyncio event loop. Progress (one event per improved incumbent) is handed
to an `on_progress` callback; Task 4 wires that callback to a ZMQ publisher.
"""

from __future__ import annotations

import asyncio
from typing import Callable

import grpc

from contrail_env import (
    CapacityBucket,
    ConflictEdge,
    CPSATResult,
    EvaluatedOption,
    build_and_evaluate_flight,
    build_capacity_buckets,
    build_conflict_graph,
    build_random_flights,
    default_european_world,
    solve_cpsat,
)

from .generated import solver_pb2, solver_pb2_grpc

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 50051

# Fallback cost weights when a client leaves all three at 0 (proto3 scalars
# cannot tell "unset" from "0", so an all-zero triple means "use the env
# defaults" rather than the degenerate all-costs-zero objective).
_DEFAULT_WEIGHTS = (1.0, 5.0, 0.5)


# =============================================================================
# SCENARIO CONSTRUCTION + SOLVE (pure, reusable, no gRPC types)
# =============================================================================

def build_scenario(
    cfg: "solver_pb2.ScenarioConfig",
) -> tuple[list[EvaluatedOption], list[ConflictEdge], list[CapacityBucket]]:
    """Reconstruct the contrail_env scenario from a ScenarioConfig.

    Everything is seeded, so identical configs produce identical problems.
    """
    world = default_european_world(seed=cfg.seed, n_issr_blobs=cfg.n_issr_blobs)
    flights = build_random_flights(
        n_flights=cfg.n_flights,
        world=world,
        seed=cfg.seed,
        corridor_frac=cfg.corridor_frac or 0.05,
        snapshot_window_s=(0.0, cfg.snapshot_window_s or 300.0),
    )

    weights = (cfg.alpha_fuel, cfg.beta_contrail, cfg.gamma_disruption)
    if weights == (0.0, 0.0, 0.0):
        weights = _DEFAULT_WEIGHTS

    evals: list[EvaluatedOption] = []
    for flight in flights:
        evals.extend(build_and_evaluate_flight(flight, world, cost_weights=weights))

    conflicts = build_conflict_graph(evals, world)
    buckets = build_capacity_buckets(evals, world)
    return evals, conflicts, buckets


def solve_scenario(
    cfg: "solver_pb2.ScenarioConfig",
    on_progress: Callable[[int, float], None] | None = None,
) -> tuple[CPSATResult, int, list[EvaluatedOption]]:
    """Build the scenario for `cfg` and solve it. Returns (result, n_conflicts, evals)."""
    evals, conflicts, buckets = build_scenario(cfg)
    result = solve_cpsat(
        evals,
        conflicts,
        buckets,
        time_limit_s=cfg.time_limit_s or 10.0,
        on_progress=on_progress,
    )
    return result, len(conflicts), evals


def make_response(
    result: CPSATResult,
    n_conflicts: int,
    evals: list[EvaluatedOption],
) -> "solver_pb2.SolveResponse":
    """Pack a CPSATResult + scenario sizes into a SolveResponse message."""
    choices = []
    for i in result.chosen_eval_indices:
        ev = evals[i]
        choices.append(
            solver_pb2.FlightChoice(
                flight_name=ev.flight_name,
                chosen_option=ev.option_index,
                fuel_kg=ev.fuel_kg,
                contrail_cells=ev.contrail_cells,
                disruption_flmin=ev.disruption_FLmin,
            )
        )
    return solver_pb2.SolveResponse(
        objective=result.objective,
        status=result.status,
        wall_clock_s=result.wall_clock_s,
        n_conflicts=n_conflicts,
        n_options_total=len(evals),
        choices=choices,
    )


# =============================================================================
# GRPC SERVICER
# =============================================================================

class SolverServicer(solver_pb2_grpc.SolverServicer):
    """Implements the Solver service. One unary RPC: Solve."""

    def _make_progress_callback(
        self, cfg: "solver_pb2.ScenarioConfig"
    ) -> Callable[[int, float], None] | None:
        """Progress sink for a single solve.

        Task 3 just logs the convergence curve; Task 4 overrides this to
        publish each improvement over ZMQ on `cfg.progress_topic`.
        """
        def _log(improvement: int, objective: float) -> None:
            print(f"[progress] improvement {improvement}: objective {objective:.2f}", flush=True)

        return _log

    async def Solve(  # noqa: N802 (gRPC method name is fixed by the proto)
        self,
        request: "solver_pb2.ScenarioConfig",
        context: grpc.aio.ServicerContext,
    ) -> "solver_pb2.SolveResponse":
        on_progress = self._make_progress_callback(request)

        # CP-SAT is blocking; run it off the event loop.
        loop = asyncio.get_running_loop()
        result, n_conflicts, evals = await loop.run_in_executor(
            None, lambda: solve_scenario(request, on_progress)
        )
        return make_response(result, n_conflicts, evals)


# =============================================================================
# SERVER LIFECYCLE
# =============================================================================

async def start_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> tuple[grpc.aio.Server, int]:
    """Create, bind, and start the server. Returns (server, bound_port).

    Pass port=0 to bind an ephemeral port (used by the test fixture); the
    actually-bound port is returned.
    """
    server = grpc.aio.server()
    solver_pb2_grpc.add_SolverServicer_to_server(SolverServicer(), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    await server.start()
    return server, bound_port


async def serve_forever(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server, bound_port = await start_server(host, port)
    print(f"Solver gRPC server ready on {host}:{bound_port}", flush=True)
    await server.wait_for_termination()


def main() -> None:
    asyncio.run(serve_forever())


if __name__ == "__main__":
    main()
