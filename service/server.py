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
from collections.abc import Callable

import grpc

from contrail_env import (
    CapacityBucket,
    ConflictEdge,
    CPSATResult,
    EvaluatedOption,
    solve_cpsat,
)

from .generated import solver_pb2, solver_pb2_grpc
from .progress import DEFAULT_PUB_ADDRESS, ProgressPublisher
from .scenario import build_scenario_full

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 50051


# =============================================================================
# SCENARIO CONSTRUCTION + SOLVE (pure, reusable, no gRPC types)
# =============================================================================

def build_scenario(
    cfg: solver_pb2.ScenarioConfig,
) -> tuple[list[EvaluatedOption], list[ConflictEdge], list[CapacityBucket]]:
    """Solver inputs for a ScenarioConfig.

    Delegates to service.scenario (shared with the GUI) and drops the world
    and flight objects, which the solver itself does not need.
    """
    _world, _flights, evals, conflicts, buckets = build_scenario_full(cfg)
    return evals, conflicts, buckets


def solve_scenario(
    cfg: solver_pb2.ScenarioConfig,
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
) -> solver_pb2.SolveResponse:
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
    """Implements the Solver service. One unary RPC: Solve.

    An optional ProgressPublisher streams each improved incumbent over ZMQ.
    When no publisher is supplied (e.g. in tests), progress is only logged.
    """

    def __init__(self, publisher: ProgressPublisher | None = None) -> None:
        self._publisher = publisher

    def _make_progress_callback(
        self, cfg: solver_pb2.ScenarioConfig
    ) -> Callable[[int, float], None] | None:
        """Build the per-solve progress sink bound to this request's topic."""
        publisher = self._publisher
        topic = cfg.progress_topic or "solve/progress"

        def _on_progress(improvement: int, objective: float) -> None:
            print(f"[progress] improvement {improvement}: objective {objective:.2f}", flush=True)
            if publisher is not None:
                publisher.publish(topic, improvement, objective)

        return _on_progress

    async def Solve(  # noqa: N802 (gRPC method name is fixed by the proto)
        self,
        request: solver_pb2.ScenarioConfig,
        context: grpc.aio.ServicerContext,
    ) -> solver_pb2.SolveResponse:
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
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    publisher: ProgressPublisher | None = None,
) -> tuple[grpc.aio.Server, int]:
    """Create, bind, and start the server. Returns (server, bound_port).

    Pass port=0 to bind an ephemeral port (used by the test fixture); the
    actually-bound port is returned. `publisher` is optional so tests can
    run without binding a ZMQ socket.
    """
    server = grpc.aio.server()
    solver_pb2_grpc.add_SolverServicer_to_server(SolverServicer(publisher), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    await server.start()
    return server, bound_port


async def serve_forever(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    progress_address: str = DEFAULT_PUB_ADDRESS,
) -> None:
    publisher = ProgressPublisher(progress_address).bind()
    server, bound_port = await start_server(host, port, publisher)
    print(f"Solver gRPC server ready on {host}:{bound_port}", flush=True)
    print(f"Publishing progress on ZMQ {progress_address}", flush=True)
    await server.wait_for_termination()


def main() -> None:
    asyncio.run(serve_forever())


if __name__ == "__main__":
    main()
