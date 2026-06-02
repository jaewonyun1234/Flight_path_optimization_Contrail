"""
client.py — Thin synchronous gRPC client for the Solver service.

The GUI worker thread uses this to issue a blocking Solve call. The server is
async (grpc.aio) but the wire protocol is identical, so a plain synchronous
client interoperates with it without any special handling.
"""

from __future__ import annotations

import grpc

from .generated import solver_pb2, solver_pb2_grpc

DEFAULT_SERVER_ADDRESS = "localhost:50051"


class SolverClient:
    """Wraps a gRPC channel and exposes a single typed `solve` call."""

    def __init__(self, address: str = DEFAULT_SERVER_ADDRESS) -> None:
        self.address = address
        self._channel = grpc.insecure_channel(address)
        self._stub = solver_pb2_grpc.SolverStub(self._channel)

    def solve(self, cfg: solver_pb2.ScenarioConfig) -> solver_pb2.SolveResponse:
        """Send a ScenarioConfig and block until the SolveResponse returns."""
        return self._stub.Solve(cfg)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> SolverClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
