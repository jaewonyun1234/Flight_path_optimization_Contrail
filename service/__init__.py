"""service — gRPC/ZMQ API layer over the contrail_env CP-SAT solver.

This package exposes the classical solver as a network service:

    * proto/solver.proto   — the gRPC contract (ScenarioConfig -> SolveResponse)
    * generated/           — protoc-generated stubs (gitignored; see scripts/gen_proto.sh)
    * progress.py          — ZMQ pub/sub helpers for streaming solver progress
    * server.py            — async gRPC server: builds the scenario, solves, streams
    * client.py            — thin synchronous gRPC client used by the GUI
"""
