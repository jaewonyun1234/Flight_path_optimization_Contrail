# Contrail-Aware Flight Optimization

[![CI](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml/badge.svg)](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml)

Pick one altitude profile per flight to minimise `fuel + α·contrail + β·disruption`,
subject to one option per flight, pairwise contrail conflicts, and sector-capacity
limits. The problem is built on a synthetic airspace, encoded as a QUBO, and solved
to optimality with OR-Tools CP-SAT behind a gRPC service. A PyQt6 dashboard drives
the service and streams solver progress over ZMQ.

CP-SAT is the ground-truth solver. Quantum backends (Pasqal, Xanadu) are planned but
not implemented — see [Roadmap](#roadmap).

## Install

```
pip install -e .
bash scripts/gen_proto.sh          # Windows: .\scripts\gen_proto.ps1
```

Requires Python 3.11+. The gRPC stubs are generated from `service/proto/solver.proto`,
not committed.

## Run

```
python -m service.server           # gRPC solver on localhost:50051
python gui/app.py                  # dashboard, in a second terminal
```

The dashboard has four panels: live CP-SAT convergence (over ZMQ), the conflict-graph
topology, QUBO matrix statistics (size, sparsity, penalty constants), and the
chosen-option trade-offs. The GUI calls the service — it never solves anything itself.

## Layout

```
contrail_env/   synthetic environment, QUBO assembly, and the CP-SAT solver
service/        gRPC service, ZMQ progress streaming, client
gui/            PyQt6 dashboard
tests/          environment build, CP-SAT vs brute-force, gRPC round-trip
```

## Development

```
pip install -e ".[dev]"
ruff check .
mypy contrail_env service
pytest
```

CI runs the same checks on every push and pull request.

## Roadmap

- Pasqal neutral-atom backend: analog-QAOA with Rydberg-blockade embedding of the
  conflict graph.
- Xanadu photonic backend: Gaussian Boson Sampling on the Takagi-decomposed adjacency.
- Approximation-ratio benchmarking of each backend against the CP-SAT optimum.

These are not implemented yet; the QUBO (`contrail_env/qubo.py`) is already the single
object they would consume.
