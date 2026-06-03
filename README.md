# Contrail-Aware Flight-Option Optimization

[![CI](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml/badge.svg)](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml)

A contrail-aware flight-option optimizer. A fully synthetic airspace
environment (`contrail_env`) turns a small fleet of flights — each with a few
candidate altitude profiles — into a constrained assignment problem (pick one
option per flight, avoid pairwise contrail conflicts, respect sector
capacities). A [Google OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)
solver finds the proven optimum behind a **gRPC** API; solver progress streams
to clients over **ZMQ** pub/sub; and a **PyQt6** desktop app configures, runs,
and monitors solves — live-plotting convergence and visualizing the problem in
a 2D ISSR map and an interactive 3D trajectory view.

The GUI never solves anything itself: it calls the service. That client/server
split is the point — it mirrors a Python-bindings-over-gRPC/ZMQ stack with
PyQt apps on top.

## Architecture

```
┌──────────────────────────┐         gRPC  Solve(ScenarioConfig) → SolveResponse
│   PyQt6 Desktop Client    │ ───────────────────────────────────────────────────▶ │
│  - scenario controls      │                                                        │
│  - "Solve" button         │ ◀───────────────────────────────────────────────────  │
│  - live objective plot    │     ZMQ pub/sub: progress (improvement, objective)  ◀──│
│  - ISSR map + routes      │                                                      ◀──│
│  - results table          │                                                       ▼▼
└──────────────────────────┘                          ┌──────────────────────────────────┐
                                                       │          Solver Service           │
                                                       │  builds contrail_env scenario      │
                                                       │  → CP-SAT solve                    │
                                                       │  → publishes progress over ZMQ     │
                                                       │  → returns chosen option/flight    │
                                                       └──────────────────────────────────┘
```

The wire request is a **scenario config** (seed, sizes, cost weights), not a
pre-built problem: the server reconstructs the entire seeded `contrail_env`
scenario from it, so the same config always produces the same problem — which
is what makes the round-trip test exact.

## Layout

```
contrail_env/        synthetic environment (units, ISSR field, airspace, aircraft,
                     flights, options, conflict graph, capacity buckets, QUBO)
  └─ solver_cpsat.py CP-SAT ground-truth solver + brute-force verification oracle
service/             the API layer
  ├─ proto/          gRPC contract (solver.proto)
  ├─ generated/      protoc-generated stubs (gitignored — regenerate, see below)
  ├─ progress.py     ZMQ publisher + subscriber helpers
  ├─ server.py       async gRPC server (builds scenario, solves, streams progress)
  └─ client.py       thin synchronous gRPC client used by the GUI
gui/app.py           PyQt6 desktop application
tests/               env build, CP-SAT vs brute-force oracle, gRPC round-trip
scripts/gen_proto.sh regenerates the gRPC stubs from the .proto
```

## Run it

```bash
# 1. Install the package and its dependencies (Python 3.11+)
pip install -e .

# 2. Generate the gRPC stubs (the generated/ folder is gitignored)
bash scripts/gen_proto.sh        # macOS / Linux / Git Bash
# Windows PowerShell has no `bash` — use the native script instead:
#   .\scripts\gen_proto.ps1

# 3. Start the solver service (one terminal)
python -m service.server

# 4. Launch the desktop client (another terminal)
python gui/app.py
```

> **Windows note:** steps that show `bash scripts/gen_proto.sh` have a
> PowerShell equivalent at `scripts/gen_proto.ps1`. The CI runs the `.sh`
> version on Linux; both produce identical stubs.

Click **Solve**: the objective curve updates live as CP-SAT improves its
incumbent (streamed over ZMQ); the **3D trajectories** tab shows each flight's
chosen route weaving through the ISSR contrail zones (drag to rotate); the **2D
ISSR map** tab shows the field at a selected flight level with routes overlaid;
and the results table fills with the chosen option, fuel, contrail cells, and
disruption per flight. Each flight starts at its own (seed-randomized) cruise
level, so the 3D view spans multiple altitudes.

> A PyQt6 GUI needs a real display. CI runs the env/solver/server tests
> headlessly; pull the repo and run `gui/app.py` locally to see the window and
> the live plot.

## Development

```bash
pip install -e ".[dev]"
bash scripts/gen_proto.sh
ruff check .
mypy contrail_env service
pytest -q
```

CI (GitHub Actions) runs exactly these steps on every push and pull request.

The nine original `contrail_env` modules are treated as vendored source: ruff
and mypy enforce the new solver, service, and GUI code, while the pre-existing
environment package is excluded from those checks.

## Roadmap / future work (not yet implemented)

This repository currently delivers the **classical** slice end-to-end. Planned,
clearly-not-yet-built extensions:

- **Pasqal neutral-atom backend** — analog-QAOA with Bayesian optimization over
  hand-coded `Ω(t)`, `δ(t)` pulses, embedding the conflict graph via Rydberg
  blockade.
- **Xanadu photonic backend** — Gaussian Boson Sampling for weighted
  max-clique / MIS via a Takagi decomposition of the conflict-graph adjacency.
- **CP-SAT-verified benchmarking** — approximation-ratio comparisons of each
  quantum backend against the CP-SAT optimum on identical instances.

The QUBO matrix (`contrail_env/qubo.py`) already exists as the single object
those quantum backends would consume; the CP-SAT solver here is the
ground-truth verifier they would be measured against.
