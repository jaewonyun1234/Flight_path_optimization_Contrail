# Contrail-Aware Flight Optimization

[![CI](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml/badge.svg)](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml)

Pick one altitude profile per flight to minimise `fuel + α·contrail + β·disruption`,
subject to one option per flight, pairwise contrail conflicts, and sector-capacity
limits. The problem is built on a synthetic airspace, encoded as a QUBO, and solved
three ways:

- **CP-SAT** (OR-Tools) — the classical ground-truth verifier, behind a gRPC service;
- **Pasqal analog-QAOA** — a hand-coded adiabatic Ω(t), δ(t) schedule on the Rydberg
  blockade Hamiltonian, tuned by Bayesian optimization (`contrail_env/pasqal_analog.py`);
- **Xanadu GBS** — Gaussian Boson Sampling of the Takagi-decomposed, WAW-weighted
  complement graph (`contrail_env/xanadu_gbs.py`).

A PyQt6 dashboard drives the service, streams solver progress over ZMQ, and runs the
head-to-head benchmark (approximation ratio vs the CP-SAT optimum with bootstrap CIs,
raw feasibility rate, wall clock).

The quantum pipelines need no quantum SDKs: each ships a dependency-free, physically
faithful backend (a split-operator state-vector simulator of the Rydberg Hamiltonian;
an exact Metropolis–Hastings sampler of the GBS distribution P(S) ∝ |Haf(B_S)|²).
Installing `pip install -e ".[quantum]"` switches them to Pulser's QuTiP emulator and
Strawberry Fields' gaussian backend automatically.

## Install

```
pip install -e .                   # core: the headless solver service
pip install -e ".[gui]"            # add the PyQt6 desktop dashboard
bash scripts/gen_proto.sh          # Windows: .\scripts\gen_proto.ps1
```

Requires Python 3.11+. Core deps are just the solver runtime (numpy, OR-Tools,
gRPC, ZMQ); the Qt/OpenGL dashboard stack lives in the `gui` extra so the
deployed server stays lean. The gRPC stubs are generated from
`service/proto/solver.proto`, not committed.

## Run

```
python -m service.server           # gRPC solver on localhost:50051
python gui/app.py                  # dashboard, in a second terminal ([gui] extra)
```

## Docker

The solver service is containerized (the headless gRPC server only — not the
desktop GUI). The bind host is read from the environment, so the container
binds `0.0.0.0` while local runs default to `localhost`.

### Pull the pre-built image (easiest)

A Docker image is published to the GitHub Container Registry on every push to
main. No cloning or building required — just Docker Desktop installed.

```
docker pull ghcr.io/jaewonyun1234/contrail-solver:latest
docker run --rm -p 50051:50051 -p 5556:5556 ghcr.io/jaewonyun1234/contrail-solver:latest
```

Then run the dashboard on your machine:

```
pip install -e ".[gui]"
python gui/app.py
```

The dashboard connects to the solver on `localhost:50051` automatically.

### Build locally from source

```
docker compose up --build          # build + run; gRPC on :50051, progress on :5556
# or, without compose:
docker build -t contrail-solver .
docker run --rm -p 50051:50051 -p 5556:5556 contrail-solver
```

CI builds the image, smoke-tests that the server boots, and publishes it to
ghcr.io on every push to main.

The dashboard has six tabs: live CP-SAT convergence (over ZMQ), the conflict-graph
topology, QUBO matrix statistics (size, sparsity, penalty constants), the
chosen-option trade-offs, the quantum benchmark (CP-SAT vs Pasqal vs Xanadu over
N seeds, with live convergence curves for the BO loop and the GBS sampler), and a
geographic map — the predicted ISSR risk as a marker overlay on a real Plotly
`geo` basemap (country borders / coastlines, drawn with SVG and bundled offline
vectors, so it needs no WebGL or network) with the chosen vs context routes on top.

The benchmark also runs headless:

```
python -m contrail_env.benchmark --flights 4 --seeds 5 --csv results.csv
```

## ISSR model — real weather instead of synthetic blobs

By default the airspace's contrail zones are synthetic Gaussian blobs. The
`contrail_ml` package replaces them with a trained model that predicts
ice-supersaturated regions (ISSRs) from real weather, bias-corrected against
in-situ IAGOS humidity, and wraps it in a full MLOps lifecycle (versioned data →
train → calibrate → register with MLflow → serve → monitor).

The model plugs in through the **same `ISSRField` interface** the synthetic
field uses, so `World` and the QUBO assembly are untouched — you flip a switch:

```python
from contrail_env import default_european_world
world = default_european_world(issr_source="ml", issr_kwargs={...})   # vs "synthetic"
```

Over gRPC the `ScenarioConfig` gained an `issr_source` field (default
`"synthetic"`, so existing clients are unaffected). Install the extra and try the
whole serving seam offline:

```
pip install -e ".[ml]"
python -m contrail_ml serve-check          # ML ISSR field -> CP-SAT solve
python -m contrail_ml train --synthetic --no-mlflow   # CV, fit, calibrate, baseline table
```

The science, the model, and how to read the model-vs-baselines table are in
[docs/ML.md](docs/ML.md); data sources (IAGOS / ARCO-ERA5 / GFS) in
[docs/DATA.md](docs/DATA.md). The hermetic tests use a guarded synthetic
fallback — no network, no credentials.

## Layout

```
contrail_env/   synthetic environment, QUBO assembly, CP-SAT solver,
                quantum pipelines (pasqal_analog, xanadu_gbs, quantum_common,
                bayes_opt) and the benchmark protocol (benchmark.py)
contrail_ml/    ISSR model (features, RHiCorrector, calibration), MLflow
                registry, serving (MLIssrField), monitoring, and the data
                loaders (IAGOS/ERA5/GFS) — the [ml] extra
service/        gRPC service, ZMQ progress streaming, client
gui/            PyQt6 dashboard
tests/          environment build, CP-SAT vs brute-force, quantum solvers vs
                brute-force, benchmark round-trip, gRPC round-trip, and the
                hermetic contrail_ml suite (test_ml_*.py)
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

- **Done:** real ISSR model (`contrail_ml`) replacing the synthetic field, with
  the train → calibrate → register → serve → monitor MLOps lifecycle (see
  [docs/ML.md](docs/ML.md)). Next: run it end-to-end on real IAGOS + ERA5 (the
  loaders are written; they need portal registration) and report the honest
  real-data comparison table.
- Run the Pasqal pipeline on real Pulser hardware: needs `[quantum]` extras plus a
  conflict graph that embeds as a valid unit-disk register (auto-detected; the
  built-in simulator is the fallback).
- Strawberry Fields X8 hardware demo on a reduced 8-mode subproblem.
- Larger instances for the Pasqal path (n > 20 qubits) via an MPS-style emulator tier.
