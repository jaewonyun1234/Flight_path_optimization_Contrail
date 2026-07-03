# Contrail-Aware Flight Optimization

[![CI](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml/badge.svg)](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml)

Pick one altitude profile per flight to minimise `fuel + α·contrail + β·disruption`,
subject to one option per flight, pairwise contrail conflicts, and sector-capacity
limits. The problem is built on a synthetic airspace, encoded as a QUBO, and solved
three ways:

- **CP-SAT** (OR-Tools) — the classical ground-truth verifier;
- **Pasqal analog-QAOA** — a hand-coded adiabatic Ω(t), δ(t) schedule on the Rydberg
  blockade Hamiltonian, tuned by Bayesian optimization (`contrail_env/pasqal_analog.py`);
- **Xanadu GBS** — Gaussian Boson Sampling of the Takagi-decomposed, WAW-weighted
  complement graph (`contrail_env/xanadu_gbs.py`).

A PyQt6 dashboard builds scenarios, solves them **in-process** with a live
convergence curve, and runs the head-to-head benchmark (approximation ratio vs the
CP-SAT optimum with bootstrap CIs, raw feasibility rate, wall clock). It's a single
desktop process — no server, no broker.

The quantum pipelines need no quantum SDKs: each ships a dependency-free, physically
faithful backend (a split-operator state-vector simulator of the Rydberg Hamiltonian;
an exact Metropolis–Hastings sampler of the GBS distribution P(S) ∝ |Haf(B_S)|²).
Installing `pip install -e ".[quantum]"` switches them to Pulser's QuTiP emulator and
Strawberry Fields' gaussian backend automatically.

## Install & run

```
pip install -e ".[gui]"            # core + the PyQt6 desktop dashboard
python gui/app.py                  # launch the dashboard
```

Requires Python 3.11+. Core deps are just the solver runtime (numpy, OR-Tools); the
Qt/OpenGL dashboard stack lives in the `gui` extra, and the optional `[quantum]`
SDKs in `[quantum]`.

The dashboard has six tabs: live CP-SAT convergence (objective vs improvement), the
conflict-graph topology, QUBO matrix statistics (size, sparsity, penalty constants),
the chosen-option trade-offs, the quantum benchmark (CP-SAT vs Pasqal vs Xanadu over
N seeds, with live convergence curves for the BO loop and the GBS sampler), and a
geographic map — the ISSR risk as a marker overlay on a real Plotly `geo` basemap
(country borders / coastlines, drawn with SVG and bundled offline vectors, so it
needs no WebGL or network) with the chosen vs context routes animated on top.

The benchmark also runs headless:

```
python -m contrail_env.benchmark --flights 4 --seeds 5 --csv results.csv
```

## Reproducibility

Every sampler is seeded, so two benchmark runs with the same seeds produce
**byte-identical scientific columns** in the CSV — costs, approximation ratios,
feasibility and success rates, and the constraint-fingerprint means all match
exactly. The only columns that legitimately vary between runs are the three
timing-derived ones (`wall_clock_s`, `tts_sample_s`, `tts_total_s`): time-to-
solution is a wall-clock-derived quantity (TTS ∝ t_shot, Rønnow et al. 2014),
so it tracks machine load rather than the physics. This contract is enforced by
`tests/test_benchmark.py::test_determinism_science_columns`.

## ISSR field

The airspace's contrail zones (ice-supersaturated regions) are synthetic
Gaussian blobs — a controllable obstacle field for the optimizer to route
around. The field is consumed through a small `ISSRField` interface
(`rhi_excess`, `is_inside`, `mask_grid`), so the *source* of the field is
pluggable without touching `World` or the QUBO assembly.

## Layout

```
contrail_env/   synthetic environment, ISSR field, geo anchor, QUBO assembly,
                CP-SAT solver, quantum pipelines (pasqal_analog, xanadu_gbs,
                quantum_common, bayes_opt), the scenario builder (scenario.py),
                and the benchmark protocol (benchmark.py)
gui/            PyQt6 dashboard (builds + solves scenarios in-process)
tests/          environment build, CP-SAT vs brute-force, quantum solvers vs
                brute-force, benchmark round-trip, GUI map panel
```

## Development

```
pip install -e ".[dev]"
ruff check .
mypy contrail_env
pytest
```

CI runs the same checks on every push and pull request.

## Roadmap

- Run the Pasqal pipeline on real Pulser hardware: needs `[quantum]` extras plus a
  conflict graph that embeds as a valid unit-disk register (auto-detected; the
  built-in simulator is the fallback).
- Strawberry Fields X8 hardware demo on a reduced 8-mode subproblem.
- Larger instances for the Pasqal path (n > 20 qubits) via an MPS-style emulator tier.
