# Contrail-Aware Flight-Option Selection (Minimal QUBO Pipeline)

[![CI](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml/badge.svg)](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/actions/workflows/ci.yml)

Pick one altitude option per flight to minimize fuel + contrail cost, with
pairwise conflicts where two flights would seed the same ice-supersaturated
region (ISSR) at the same time. Small enough to read line-by-line; still
scientifically defensible (exact optimum, null baseline, feasibility metrics,
seeded reproducibility).

## The QUBO

$$H = \sum_i c_i x_i \;+\; A \sum_f \Big(\sum_k x_{f,k} - 1\Big)^2 \;+\; B \sum_{(i,j)\in\mathcal{C}} x_i x_j$$

- **cost** — $c_i$ = fuel proxy + $\alpha\,\cdot$ contrail exposure of option $i$
- **one-hot** — every flight picks exactly one altitude option
- **conflict** — conflicting option pairs $\mathcal{C}$ must not both be chosen

Penalties are auto-computed with $A, B > c_\max$ (2x safety), so breaking a
constraint never pays. The bound must exceed the largest option **cost**, not
the cost spread: skipping a flight entirely saves its whole option cost, so a
spread-sized penalty would make infeasible ground states optimal.

## Solvers reported

| solver | role |
|---|---|
| brute force (`exact.py`) | exact ground truth $E_\min$ |
| greedy (`exact.py`) | the repair heuristic run once from a blank slate — the honest classical bar |
| uniform random + repair (`exact.py`) | null baseline — structure-free floor |
| analog-QAOA + Bayesian optimization (`pasqal_analog.py`) | the algorithm under study |

**Raw energy is the primary metric.** The repair pass is itself a competent
greedy heuristic that solves easy instances from pure noise, so repaired cost
(and the approximation ratio built on it) measures the pipeline, not the
sampler. `raw_best_E` / `raw_mean_E` — QUBO energies of the *unrepaired*
bitstrings — are what the sampler itself achieves and cannot be gamed by
repair; the repaired-cost table is secondary.

## Install & run

```bash
pip install -e ".[dev]"          # numpy only at runtime
python run.py --flights 4 --options 3 --seeds 10 --shots 1000 --csv results.csv
```

The CSV has one row per (seed, solver): `n_vars`, `E_min`, `best_cost`,
`approx_ratio`, `feasibility_rate`, `wall_clock_s`; a mean ± std summary
prints at the end.

The unit-disk embeddability study (purely classical — how large can the
conflict graph grow before Pasqal's blockade geometry stops fitting?):

```bash
python -m contrail_env.embedding_study --csv embedding.csv
```

Embedding uses one pipeline everywhere (`embed()`): a greedy initial
placement, force-directed refinement with seeded annealing kicks, and
multi-start restarts with Gaussian jitter — deterministic given the seed.
Failures are attributed in the CSV: nodes of independence-degree ≥ 6 (2D
packing limit) or flight pairs conflicting at **every** level, whose prism
subgraph provably has no valid placement under the safety margins.

Optional: `pip install -e ".[quantum]"` adds the real Pulser/QuTiP emulator
backend (used automatically when the instance embeds and fits 12 qubits).

## Modules

```
contrail_env/problem.py          scenario generator (grid, ISSR blobs, costs, conflicts)
contrail_env/qubo.py             QUBO matrix assembly (cost + one-hot + conflict)
contrail_env/exact.py            brute force, repair, random baseline, metrics
contrail_env/pasqal_analog.py    analog schedule, statevector simulator, BO loop
contrail_env/bayes_opt.py        minimal GP + expected-improvement optimizer
contrail_env/embedding_study.py  greedy unit-disk embedder + scaling study
run.py                           end-to-end experiment script
```

The full multi-solver version (CP-SAT, Xanadu GBS, PyQt6 GUI, diagnostics)
lives on the [`archive/full-benchmark`](https://github.com/jaewonyun1234/Flight_path_optimization_Contrail/tree/archive/full-benchmark) branch.
