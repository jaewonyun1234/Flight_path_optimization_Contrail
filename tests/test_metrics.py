"""Per-shot statistics: success probability + time-to-solution (Rønnow 2014).

The closed forms are checked directly, and the multiplicity bookkeeping in
SampleEvaluation.repaired_unique is checked against known counts. A tiny
benchmark run confirms the new CSV columns exist and are consistent.
"""

import math

import numpy as np

from contrail_env.metrics import success_probability, time_to_solution
from contrail_env.quantum_common import (
    SampleEvaluation,
    build_option_graph,
    evaluate_samples,
)
from contrail_env.scenario import ScenarioConfig, build_scenario_full


def _eval_with(repaired_unique):
    """A SampleEvaluation carrying only the fields success_probability reads."""
    n = sum(m for _i, _c, m in repaired_unique)
    return SampleEvaluation(
        best_indices=[], best_cost=math.inf, feasibility_rate=0.0,
        best_raw_cost=None, n_samples=n, repaired_unique=repaired_unique,
    )


def test_tts_closed_form():
    assert time_to_solution(2.0, 1.0) == 2.0            # p=1 -> one shot
    assert math.isinf(time_to_solution(2.0, 0.0))       # p=0 -> never
    # p=0.5, q=0.99 -> t * ln(0.01)/ln(0.5) ~= 6.6439 t
    got = time_to_solution(1.0, 0.5, quantile=0.99)
    assert got == math.log(0.01) / math.log(0.5)
    assert abs(got - 6.6438561897747395) < 1e-6


def test_success_probability_counts():
    evaluation = _eval_with([((0, 2), 10.0, 3), ((1, 2), 12.0, 7)])
    # Only the 3 shots at cost 10.0 hit optimum_exact=10.0 -> p_s = 0.3.
    assert success_probability(evaluation, 10.0) == 0.3


def test_multiplicities_sum():
    cfg = ScenarioConfig(seed=2, n_flights=3, beta_contrail=5.0, time_limit_s=5.0)
    _w, _f, evals, conflicts, buckets = build_scenario_full(cfg)
    graph = build_option_graph(evals, conflicts, buckets)
    rng = np.random.default_rng(0)
    samples = (rng.random((100, graph.n)) < 0.4).astype(np.uint8)
    evaluation = evaluate_samples(graph, samples)
    assert sum(m for _i, _c, m in evaluation.repaired_unique) == 100


def test_benchmark_csv_columns(tmp_path):
    from contrail_env.benchmark import default_scenario_factory, run_benchmark

    report = run_benchmark(
        default_scenario_factory(n_flights=3), seeds=[1], n_shots=100, bo_iters=4,
    )
    csv_path = tmp_path / "b.csv"
    report.to_csv(str(csv_path))
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    for col in ("success_prob", "tts_sample_s", "tts_total_s"):
        assert col in header

    # sa rows with any success must show a finite sampling TTS.
    cols = header.split(",")
    i_solver = cols.index("solver")
    i_succ = cols.index("success_prob")
    i_tts = cols.index("tts_sample_s")
    for line in lines[1:]:
        parts = line.split(",")
        if parts[i_solver] == "sa" and float(parts[i_succ]) > 0.0:
            assert math.isfinite(float(parts[i_tts]))
