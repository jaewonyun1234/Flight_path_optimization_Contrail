"""contrail_env.scenario: deterministic scenario build + in-process CP-SAT solve.

Replaces the old gRPC round-trip test — the solve now runs in-process, so we
test the builder + solver directly with no network layer.
"""

from contrail_env.scenario import ScenarioConfig, build_scenario_full, solve_scenario


def test_build_scenario_is_deterministic():
    cfg = ScenarioConfig(seed=7, n_flights=3)
    _w1, _f1, e1, c1, b1 = build_scenario_full(cfg)
    _w2, _f2, e2, c2, b2 = build_scenario_full(cfg)
    assert [ev.flight_name for ev in e1] == [ev.flight_name for ev in e2]
    assert len(c1) == len(c2)
    assert len(b1) == len(b2)


def test_solve_scenario_returns_one_choice_per_flight():
    cfg = ScenarioConfig(seed=3, n_flights=3, time_limit_s=5.0)
    seen = []
    result = solve_scenario(cfg, on_progress=lambda i, o: seen.append((i, o)))
    assert len(result.choices) == 3
    assert len({c.flight_name for c in result.choices}) == 3
    assert isinstance(result.objective, float)
    assert result.n_options_total >= 3
