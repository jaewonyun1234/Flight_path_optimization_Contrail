"""
contrail_env — Minimal contrail-aware flight-option QUBO pipeline.

One algorithm under study (Pasqal-style analog-QAOA + Bayesian
optimization), benchmarked against brute-force ground truth and a
uniform-random baseline. See problem.py for the model in plain language.
"""

from .bayes_opt import BOResult, gp_minimize
from .embedding_study import (
    EmbeddingReport,
    check_embedding,
    greedy_embedding,
    independence_edges,
    run_embedding_study,
)
from .exact import (
    SolveResult,
    approximation_ratio,
    brute_force_optimum,
    evaluate_samples,
    mean_random_cost,
    repair,
    solve_random,
)
from .pasqal_analog import (
    AnnealSchedule,
    BackendBudgetError,
    EmbeddingError,
    RydbergStatevector,
    node_weights,
    pulser_available,
    solve_pasqal_analog,
)
from .problem import Scenario, make_scenario
from .qubo import QUBOInstance, assemble_qubo, cost_of_assignment, is_feasible

__all__ = [
    "AnnealSchedule", "BOResult", "BackendBudgetError", "EmbeddingError",
    "EmbeddingReport", "QUBOInstance", "RydbergStatevector", "Scenario",
    "SolveResult", "approximation_ratio", "assemble_qubo", "brute_force_optimum",
    "check_embedding", "cost_of_assignment", "evaluate_samples", "gp_minimize",
    "greedy_embedding", "independence_edges", "is_feasible", "make_scenario",
    "mean_random_cost", "node_weights", "pulser_available", "repair",
    "run_embedding_study", "solve_pasqal_analog", "solve_random",
]
