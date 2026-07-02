"""
contrail_env — Synthetic environment for contrail-aware flight option selection.

This is the FOUNDATION of the quantum-optimization project. It produces
all the structures the quantum solvers (Pasqal, Xanadu) and the classical
verifier (CP-SAT) consume:

    - A planning region (AirspaceGrid)
    - Ice-supersaturated regions (ISSRField)
    - ATC sectors with capacities (SectorMap, Sector)
    - Aircraft performance (Aircraft, LinearJetPerformance)
    - Flights with baseline + alternative altitude profiles (Flight, AltitudeProfile)
    - Scored options (EvaluatedOption)
    - The conflict graph and QUBO matrix (QUBOInstance)

USAGE PATTERN
=============

    from contrail_env import (
        default_european_world,
        build_random_flights, build_and_evaluate_flight,
        build_conflict_graph, build_capacity_buckets,
        assemble_qubo, brute_force_optimum,
    )

    # 1. Build the world (airspace + ISSRs + sectors)
    world = default_european_world(seed=42)

    # 2. Build flights and their options
    flights = build_random_flights(n_flights=4, world=world, seed=42)
    all_evals = []
    for f in flights:
        all_evals.extend(build_and_evaluate_flight(f, world))

    # 3. Construct constraint structures
    conflicts = build_conflict_graph(all_evals, world)
    buckets   = build_capacity_buckets(all_evals, world)

    # 4. Assemble the QUBO matrix
    qubo = assemble_qubo(all_evals, conflicts, buckets)

    # 5. Hand qubo.Q off to your solver (CP-SAT / Pasqal / Xanadu)
    #    or brute-force for verification on small instances:
    z_opt, cost_opt, energy_opt = brute_force_optimum(qubo, all_evals)
"""

# Units
from .units import (
    M_PER_FT, FT_PER_M, M_PER_NM, NM_PER_KM,
    MS_PER_KT, KT_PER_MS, KM_PER_DEG_LAT,
    fl_to_ft, ft_to_fl, fl_to_m, m_to_fl,
    isa_temperature, speed_of_sound_ms, mach_to_ms, ms_to_mach,
)

# Synthetic ISSRs
from .synthetic_issr import (
    ISSRBlob, ISSRField,
    random_issr_field, hand_placed_issr_field,
    issr_area_coverage,
)

# Airspace
from .airspace import (
    AirspaceGrid, Sector, SectorMap,
    uniform_sector_grid,
    voxelize_trajectory, densify_waypoints,
)

# World
from .world import World, default_european_world

# Aircraft
from .aircraft import (
    AircraftPerformance, LinearJetPerformance, Aircraft, a320_like,
)

# Flight model
from .flight import (
    AltitudeSegment, AltitudeProfile, Flight,
    available_fls,
    build_baseline_profile, replace_segment_fl, shift_all_segments,
    waypoints_for, evaluate_option, EvaluatedOption,
)

# Options
from .options import (
    OptionKind, OptionSpec,
    enumerate_option_specs,
    build_and_evaluate_flight, build_random_flights,
)

# QUBO
from .qubo import (
    ConflictEdge, CapacityBucket, QUBOInstance,
    build_conflict_graph, build_capacity_buckets, assemble_qubo,
    is_feasible, cost_of_assignment, brute_force_optimum,
)

# CP-SAT classical ground-truth solver
from .solver_cpsat import (
    CPSATResult, solve_cpsat, enumerate_optimum,
)

# Quantum solver core (shared by Pasqal + Xanadu pipelines)
from .quantum_common import (
    BackendBudgetError, OptionGraph, QuantumResult,
    build_option_graph, repair_sample, sample_is_feasible,
    penalized_energy, evaluate_samples,
)

# Pasqal neutral-atom analog pipeline
from .pasqal_analog import (
    AnnealSchedule, RydbergStatevector,
    solve_pasqal_analog, pulser_available,
)

# Xanadu photonic GBS pipeline
from .xanadu_gbs import (
    GBSEncoding, GBSSubsetSampler,
    encode_option_graph, hafnian, takagi_symmetric,
    solve_xanadu_gbs, strawberryfields_available,
)

# Classical sampling baselines (null control + simulated annealing)
from .classical_baselines import (
    solve_random_repair, solve_simulated_annealing,
)

# Per-shot benchmark statistics (success probability + time-to-solution)
from .metrics import success_probability, time_to_solution

# Instantaneous spectrum of the analog sweep (adiabaticity diagnostic)
from .spectral import (
    AnalogSpectrum, hamiltonian_matvec, instantaneous_spectrum, lowest_eigenpairs,
)

# Benchmark protocol (CP-SAT vs Pasqal vs Xanadu)
from .benchmark import (
    BenchmarkReport, InstanceResult, SolverRun, SolverStats,
    run_benchmark, default_scenario_factory, bootstrap_ci,
)

__all__ = [
    # Units
    "M_PER_FT", "FT_PER_M", "M_PER_NM", "NM_PER_KM",
    "MS_PER_KT", "KT_PER_MS", "KM_PER_DEG_LAT",
    "fl_to_ft", "ft_to_fl", "fl_to_m", "m_to_fl",
    "isa_temperature", "speed_of_sound_ms", "mach_to_ms", "ms_to_mach",
    # ISSRs
    "ISSRBlob", "ISSRField",
    "random_issr_field", "hand_placed_issr_field",
    "issr_area_coverage",
    # Airspace
    "AirspaceGrid", "Sector", "SectorMap",
    "uniform_sector_grid",
    "voxelize_trajectory", "densify_waypoints",
    # World
    "World", "default_european_world",
    # Aircraft
    "AircraftPerformance", "LinearJetPerformance", "Aircraft", "a320_like",
    # Flight
    "AltitudeSegment", "AltitudeProfile", "Flight",
    "available_fls",
    "build_baseline_profile", "replace_segment_fl", "shift_all_segments",
    "waypoints_for", "evaluate_option", "EvaluatedOption",
    # Options
    "OptionKind", "OptionSpec",
    "enumerate_option_specs",
    "build_and_evaluate_flight", "build_random_flights",
    # QUBO
    "ConflictEdge", "CapacityBucket", "QUBOInstance",
    "build_conflict_graph", "build_capacity_buckets", "assemble_qubo",
    "is_feasible", "cost_of_assignment", "brute_force_optimum",
    # CP-SAT solver
    "CPSATResult", "solve_cpsat", "enumerate_optimum",
    # Quantum core
    "BackendBudgetError", "OptionGraph", "QuantumResult",
    "build_option_graph", "repair_sample", "sample_is_feasible",
    "penalized_energy", "evaluate_samples",
    # Pasqal
    "AnnealSchedule", "RydbergStatevector",
    "solve_pasqal_analog", "pulser_available",
    # Xanadu
    "GBSEncoding", "GBSSubsetSampler",
    "encode_option_graph", "hafnian", "takagi_symmetric",
    "solve_xanadu_gbs", "strawberryfields_available",
    # Classical baselines
    "solve_random_repair", "solve_simulated_annealing",
    # Per-shot metrics
    "success_probability", "time_to_solution",
    # Spectral diagnostics
    "AnalogSpectrum", "hamiltonian_matvec", "instantaneous_spectrum", "lowest_eigenpairs",
    # Benchmark
    "BenchmarkReport", "InstanceResult", "SolverRun", "SolverStats",
    "run_benchmark", "default_scenario_factory", "bootstrap_ci",
]
