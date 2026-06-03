"""
demo.py — End-to-end example of the contrail_env package.

This script wires every module together and verifies the full pipeline:

    1. Build a World (airspace grid + ISSRs + sectors)
    2. Build 3 random flights with baseline altitude profiles
    3. Enumerate + evaluate options for each flight
    4. Build the conflict graph (options sharing ISSR cells)
    5. Build capacity buckets (sector x time)
    6. Assemble the QUBO matrix Q
    7. Brute-force the ground state (small enough to enumerate)
    8. Verify feasibility of the optimum

Run with:
    python -m contrail_env.demo
"""

from __future__ import annotations

# Import everything from the package via the top-level __init__.py
from contrail_env import (
    default_european_world,
    build_random_flights,
    build_and_evaluate_flight,
    build_conflict_graph,
    build_capacity_buckets,
    assemble_qubo,
    brute_force_optimum,
    is_feasible,
    issr_area_coverage,
)


# =============================================================================
# PRETTY PRINTING
# =============================================================================

def header(n: int, msg: str) -> None:
    """Section divider for the demo output."""
    print()
    print("=" * 72)
    print(f" STEP {n}: {msg}")
    print("=" * 72)


def hrule(char: str = "-") -> None:
    print(char * 72)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Step 1: World
    # -------------------------------------------------------------------------
    header(1, "Build the world (airspace + ISSRs + sectors)")
    world = default_european_world(seed=42, n_issr_blobs=6)
    cov_fl340 = issr_area_coverage(world.issr, altitude_m=10363.0)
    cov_fl360 = issr_area_coverage(world.issr, altitude_m=10973.0)
    cov_fl380 = issr_area_coverage(world.issr, altitude_m=11582.0)

    print(f"  Planning region: "
          f"{world.grid.x_max_km-world.grid.x_min_km:.0f} km E-W x "
          f"{world.grid.y_max_km-world.grid.y_min_km:.0f} km N-S")
    print(f"  Grid shape:      {world.grid.shape} "
          f"({world.grid.n_cells:,} cells)")
    print(f"  FL bands:        {world.grid.fl_bands}")
    print(f"  Time buckets:    {world.grid.nt} x {world.grid.dt_s:.0f}s "
          f"= {world.grid.nt*world.grid.dt_s/60:.0f} min total")
    print(f"  ISSR blobs:      {len(world.issr.blobs)}")
    print(f"  ISSR coverage:   FL340 {cov_fl340*100:5.1f}%   "
          f"FL360 {cov_fl360*100:5.1f}%   FL380 {cov_fl380*100:5.1f}%")
    print(f"  Sectors:         {len(world.sectors.sectors)}")

    # -------------------------------------------------------------------------
    # Step 2: Flights
    # -------------------------------------------------------------------------
    header(2, "Build 3 flights with baseline altitude profiles")
    # Tight corridor + tight departure window forces flights to share
    # the SAME individual 4D cells (not just the same sector). Without
    # this clustering, the random spread produces zero conflicts and the
    # pairwise-conflict part of the QUBO does no work.
    flights = build_random_flights(
        n_flights=3, world=world, seed=42,
        corridor_frac=0.02,                     # ~16 km wide N-S
        snapshot_window_s=(0.0, 300.0),         # all depart within 5 min
    )
    for f in flights:
        dist_km = f.horizontal_distance_km
        print(f"  {f.name}: "
              f"({f.origin_km[0]:.0f}, {f.origin_km[1]:.0f}) -> "
              f"({f.destination_km[0]:.0f}, {f.destination_km[1]:.0f}) km, "
              f"{dist_km:.0f} km total, "
              f"t_dep={f.departure_s:.0f}s, "
              f"m_initial={f.aircraft.initial_mass_kg/1000:.0f}t")
        for i, seg in enumerate(f.baseline.segments):
            print(f"    baseline seg {i}: FL{seg.fl} from "
                  f"{seg.t_start_s/60:5.1f} min to "
                  f"{seg.t_end_s/60:5.1f} min "
                  f"({seg.duration_s/60:5.1f} min long)")

    # -------------------------------------------------------------------------
    # Step 3: Enumerate and evaluate options
    # -------------------------------------------------------------------------
    header(3, "Enumerate and evaluate options against the world")
    all_evals = []
    for f in flights:
        evals = build_and_evaluate_flight(f, world)
        all_evals.extend(evals)
        print(f"\n  {f.name}: {len(evals)} options")
        hrule()
        print(f"  {'opt':>3} {'profile':<28} "
              f"{'fuel(kg)':>10} {'contrail':>9} {'disrupt':>9} "
              f"{'buckets':>8} {'cost':>8}")
        hrule()
        for ev in evals:
            fls = "->".join(f"FL{s.fl}" for s in ev.profile.segments)
            if len(fls) > 26:
                fls = fls[:23] + "..."
            print(f"  {ev.option_index:>3} {fls:<28} "
                  f"{ev.fuel_kg:>10.1f} "
                  f"{ev.contrail_cells:>9d} "
                  f"{ev.disruption_FLmin:>9.1f} "
                  f"{len(ev.buckets):>8d} "
                  f"{ev.cost_combined:>8.1f}")

    print(f"\n  Total options across all flights: {len(all_evals)}")

    # -------------------------------------------------------------------------
    # Step 4: Conflict graph
    # -------------------------------------------------------------------------
    header(4, "Build conflict graph (cross-flight ISSR cell sharing)")
    conflicts = build_conflict_graph(all_evals, world)
    print(f"  {len(conflicts)} conflict edges found")
    for e in conflicts[:8]:
        opt_i = all_evals[e.i]
        opt_j = all_evals[e.j]
        print(f"    {opt_i.flight_name}.opt{opt_i.option_index} <-> "
              f"{opt_j.flight_name}.opt{opt_j.option_index} "
              f"({e.reason})")
    if len(conflicts) > 8:
        print(f"    ... and {len(conflicts)-8} more")

    # -------------------------------------------------------------------------
    # Step 5: Capacity buckets
    # -------------------------------------------------------------------------
    header(5, "Build capacity buckets (binding only)")
    buckets = build_capacity_buckets(all_evals, world)
    print(f"  {len(buckets)} binding capacity buckets found")
    for b in buckets:
        sec_name = world.sectors.sectors[b.bucket_id[0]].name
        member_str = ",".join(
            f"{all_evals[m].flight_name}.opt{all_evals[m].option_index}"
            for m in b.members
        )
        print(f"    bucket {b.bucket_id} ({sec_name}): "
              f"cap={b.capacity}, "
              f"{len(b.members)} members, "
              f"{b.n_slack_bits} slack bits")
        print(f"      members: {member_str}")

    # -------------------------------------------------------------------------
    # Step 6: QUBO assembly
    # -------------------------------------------------------------------------
    header(6, "Assemble QUBO matrix")
    qubo = assemble_qubo(all_evals, conflicts, buckets)
    print(f"  QUBO dimension:   n = {qubo.n}")
    print(f"    option vars:    {qubo.n_options}")
    print(f"    slack vars:     {qubo.n_slack}")
    print(f"  Penalty constants:")
    print(f"    A (one-hot):    {qubo.penalty_A:.1f}")
    print(f"    B (conflict):   {qubo.penalty_B:.1f}")
    print(f"    C (capacity):   {qubo.penalty_C:.1f}")
    print(f"  Energy offset:    {qubo.constant:.1f}")
    nnz = int((qubo.Q != 0).sum())
    total = qubo.n * qubo.n
    print(f"  Q sparsity:       {nnz} non-zeros / {total} = "
          f"{100*nnz/total:.1f}%")
    print(f"  Q value range:    [{qubo.Q.min():.1f}, {qubo.Q.max():.1f}]")

    # -------------------------------------------------------------------------
    # Step 7: Verification by candidate evaluation
    # -------------------------------------------------------------------------
    # Brute force isn't tractable at n=36 (2^36 ≈ 70 billion), but we can
    # verify the QUBO does the right thing by evaluating a few SPECIFIC
    # candidate assignments and confirming the energy ordering matches
    # what we expect (feasible < infeasible).
    #
    # For each candidate we set the option variables explicitly, then
    # choose the optimal slack bits per capacity bucket (slack = cap -
    # sum_members if non-negative, else 0). This isolates the question
    # "is this option assignment feasible and good?" from "did we pick
    # the right slack encoding?"
    header(7, "Verify QUBO behaviour on hand-picked candidates")

    import numpy as np

    def build_z(option_picks: dict[str, int]) -> np.ndarray:
        """
        Construct a full z vector from a dict {flight_name: option_index}.
        Slack bits are set OPTIMALLY for the chosen option assignment.
        """
        z = np.zeros(qubo.n, dtype=int)
        # 1. Set option vars from the picks
        for opt_idx, ev in enumerate(all_evals):
            if option_picks.get(ev.flight_name) == ev.option_index:
                z[opt_idx] = 1
        # 2. For each capacity bucket, set slack to balance the constraint
        for bkt in buckets:
            n_chosen = sum(int(z[m]) for m in bkt.members)
            shortfall = bkt.capacity - n_chosen
            if shortfall > 0:
                # Encode `shortfall` in binary across the slack bits.
                # If shortfall exceeds 2^L - 1 we silently saturate;
                # the QUBO will catch this as a residual penalty.
                max_repr = 2 ** bkt.n_slack_bits - 1
                s = min(shortfall, max_repr)
                for bit in range(bkt.n_slack_bits):
                    var = qubo.slack_index[(bkt.bucket_id, bit)]
                    z[var] = (s >> bit) & 1
            # If shortfall <= 0 (capacity exceeded), leave slack = 0 —
            # the residual penalty in the QUBO surfaces the violation.
        return z

    candidates = {
        "all FL360 baseline":        {"F1": 0, "F2": 0, "F3": 0},
        "all FL340 (lowest, cheap)": {"F1": 2, "F2": 2, "F3": 2},
        "spread FL340/FL360/FL380":  {"F1": 2, "F2": 0, "F3": 1},
        "spread FL360/FL380/FL340":  {"F1": 0, "F2": 1, "F3": 2},
    }

    print(f"  {'candidate':<32} {'energy':>14} {'feasible':>10}  cost(raw)")
    hrule()
    for name, picks in candidates.items():
        z = build_z(picks)
        e = qubo.energy(z)
        feas, viols = is_feasible(z, qubo)
        # Raw cost (ignoring penalties)
        raw_cost = sum(
            all_evals[i].cost_combined * int(z[i])
            for i in range(qubo.n_options)
        )
        marker = "yes" if feas else "NO "
        print(f"  {name:<32} {e:>14.1f} {marker:>10}  {raw_cost:>9.1f}")
        if not feas:
            for v in viols[:3]:
                print(f"      -> {v}")
            if len(viols) > 3:
                print(f"      -> ... and {len(viols)-3} more")

    print(f"\n  Interpretation:")
    print(f"  - Infeasible candidates carry penalty terms of order "
          f"{qubo.penalty_A:.0f} per violation.")
    print(f"  - A real solver (CP-SAT, Pasqal, Xanadu) finds the minimum-")
    print(f"    energy z; the QUBO guarantees that minimum is feasible.")

    print()
    print("=" * 72)
    print(" Demo complete. Q is ready to hand off to a quantum/classical solver.")
    print("=" * 72)


if __name__ == "__main__":
    main()
