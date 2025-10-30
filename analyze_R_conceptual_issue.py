"""
Analysis of the conceptual issue with R (holding vehicles) calculation.

This script demonstrates why R1 (all stopped vehicles) is NOT a valid
approximation for "holding vehicles" according to the paper's definition.
"""

# Parameters from baseline configuration
LANE_LEN = 1000.0  # m
V_FREE = 13.89     # m/s
CYCLE = 60.0       # s
RED = 30.0         # s

# Calculate free-flow travel time
tau_ff = LANE_LEN / V_FREE
print("="*70)
print("CONCEPTUAL ANALYSIS: R (Holding Vehicles) Calculation")
print("="*70)
print(f"\nParameters:")
print(f"  Lane length (l): {LANE_LEN} m")
print(f"  Free-flow speed (vf): {V_FREE} m/s")
print(f"  Free-flow travel time (tau_ff): {tau_ff:.2f} s")
print(f"  Cycle length: {CYCLE} s")
print(f"  Red duration: {RED} s")

print(f"\n{'='*70}")
print("PAPER'S DEFINITION OF HOLDING VEHICLES:")
print("="*70)
print('"Holding vehicles are vehicles that, based on their projected')
print('trajectories using cruise speeds, SHOULD HAVE BEEN DISCHARGED')
print('by that instant but remain held by the system."')
print("\nMathematical interpretation:")
print("  R = count of vehicles that:")
print("    1. Are currently in the lane at EoR")
print("    2. Entered before (t_EoR - tau_ff)")
print("    3. Would have exited by t_EoR if traveling at free-flow speed")

print(f"\n{'='*70}")
print("ANALYSIS AT FIRST EoR (t = 30s):")
print("="*70)

t_EoR = RED  # First EoR at 30s
cutoff_time = t_EoR - tau_ff

print(f"\nAt first EoR:")
print(f"  t_EoR = {t_EoR} s")
print(f"  tau_ff = {tau_ff:.2f} s")
print(f"  Cutoff time = t_EoR - tau_ff = {t_EoR} - {tau_ff:.2f} = {cutoff_time:.2f} s")

print(f"\nVehicles that SHOULD have been discharged:")
print(f"  → Those that entered BEFORE {cutoff_time:.2f}s")
print(f"  → Since simulation starts at t=0, NO vehicles entered before {cutoff_time:.2f}s")
print(f"  → Therefore, TRUE R = 0 (or very close to 0)")

print(f"\nWhat current implementation calculates:")
print(f"  R1 = count of ALL stopped vehicles at EoR")
print(f"  → This includes the ENTIRE queue that formed during red phase (0-30s)")
print(f"  → Typical queue after 30s red: ~12-15 vehicles (assuming q=0.4 veh/s)")
print(f"  → So R1 ≈ 12-15 >> 0")

print(f"\n{'='*70}")
print("CONCLUSION: R1 IS OVERCOUNTING!")
print("="*70)
print("\nNot all stopped vehicles are 'holding vehicles'.")
print("Most stopped vehicles at first EoR entered AFTER the cutoff time")
print(f"(i.e., after {cutoff_time:.2f}s) and therefore haven't been in the lane")
print("long enough to have 'should have been discharged'.")

print(f"\n{'='*70}")
print("WHAT ABOUT R2?")
print("="*70)
print("\nR2 attempts to estimate additional NCs that:")
print("  - Arrived after T_k1 (when last stopped CV started stopping)")
print("  - Would have been discharged by EoR")
print(f"\nR2 = q_nc * effective_window")
print(f"where effective_window = max(0, (t_EoR - tau_ff) - T_k1)")

print(f"\nAt first EoR (t=30s):")
print(f"  effective_window = max(0, ({t_EoR} - {tau_ff:.2f}) - T_k1)")
print(f"                   = max(0, {cutoff_time:.2f} - T_k1)")
print(f"\nSince T_k1 ≥ 0 (vehicles enter after t=0):")
print(f"  effective_window = max(0, {cutoff_time:.2f} - T_k1) ≈ 0 (or negative)")
print(f"  Therefore, R2 ≈ 0")

print(f"\nSo R2 doesn't fix the overcounting in R1!")

print(f"\n{'='*70}")
print("NUMERIC EXAMPLE:")
print("="*70)

# Example scenario
q = 0.4  # veh/s arrival rate
print(f"\nAssume q = {q} veh/s, uniform arrivals")
print(f"\nVehicles in lane at first EoR (t=30s):")

# Vehicles that entered during [0, 30s]
expected_arrivals = q * RED
print(f"  Expected arrivals during [0, 30s]: {expected_arrivals:.1f} vehicles")

# How many should have exited?
# Only those that entered before (30 - 72) = -42s → NONE!
print(f"\nVehicles that should have exited by t=30s:")
print(f"  → Entered before t = {cutoff_time:.2f}s")
print(f"  → Count: 0 (no arrivals before t=0)")

print(f"\nVehicles that are 'holding vehicles' (TRUE R):")
print(f"  → TRUE R = 0")

print(f"\nWhat implementation calculates:")
print(f"  → R1 (all stopped) ≈ {expected_arrivals:.1f}")
print(f"  → R2 ≈ 0 (effective_window ≈ 0)")
print(f"  → CALCULATED R ≈ {expected_arrivals:.1f}")

print(f"\nERROR: CALCULATED R / TRUE R = {expected_arrivals:.1f} / 0 → INFINITE!")

print(f"\n{'='*70}")
print("WHEN DOES THIS APPROXIMATION BECOME VALID?")
print("="*70)

print(f"\nThe approximation R1 ≈ holding vehicles only becomes reasonable")
print(f"after t > tau_ff = {tau_ff:.2f}s")
print(f"\nAt second EoR (t=90s):")
print(f"  Cutoff: {90 - tau_ff:.2f}s")
print(f"  Vehicles that entered before {90 - tau_ff:.2f}s would have exited")
print(f"  This is closer to steady-state, but still not exact!")

print(f"\n{'='*70}")
print("THE CORRECT APPROACH:")
print("="*70)
print("\nTo properly calculate R, we need to:")
print("  1. Track when each vehicle entered the lane (t_entry)")
print("  2. For each vehicle at EoR:")
print("     - Calculate time_in_lane = t_EoR - t_entry")
print("     - If time_in_lane > tau_ff: it's a holding vehicle")
print("  3. Count only these vehicles")
print("\nThis requires tracking vehicle entry times, which is NOT")
print("currently done in the implementation!")

print(f"\n{'='*70}")
print("SUMMARY:")
print("="*70)
print("\n✗ Current implementation overcounts R, especially in early cycles")
print("✗ R1 (all stopped vehicles) ≠ holding vehicles")
print("✗ R2 doesn't compensate for R1's overcounting")
print("✗ This is a CONCEPTUAL BUG, not just an approximation error")
print("\n✓ The paper's definition is clear and precise")
print("✓ A correct implementation would track vehicle entry times")
print("✓ The error is largest in early cycles (cold start)")
print("✓ Even in steady-state, there's systematic overcounting")

print("\n" + "="*70)
