# Analysis: Conceptual Issue with R (Holding Vehicles) Calculation

## Executive Summary

**YES, this is a genuine conceptual bug, not a valid approximation.**

The current implementation calculates R1 as "all stopped vehicles at EoR", but this fundamentally misinterprets the paper's definition of "holding vehicles." The error is most severe during early cycles (cold start) and persists even in steady state.

---

## Paper's Definition

From the paper:
> "Holding vehicles are vehicles that, based on their projected trajectories using cruise speeds, **should have been discharged by that instant** but remain held by the system."

**Mathematical interpretation:**
- A vehicle is a "holding vehicle" at time t_EoR if:
  1. It is currently in the lane (position <= LANE_LEN)
  2. It entered at time t_entry where t_entry < (t_EoR - tau_ff)
  3. Had it traveled at free-flow speed, it would have exited by t_EoR

**Key insight:** Only vehicles that have been in the lane **longer than tau_ff** are holding vehicles.

---

## Current Implementation

### File: `/home/user/replicate_baseline_with_SUMO/estimate_dynamic_params_improved.py`

Lines 300-326:
```python
# R1: Vehicles currently stopped (observable in queue)
stopped_all = snap_df[stopped]
R1 = len(stopped_all)

# R2: Additional NCs that arrived after T_k1 and would have been discharged by EoR
effective_window = max(0.0, (eor_time - tau_ff) - T_k1)
R2 = q_nc * effective_window

# Total holding vehicles
R = int(R1 + R2)
```

**Problem:** R1 counts **ALL** stopped vehicles, regardless of how long they've been in the lane.

---

## The Bug Demonstrated

### Configuration
- Lane length l = 1000m
- Free-flow speed vf = 13.89 m/s
- Free-flow travel time tau_ff = l/vf = **72 seconds**
- Cycle = 60s, Red = 30s

### At First EoR (t = 30s)

**Correct calculation:**
- Cutoff time = t_EoR - tau_ff = 30 - 72 = **-42s**
- Vehicles that should have been discharged: those that entered before -42s
- Since simulation starts at t=0: **No vehicles entered before -42s**
- **TRUE R = 0** (or very close to 0)

**Current implementation:**
- R1 = count of all stopped vehicles
- Assuming q=0.4 veh/s: ~12 vehicles arrived during [0, 30s]
- Most/all are stopped due to red signal
- **CALCULATED R1 ≈ 12**
- R2 ≈ 0 (since effective_window = max(0, -42 - T_k1) ≈ 0)
- **CALCULATED R ≈ 12**

**Error:** CALCULATED R / TRUE R = 12 / 0 → **Infinite overcounting!**

---

## Why R2 Doesn't Fix This

R2 is defined as:
```
R2 = q_nc * effective_window
where effective_window = max(0, (t_EoR - tau_ff) - T_k1)
```

At first EoR:
- effective_window = max(0, (30 - 72) - T_k1) = max(0, -42 - T_k1)
- Since T_k1 >= 0 (vehicles enter after simulation starts)
- effective_window ≈ 0 or negative
- **R2 ≈ 0**

Therefore, R2 does **not** compensate for the overcounting in R1.

---

## When Does the Error Occur?

### Early Cycles (Cold Start)
The error is **most severe** during the first cycles:
- At t=30s: TRUE R ≈ 0, CALCULATED R ≈ 12 (infinite error)
- At t=90s: TRUE R would be vehicles that entered before 18s, still significant undercounting

### Steady State
Even after t > tau_ff, there's **systematic overcounting**:
- Not all stopped vehicles have been in the lane longer than tau_ff
- Vehicles that entered recently (within tau_ff before EoR) are incorrectly counted as holding vehicles
- The approximation becomes "closer" but never exact

---

## Why This Matters

### Impact on Queue Estimation (Q)
The paper's queue estimation formula relies on R:
```
Q = q_NC * tau_ff + R_NC
```

If R is overcounted:
- R_NC is overcounted
- Q is overcounted
- Location estimation for non-CV vehicles becomes biased

### Cascading Errors
- MLE estimation of q and p may be affected if R feeds back into the estimation
- Any control or traffic management decisions based on Q will be suboptimal

---

## The Correct Approach

### What's Needed
To properly calculate R according to the paper's definition:

1. **Track vehicle entry times:**
   ```python
   # For each vehicle, record when it first appeared in the lane
   entry_times = df_lane.groupby("id")["time"].min()
   ```

2. **At each EoR, filter by time in lane:**
   ```python
   tau_ff = LANE_LEN / V_FREE
   cutoff_time = eor_time - tau_ff

   holding_vehicles = []
   for vehicle in snap_df:
       vid = vehicle["id"]
       t_entry = entry_times[vid]

       # Only count if vehicle entered before cutoff
       if t_entry < cutoff_time:
           holding_vehicles.append(vid)

   R = len(holding_vehicles)
   ```

3. **Split by CV/NC for R_CV and R_NC:**
   ```python
   R_CV = count of holding_vehicles that are CVs
   R_NC = count of holding_vehicles that are NCs
   ```

### Why This Wasn't Done
Looking at the codebase:
- The FCD data **does** contain sufficient information (timesteps per vehicle)
- File `/home/user/replicate_baseline_with_SUMO/estimate_Q_from_CV_only.py` already tracks entry times using `first_seen`
- This suggests the capability exists but wasn't applied to R calculation

---

## Evidence from Existing Code

### Entry Time Tracking is Already Used

File: `/home/user/replicate_baseline_with_SUMO/estimate_Q_from_CV_only.py`, lines 59-66:
```python
def detect_cv_entries(df_lane: pd.DataFrame, horizon_s: float, t_center: float) -> int:
    """Count CVs that ENTERED the lane during (t_center - horizon_s, t_center]."""
    iscv_all = is_cv(df_lane["type"], df_lane["id"])
    df_cv = df_lane[iscv_all].copy()
    first_seen = df_cv.groupby("id", as_index=False)["time"].min()
    return int(((first_seen["time"] > t_center - horizon_s) &
                (first_seen["time"] <= t_center)).sum())
```

**This proves that:**
1. FCD data contains entry time information
2. The codebase already has logic to track when vehicles enter
3. A correct implementation of R is technically feasible

---

## Potential Defenses (and Why They Don't Hold)

### "Maybe it's a steady-state approximation?"
- **Counter:** The paper's definition is precise and cycle-by-cycle
- No mention of steady-state assumptions in the paper or code
- The error persists even after steady state is reached

### "Maybe R1 + R2 balances out somehow?"
- **Counter:** R2 ≈ 0 in early cycles when the error is worst
- R2 is meant to capture additional NCs, not compensate for R1's overcounting
- The math doesn't support this interpretation

### "Maybe stopped vehicles are close enough to holding vehicles?"
- **Counter:** At first EoR: TRUE R = 0, R1 = 12 (1200% error)
- This isn't a "close approximation" - it's fundamentally wrong
- The paper's definition is clear and implementable

---

## Conclusion

### Summary of Findings

1. **Confirmed Bug:** R1 (all stopped vehicles) is NOT a valid approximation for holding vehicles as defined by the paper

2. **Root Cause:** Missing entry time tracking in the R calculation logic

3. **Severity:**
   - Infinite error during cold start
   - Systematic overcounting even in steady state
   - Cascades into Q estimation errors

4. **Fixability:** HIGH - FCD data contains necessary information, and similar logic already exists elsewhere in the codebase

### Recommendations

1. **Immediate:** Document this as a known issue
2. **Short-term:** Implement proper entry time tracking for R calculation
3. **Long-term:** Re-run all baseline experiments with corrected R calculation
4. **Validation:** Compare corrected R values against ground truth from SUMO

---

## References

- Implementation: `/home/user/replicate_baseline_with_SUMO/estimate_dynamic_params_improved.py`, lines 267-338
- Alternative calculation: `/home/user/replicate_baseline_with_SUMO/recompute_R_breakdown_consistent.py`, lines 101-119
- Entry time tracking example: `/home/user/replicate_baseline_with_SUMO/estimate_Q_from_CV_only.py`, lines 59-66

---

**Analysis performed:** 2025-10-30
**Status:** CONFIRMED CONCEPTUAL BUG
