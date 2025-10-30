"""
Compute NC distribution in Type B segments (moving traffic)
Implements Proposition 2, Part B from the paper (Equations 5-11)
"""

import pandas as pd
import numpy as np
from math import floor
from pathlib import Path

# Input files
SEGMENTS_FILE = "segments_AB_resegmented_v05_entr.csv"
Q_FILE = "eor_Q_estimated.csv"
A_COUNTS_FILE = "A_counts_ROUND.csv"

# Output files
OUT_COUNTS = "B_counts.csv"
OUT_POSITIONS = "B_positions.csv"

# Parameters
L_E = 6.44  # average effective vehicle length (m)
DELTA_T = 2.04  # minimum safe time headway (s) - from paper Section 5.1
LANE_LEN = 1000.0  # stopline coordinate (m)

def round_half_up(x: float) -> int:
    """Round to nearest, halves up (0.5 -> 1), matching the paper's ⌊·⌉ symbol."""
    return int(floor(x + 0.5))

def compute_Q_tilde_B_i(Li, Li_plus_1, Vi, Vi_plus_1, is_boundary):
    """
    Compute maximum number of NCs that can be inserted between Li and Li+1.
    Implements Eq. (7) from the paper.

    Args:
        Li: downstream position (m)
        Li_plus_1: upstream position (m)
        Vi: downstream speed (m/s)
        Vi_plus_1: upstream speed (m/s)
        is_boundary: True if i=0 or i=m (can insert at stop bar or entrance)

    Returns:
        Q_tilde_B_i: maximum number of NCs
    """
    if Vi == 0 and Vi_plus_1 == 0:
        return 0

    numerator = 2 * (Li - Li_plus_1) - 2 * DELTA_T * Vi_plus_1
    denominator = DELTA_T * (Vi + Vi_plus_1)

    if denominator <= 0:
        return 0

    k = numerator / denominator

    if is_boundary:
        # i=0 or i=m: can insert at boundary
        Q_tilde = round_half_up(k + 1)
    else:
        # i ∈ (0,m): enclosed by two CVs
        Q_tilde = round_half_up(k) - 1

    return max(0, Q_tilde)

def compute_bounds_and_speeds(Li, Li_plus_1, Vi, Vi_plus_1, Q_B_i, is_first, is_last):
    """
    Compute feasible space bounds and speeds for NC insertion.
    Implements Eqs. (9)-(11) from the paper.

    Returns:
        L_lower, L_upper, speeds (list of speeds for each NC)
    """
    if Q_B_i == 0:
        return None, None, []

    # Lower bound L_i^(l) - Eq. (9)
    if is_last:  # i = m (upstream entrance)
        L_lower = 0.0
    else:
        L_lower = Li_plus_1 + max(Vi_plus_1 * DELTA_T, L_E)

    # Upper bound L_i^(u) - Eq. (10)
    if is_first:  # i = 0 (stop bar)
        L_upper = LANE_LEN
    else:
        # Need V_Q_B_i^i (speed of last inserted NC)
        # Computed from Eq. (11)
        if is_last:  # i = m
            # V_j^m = V_{m+1} + (j-1) * (V_m - V_{m+1}) / Q_B_m
            V_Q_B_i = Vi_plus_1 + (Q_B_i - 1) * (Vi - Vi_plus_1) / Q_B_i
        else:  # i ∈ (0,m)
            # V_j^i = V_{i+1} + j * (V_i - V_{i+1}) / (Q_B_i + 1)
            V_Q_B_i = Vi_plus_1 + Q_B_i * (Vi - Vi_plus_1) / (Q_B_i + 1)

        L_upper = Li - max(V_Q_B_i * DELTA_T, L_E)

    # Compute speeds for all Q_B_i NCs - Eq. (11)
    speeds = []
    if is_first:  # i = 0
        delta_V = (Vi - Vi_plus_1) / Q_B_i
        for j in range(1, Q_B_i + 1):
            V_j = Vi_plus_1 + j * delta_V
            speeds.append(V_j)
    elif is_last:  # i = m
        delta_V = (Vi - Vi_plus_1) / Q_B_i
        for j in range(1, Q_B_i + 1):
            V_j = Vi_plus_1 + (j - 1) * delta_V
            speeds.append(V_j)
    else:  # i ∈ (0,m)
        delta_V = (Vi - Vi_plus_1) / (Q_B_i + 1)
        for j in range(1, Q_B_i + 1):
            V_j = Vi_plus_1 + j * delta_V
            speeds.append(V_j)

    return L_lower, L_upper, speeds

def main():
    # Load data
    segments = pd.read_csv(SEGMENTS_FILE)
    Q_data = pd.read_csv(Q_FILE)
    A_counts = pd.read_csv(A_COUNTS_FILE)

    # Filter Type B segments
    B_segs = segments[segments["seg_type"].str.upper() == "B"].copy()

    if B_segs.empty:
        print("No Type B segments found!")
        pd.DataFrame().to_csv(OUT_COUNTS, index=False)
        pd.DataFrame().to_csv(OUT_POSITIONS, index=False)
        return

    count_rows = []
    position_rows = []

    # Process each EoR time
    for t in sorted(B_segs["EoR_s"].unique()):
        # Get total Q for this time
        q_row = Q_data[Q_data["EoR_s"].round(3) == round(t, 3)]
        if q_row.empty:
            continue
        Q_total = float(q_row.iloc[0]["Q_est"])

        # Get sum of Q_A_i for this time
        a_sum_row = A_counts[A_counts["EoR_s"].round(3) == round(t, 3)]
        Q_A_total = int(a_sum_row["Q_Ai"].sum()) if not a_sum_row.empty else 0

        # Remaining NCs to distribute to Type B
        Q_remaining = max(0, Q_total - Q_A_total)

        # Get B segments for this time, sorted downstream to upstream
        b_t = B_segs[B_segs["EoR_s"].round(3) == round(t, 3)].copy()
        b_t = b_t.sort_values("down_m", ascending=False).reset_index(drop=True)

        if len(b_t) == 0:
            continue

        # Step 1: Compute Q̃_B_i for all segments - Eq. (7)
        Q_tilde_values = []
        for i, row in b_t.iterrows():
            Li = float(row["down_m"])
            Li_plus_1 = float(row["up_m"])
            Vi = float(row["down_speed"])
            Vi_plus_1 = float(row["up_speed"])

            is_boundary = (i == 0) or (i == len(b_t) - 1)
            Q_tilde = compute_Q_tilde_B_i(Li, Li_plus_1, Vi, Vi_plus_1, is_boundary)
            Q_tilde_values.append(Q_tilde)

        sum_Q_tilde = sum(Q_tilde_values)

        # Step 2: Distribute Q_remaining to segments using scaling - Eq. (6)
        Q_B_values = []
        excess = 0

        for i, Q_tilde in enumerate(Q_tilde_values):
            if sum_Q_tilde > 0:
                # Scaling factor ρ = Q_remaining / sum(Q̃_B_i)
                Q_B_i = round_half_up(Q_remaining * Q_tilde / sum_Q_tilde) + excess
            else:
                Q_B_i = excess

            # Ensure Q_B_i doesn't exceed Q_tilde
            if Q_B_i > Q_tilde:
                excess = Q_B_i - Q_tilde
                Q_B_i = Q_tilde
            else:
                excess = 0

            Q_B_values.append(Q_B_i)

        # Step 3: Compute positions for each segment - Eq. (5)
        for i, row in b_t.iterrows():
            Q_B_i = Q_B_values[i]
            Q_tilde_i = Q_tilde_values[i]

            Li = float(row["down_m"])
            Li_plus_1 = float(row["up_m"])
            Vi = float(row["down_speed"])
            Vi_plus_1 = float(row["up_speed"])

            is_first = (i == 0)
            is_last = (i == len(b_t) - 1)

            # Record counts
            count_rows.append({
                "EoR_s": t,
                "lane": row.get("lane", "lane7_0"),
                "seg_idx_B": i,
                "Q_Bi": Q_B_i,
                "Q_tilde_Bi": Q_tilde_i,
                "down_m": Li,
                "up_m": Li_plus_1,
                "down_speed": Vi,
                "up_speed": Vi_plus_1
            })

            if Q_B_i == 0:
                continue

            # Compute bounds and speeds
            L_lower, L_upper, speeds = compute_bounds_and_speeds(
                Li, Li_plus_1, Vi, Vi_plus_1, Q_B_i, is_first, is_last
            )

            if L_lower is None or L_upper is None:
                continue

            # Distribute NCs uniformly within [L_lower, L_upper] - Eq. (5)
            if Q_B_i == 1:
                # Single NC at center
                pos = 0.5 * (L_upper + L_lower)
                position_rows.append({
                    "EoR_s": t,
                    "lane": row.get("lane", "lane7_0"),
                    "seg_idx_B": i,
                    "j_in_seg": 1,
                    "est_pos_m": float(pos),
                    "est_speed_m_s": float(speeds[0])
                })
            else:
                # Multiple NCs uniformly distributed
                for j in range(1, Q_B_i + 1):
                    pos = L_lower + (j - 1) * (L_upper - L_lower) / (Q_B_i - 1)
                    position_rows.append({
                        "EoR_s": t,
                        "lane": row.get("lane", "lane7_0"),
                        "seg_idx_B": i,
                        "j_in_seg": j,
                        "est_pos_m": float(pos),
                        "est_speed_m_s": float(speeds[j-1])
                    })

    # Save results
    pd.DataFrame(count_rows).to_csv(OUT_COUNTS, index=False)
    pd.DataFrame(position_rows).to_csv(OUT_POSITIONS, index=False)
    print(f"Saved {OUT_COUNTS} and {OUT_POSITIONS}")
    print(f"Processed {len(count_rows)} Type B segments")
    print(f"Generated {len(position_rows)} NC position estimates")

if __name__ == "__main__":
    main()
