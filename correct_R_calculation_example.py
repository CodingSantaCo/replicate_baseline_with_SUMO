"""
Proof of concept: CORRECT implementation of R (holding vehicles) calculation.

This demonstrates how to properly calculate R according to the paper's definition:
"Holding vehicles are vehicles that, based on their projected trajectories using
cruise speeds, should have been discharged by that instant but remain held by
the system."
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# Parameters
LANE_LEN = 1000.0
V_FREE = 13.89
V_STOP = 0.5
CYCLE = 60.0
RED = 30.0
EOR_TOL = 0.5


def load_fcd(path: Path) -> pd.DataFrame:
    """Load FCD XML file."""
    rows, cur_t = [], 0.0
    for event, elem in ET.iterparse(str(path), events=("start","end")):
        if event == "start" and elem.tag == "timestep":
            cur_t = float(elem.attrib["time"])
        elif event == "start" and elem.tag == "vehicle":
            a = elem.attrib
            rows.append({
                "time":  cur_t,
                "id":    a.get("id"),
                "lane":  a.get("lane"),
                "pos":   float(a.get("pos","nan")),
                "speed": float(a.get("speed","nan")),
                "type":  a.get("type","")
            })
        if event == "end":
            elem.clear()
    return pd.DataFrame(rows)


def choose_lane(df: pd.DataFrame) -> str:
    """Select the main lane."""
    lanes = df["lane"].dropna().unique().tolist()
    return "lane7_0" if "lane7_0" in lanes else df["lane"].value_counts().idxmax()


def is_cv(type_series: pd.Series, id_series: pd.Series) -> pd.Series:
    """Identify CVs from type or id naming convention."""
    t = type_series.fillna("").astype(str).str.upper()
    iscv = (t == "CV")
    undecided = ~((t=="CV") | (t=="NC"))
    if undecided.any():
        ids = id_series[undecided].astype(str).str.lower()
        iscv.loc[undecided] = ids.str.contains("cv") | ids.str.contains("f_cv")
    return iscv.astype(bool)


def nearest_snapshot(df: pd.DataFrame, t_star: float, tol: float) -> pd.DataFrame:
    """Get vehicles at time t_star within tolerance."""
    snap = df[df["time"].between(t_star - tol, t_star + tol)].copy()
    if snap.empty:
        return snap
    snap["dt"] = (snap["time"] - t_star).abs()
    snap = snap.sort_values(["id","dt"]).drop_duplicates("id", keep="first")
    return snap[(snap["pos"] <= LANE_LEN) & (snap["pos"] >= 0.0)].copy()


def calculate_holding_vehicles_CORRECT(snap_df: pd.DataFrame, entry_times: pd.DataFrame,
                                       eor_time: float, tau_ff: float) -> dict:
    """
    CORRECT calculation of holding vehicles according to paper's definition.

    Args:
        snap_df: DataFrame of vehicles at EoR snapshot
        entry_times: DataFrame with columns ['id', 'entry_time']
        eor_time: End of Red time
        tau_ff: Free-flow travel time through the lane

    Returns:
        dict with R, R_CV, R_NC, and diagnostic info
    """
    # Cutoff time: vehicles that entered before this should have exited by now
    cutoff_time = eor_time - tau_ff

    # Merge snapshot with entry times
    snap_with_entry = snap_df.merge(entry_times, on='id', how='left')

    # Filter: only vehicles that entered before cutoff are holding vehicles
    holding_mask = snap_with_entry['entry_time'] < cutoff_time
    holding_vehicles = snap_with_entry[holding_mask].copy()

    # Total holding vehicles
    R = len(holding_vehicles)

    # Split by CV/NC
    if R > 0:
        iscv_holding = is_cv(holding_vehicles["type"], holding_vehicles["id"])
        R_CV = int(iscv_holding.sum())
        R_NC = int((~iscv_holding).sum())
    else:
        R_CV = 0
        R_NC = 0

    # Diagnostic info
    total_in_lane = len(snap_df)
    stopped_vehicles = len(snap_df[snap_df["speed"] <= V_STOP])

    return {
        'R': R,
        'R_CV': R_CV,
        'R_NC': R_NC,
        'cutoff_time': cutoff_time,
        'total_in_lane': total_in_lane,
        'stopped_vehicles': stopped_vehicles,
        'holding_ratio': R / max(stopped_vehicles, 1),  # How many stopped are actually holding
        'method': 'correct_entry_time_based'
    }


def calculate_holding_vehicles_INCORRECT(snap_df: pd.DataFrame) -> dict:
    """
    INCORRECT calculation (current implementation) for comparison.

    This counts ALL stopped vehicles as R1.
    """
    stopped = snap_df["speed"] <= V_STOP
    R1 = int(stopped.sum())

    # Split by CV/NC
    if R1 > 0:
        iscv_stopped = is_cv(snap_df[stopped]["type"], snap_df[stopped]["id"])
        R_CV = int(iscv_stopped.sum())
        R_NC = int((~iscv_stopped).sum())
    else:
        R_CV = 0
        R_NC = 0

    return {
        'R': R1,
        'R_CV': R_CV,
        'R_NC': R_NC,
        'method': 'incorrect_all_stopped'
    }


def main():
    base = Path(".")
    fcd_path = base / "fcd_output.xml"

    if not fcd_path.exists():
        print(f"ERROR: {fcd_path} not found")
        print("This is a proof-of-concept script. Run it after generating SUMO data.")
        return

    print("="*70)
    print("PROOF OF CONCEPT: Correct R Calculation")
    print("="*70)

    # Load data
    print("\nLoading FCD data...")
    df_all = load_fcd(fcd_path)
    lane_id = choose_lane(df_all)
    df_lane = df_all[df_all["lane"]==lane_id].copy()

    print(f"Selected lane: {lane_id}")
    print(f"Total records: {len(df_lane)}")

    # Calculate entry times for all vehicles
    print("\nCalculating vehicle entry times...")
    entry_times = df_lane.groupby("id", as_index=False)["time"].min()
    entry_times.rename(columns={"time": "entry_time"}, inplace=True)
    print(f"Tracked {len(entry_times)} unique vehicles")

    # Free-flow travel time
    tau_ff = LANE_LEN / V_FREE
    print(f"\nFree-flow travel time (tau_ff): {tau_ff:.2f} s")

    # Analyze first few EoR snapshots
    sim_end = float(df_lane["time"].max())
    eor_times = [RED, RED + CYCLE, RED + 2*CYCLE]  # First 3 EoRs

    print("\n" + "="*70)
    print("COMPARISON: Correct vs. Incorrect R Calculation")
    print("="*70)

    results = []

    for t in eor_times:
        if t > sim_end:
            break

        print(f"\n{'-'*70}")
        print(f"EoR @ {t:.1f}s")
        print(f"{'-'*70}")

        snap = nearest_snapshot(df_lane, t, EOR_TOL)
        if snap.empty:
            print(f"  Warning: No vehicles at EoR {t:.1f}s")
            continue

        # CORRECT calculation
        correct = calculate_holding_vehicles_CORRECT(snap, entry_times, t, tau_ff)

        # INCORRECT calculation (current implementation)
        incorrect = calculate_holding_vehicles_INCORRECT(snap)

        print(f"\nCutoff time: {correct['cutoff_time']:.2f}s")
        print(f"(Vehicles that entered before this should have exited by now)")

        print(f"\nVehicles in lane: {correct['total_in_lane']}")
        print(f"Stopped vehicles: {correct['stopped_vehicles']}")

        print(f"\nCORRECT R (entry time based):")
        print(f"  R = {correct['R']}")
        print(f"  R_CV = {correct['R_CV']}")
        print(f"  R_NC = {correct['R_NC']}")
        print(f"  Holding ratio = {correct['holding_ratio']:.2%}")

        print(f"\nINCORRECT R (all stopped):")
        print(f"  R = {incorrect['R']}")
        print(f"  R_CV = {incorrect['R_CV']}")
        print(f"  R_NC = {incorrect['R_NC']}")

        error = incorrect['R'] - correct['R']
        error_pct = (error / max(correct['R'], 1)) * 100

        print(f"\nERROR:")
        print(f"  Overcounting = {error} vehicles")
        if correct['R'] > 0:
            print(f"  Relative error = {error_pct:.1f}%")
        else:
            print(f"  Relative error = INFINITE (correct R = 0, calculated R = {incorrect['R']})")

        results.append({
            'EoR_s': t,
            'cutoff_s': correct['cutoff_time'],
            'R_correct': correct['R'],
            'R_incorrect': incorrect['R'],
            'R_CV_correct': correct['R_CV'],
            'R_CV_incorrect': incorrect['R_CV'],
            'R_NC_correct': correct['R_NC'],
            'R_NC_incorrect': incorrect['R_NC'],
            'error': error,
            'total_in_lane': correct['total_in_lane'],
            'stopped': correct['stopped_vehicles']
        })

    # Save results
    if results:
        df_out = pd.DataFrame(results)
        out_file = base / "R_comparison_correct_vs_incorrect.csv"
        df_out.to_csv(out_file, index=False)
        print(f"\n{'='*70}")
        print(f"Saved comparison to: {out_file}")
        print(f"{'='*70}")

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("The CORRECT calculation tracks entry times and only counts vehicles")
    print("that have been in the lane longer than tau_ff (free-flow travel time).")
    print("\nThe INCORRECT calculation counts ALL stopped vehicles, regardless")
    print("of how long they've been in the lane.")
    print("\nThis leads to massive overcounting, especially in early cycles!")
    print("="*70)


if __name__ == "__main__":
    main()
