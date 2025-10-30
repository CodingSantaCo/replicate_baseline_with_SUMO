"""
Evaluate CVVL-S estimates against ground truth from SUMO simulation.
Computes precision, recall, and F1 score for Table 1 baseline case.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

# Parameters
FCD_FILE = "fcd_output.xml"
ESTIMATES_FILE = "all_NC_positions_estimated.csv"
OUTPUT_FILE = "evaluation_results.csv"

LANE_ID = "lane7_0"
TOLERANCE = 10.0  # meters - threshold for matching (from paper Section 5.2)
TIME_TOL = 0.5    # seconds - snapshot tolerance

def load_fcd(path: Path) -> pd.DataFrame:
    """Load SUMO FCD output."""
    rows, cur_t = [], 0.0
    for event, elem in ET.iterparse(str(path), events=("start", "end")):
        if event == "start" and elem.tag == "timestep":
            cur_t = float(elem.attrib["time"])
        elif event == "start" and elem.tag == "vehicle":
            a = elem.attrib
            rows.append({
                "time": cur_t,
                "id": a.get("id"),
                "lane": a.get("lane"),
                "pos": float(a.get("pos", "nan")),
                "speed": float(a.get("speed", "nan")),
                "type": a.get("type", "")
            })
        if event == "end":
            elem.clear()
    return pd.DataFrame(rows)

def is_nc(type_series: pd.Series, id_series: pd.Series) -> pd.Series:
    """Identify non-connected vehicles (NCs)."""
    t = type_series.fillna("").astype(str).str.upper()
    is_nc_type = (t == "NC")

    # Also check ID for vehicles without explicit type
    undecided = ~((t == "CV") | (t == "NC"))
    if undecided.any():
        ids = id_series[undecided].astype(str).str.lower()
        # If ID contains "cv", it's a CV, otherwise NC
        is_cv_id = ids.str.contains("cv") | ids.str.contains("f_cv")
        is_nc_type.loc[undecided] = ~is_cv_id

    return is_nc_type.astype(bool)

def get_ground_truth_at_time(df_lane: pd.DataFrame, t: float, tol: float) -> pd.DataFrame:
    """Get ground truth NC positions at time t."""
    # Get snapshot at time t ± tolerance
    snap = df_lane[df_lane["time"].between(t - tol, t + tol)].copy()
    if snap.empty:
        return pd.DataFrame()

    # Take closest time for each vehicle
    snap["dt"] = (snap["time"] - t).abs()
    snap = snap.sort_values(["id", "dt"]).drop_duplicates("id", keep="first")

    # Filter NCs
    is_nc_mask = is_nc(snap["type"], snap["id"])
    ncs = snap[is_nc_mask].copy()

    return ncs[["id", "pos", "speed"]].reset_index(drop=True)

def compute_metrics(estimated: pd.DataFrame, ground_truth: pd.DataFrame, tolerance: float):
    """
    Compute precision, recall, and F1 score.

    TP (True Positive): Estimated NC within tolerance of a ground truth NC
    FP (False Positive): Estimated NC with no ground truth NC within tolerance
    FN (False Negative): Ground truth NC with no estimated NC within tolerance
    """
    if estimated.empty and ground_truth.empty:
        return {"TP": 0, "FP": 0, "FN": 0, "precision": 1.0, "recall": 1.0, "F1": 1.0}

    if estimated.empty:
        return {
            "TP": 0,
            "FP": 0,
            "FN": len(ground_truth),
            "precision": 0.0,
            "recall": 0.0,
            "F1": 0.0
        }

    if ground_truth.empty:
        return {
            "TP": 0,
            "FP": len(estimated),
            "FN": 0,
            "precision": 0.0,
            "recall": 0.0,
            "F1": 0.0
        }

    est_positions = estimated["est_pos_m"].values
    gt_positions = ground_truth["pos"].values

    # For each estimated position, find if there's a ground truth within tolerance
    matched_est = np.zeros(len(est_positions), dtype=bool)
    matched_gt = np.zeros(len(gt_positions), dtype=bool)

    for i, est_pos in enumerate(est_positions):
        # Find distance to all ground truth positions
        distances = np.abs(gt_positions - est_pos)
        within_tol = distances <= tolerance

        if np.any(within_tol):
            # Find closest ground truth position
            closest_idx = np.argmin(distances)
            if distances[closest_idx] <= tolerance and not matched_gt[closest_idx]:
                matched_est[i] = True
                matched_gt[closest_idx] = True

    TP = int(matched_est.sum())
    FP = int((~matched_est).sum())
    FN = int((~matched_gt).sum())

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": F1
    }

def main():
    print("\n" + "="*60)
    print("EVALUATING CVVL-S ESTIMATES")
    print("="*60)

    # Load FCD data
    fcd_path = Path(FCD_FILE)
    if not fcd_path.exists():
        print(f"❌ Error: {FCD_FILE} not found!")
        return

    print(f"\nLoading SUMO simulation data from {FCD_FILE}...")
    df_all = load_fcd(fcd_path)
    df_lane = df_all[df_all["lane"] == LANE_ID].copy()
    print(f"  Loaded {len(df_lane)} vehicle observations on {LANE_ID}")

    # Load estimates
    est_path = Path(ESTIMATES_FILE)
    if not est_path.exists():
        print(f"❌ Error: {ESTIMATES_FILE} not found!")
        print("Please run: python run_cvvl_s_pipeline.py")
        return

    print(f"\nLoading estimates from {ESTIMATES_FILE}...")
    estimates = pd.read_csv(est_path)
    print(f"  Loaded {len(estimates)} NC position estimates")

    # Evaluate for each EoR time
    results = []
    eor_times = sorted(estimates["EoR_s"].unique())

    print(f"\nEvaluating at {len(eor_times)} time points...")
    print(f"Tolerance: {TOLERANCE} meters\n")

    for t in eor_times:
        # Get estimates at this time
        est_t = estimates[estimates["EoR_s"].round(3) == round(t, 3)].copy()

        # Get ground truth at this time
        gt_t = get_ground_truth_at_time(df_lane, t, TIME_TOL)

        # Compute metrics
        metrics = compute_metrics(est_t, gt_t, TOLERANCE)

        results.append({
            "EoR_s": t,
            "n_estimated": len(est_t),
            "n_ground_truth": len(gt_t),
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "F1": metrics["F1"]
        })

        if t == eor_times[0]:  # Print first result as example
            print(f"Example (t={t:.1f}s):")
            print(f"  Estimated NCs: {len(est_t)}")
            print(f"  Ground truth NCs: {len(gt_t)}")
            print(f"  TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}")
            print(f"  Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['F1']:.3f}\n")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Compute overall metrics
    total_TP = results_df["TP"].sum()
    total_FP = results_df["FP"].sum()
    total_FN = results_df["FN"].sum()

    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    overall_F1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    # Save detailed results
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Print summary
    print("="*60)
    print("EVALUATION RESULTS - BASELINE CASE")
    print("="*60)
    print(f"\nOverall Performance:")
    print(f"  Precision: {overall_precision:.3f} ({overall_precision*100:.1f}%)")
    print(f"  Recall:    {overall_recall:.3f} ({overall_recall*100:.1f}%)")
    print(f"  F1 Score:  {overall_F1:.3f} ({overall_F1*100:.1f}%)")
    print(f"\nTotal Statistics:")
    print(f"  True Positives (TP):  {total_TP}")
    print(f"  False Positives (FP): {total_FP}")
    print(f"  False Negatives (FN): {total_FN}")
    print(f"  Total Estimated NCs:  {total_TP + total_FP}")
    print(f"  Total Ground Truth NCs: {total_TP + total_FN}")

    print(f"\nPer-cycle Statistics:")
    print(f"  Mean Precision: {results_df['precision'].mean():.3f} (std={results_df['precision'].std():.3f})")
    print(f"  Mean Recall:    {results_df['recall'].mean():.3f} (std={results_df['recall'].std():.3f})")
    print(f"  Mean F1:        {results_df['F1'].mean():.3f} (std={results_df['F1'].std():.3f})")

    print(f"\n✓ Detailed results saved to {OUTPUT_FILE}")

    # Compare with Table 1 baseline (EVLS vs CVVL-S)
    print("\n" + "="*60)
    print("COMPARISON WITH PAPER TABLE 1 BASELINE")
    print("="*60)
    print("\nExpected results from paper (CVVL-S):")
    print("  Precision: 58%")
    print("  Recall:    76%")
    print("  F1:        66%")
    print(f"\nOur implementation:")
    print(f"  Precision: {overall_precision*100:.0f}%")
    print(f"  Recall:    {overall_recall*100:.0f}%")
    print(f"  F1:        {overall_F1*100:.0f}%")

    diff_prec = abs(overall_precision - 0.58) * 100
    diff_rec = abs(overall_recall - 0.76) * 100
    diff_f1 = abs(overall_F1 - 0.66) * 100

    print(f"\nDifference from paper:")
    print(f"  Precision: {diff_prec:+.1f}%")
    print(f"  Recall:    {diff_rec:+.1f}%")
    print(f"  F1:        {diff_f1:+.1f}%")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
