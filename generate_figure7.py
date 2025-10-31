"""
Generate Figure 7 visualization - vehicle location estimation results.
Shows ground truth vs estimated NC and CV positions for baseline case.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters
FCD_FILE = "fcd_output.xml"
ESTIMATES_FILE = "all_NC_positions_estimated.csv"
OUTPUT_FILE = "figure7_baseline.png"

LANE_ID = "lane7_0"
LANE_LEN = 1000.0  # meters
TIME_TOL = 0.5     # seconds

# Select one representative cycle to visualize
CYCLE_TO_PLOT = 5  # 5th cycle (around 300s = 30s + 4*60s)

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

def classify_vehicles(type_series: pd.Series, id_series: pd.Series):
    """Classify vehicles as CV or NC."""
    t = type_series.fillna("").astype(str).str.upper()
    is_cv = (t == "CV")

    # Check ID for vehicles without explicit type
    undecided = ~((t == "CV") | (t == "NC"))
    if undecided.any():
        ids = id_series[undecided].astype(str).str.lower()
        is_cv.loc[undecided] = ids.str.contains("cv") | ids.str.contains("f_cv")

    return is_cv.astype(bool)

def get_snapshot_at_time(df_lane: pd.DataFrame, t: float, tol: float) -> tuple:
    """Get ground truth positions at time t, separated by CV/NC."""
    # Get snapshot
    snap = df_lane[df_lane["time"].between(t - tol, t + tol)].copy()
    if snap.empty:
        return [], []

    # Take closest time for each vehicle
    snap["dt"] = (snap["time"] - t).abs()
    snap = snap.sort_values(["id", "dt"]).drop_duplicates("id", keep="first")

    # Classify
    is_cv = classify_vehicles(snap["type"], snap["id"])

    cv_positions = snap[is_cv]["pos"].values
    nc_positions = snap[~is_cv]["pos"].values

    return cv_positions, nc_positions

def create_figure7_plot(t_plot, cv_gt, nc_gt, cv_est, nc_est):
    """Create Figure 7 style visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Y-axis levels
    y_gt = 1.0      # Ground truth
    y_cvvl = 0.0    # CVVL-S estimate

    # Plot ground truth
    if len(cv_gt) > 0:
        ax.scatter(cv_gt, [y_gt] * len(cv_gt), c='red', marker='o', s=100,
                   label='Ground truth CV', zorder=3, edgecolors='darkred', linewidths=1.5)

    if len(nc_gt) > 0:
        ax.scatter(nc_gt, [y_gt] * len(nc_gt), c='black', marker='s', s=100,
                   label='Ground truth NC', zorder=3, edgecolors='gray', linewidths=1.5)

    # Plot CVVL-S estimates
    if len(cv_est) > 0:
        ax.scatter(cv_est, [y_cvvl] * len(cv_est), c='red', marker='o', s=100,
                   label='CVVL-S CV', zorder=3, alpha=0.7, edgecolors='darkred', linewidths=1.5)

    if len(nc_est) > 0:
        ax.scatter(nc_est, [y_cvvl] * len(nc_est), c='green', marker='♦', s=120,
                   label='CVVL-S NC (estimated)', zorder=3, alpha=0.7, edgecolors='darkgreen', linewidths=1.5)

    # Formatting
    ax.set_xlabel('Location (m)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, LANE_LEN)
    ax.set_ylim(-0.5, 1.5)

    # Y-axis labels
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CVVL-S', 'Ground truth'], fontsize=12)

    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add stop bar line
    ax.axvline(x=LANE_LEN, color='orange', linestyle='--', linewidth=2, label='Stop bar', alpha=0.7)

    # Title
    ax.set_title(f'Figure 7 - Baseline Case (t = {t_plot:.1f}s, EoR)',
                 fontsize=15, fontweight='bold', pad=15)

    # Legend
    ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.9)

    plt.tight_layout()
    return fig

def main():
    print("\n" + "="*60)
    print("GENERATING FIGURE 7 VISUALIZATION")
    print("="*60)

    # Load FCD data
    fcd_path = Path(FCD_FILE)
    if not fcd_path.exists():
        print(f"❌ Error: {FCD_FILE} not found!")
        return

    print(f"\nLoading SUMO simulation data...")
    df_all = load_fcd(fcd_path)
    df_lane = df_all[df_all["lane"] == LANE_ID].copy()

    # Load estimates
    est_path = Path(ESTIMATES_FILE)
    if not est_path.exists():
        print(f"❌ Error: {ESTIMATES_FILE} not found!")
        print("Please run: python run_cvvl_s_pipeline.py")
        return

    print(f"Loading estimates...")
    estimates = pd.read_csv(est_path)

    # Get list of EoR times
    eor_times = sorted(estimates["EoR_s"].unique())

    if CYCLE_TO_PLOT > len(eor_times):
        print(f"Warning: Only {len(eor_times)} cycles available, using last cycle")
        t_plot = eor_times[-1]
    else:
        t_plot = eor_times[CYCLE_TO_PLOT - 1]

    print(f"\nPlotting cycle {CYCLE_TO_PLOT} at t={t_plot:.1f}s")

    # Get ground truth for this time
    cv_gt, nc_gt = get_snapshot_at_time(df_lane, t_plot, TIME_TOL)
    print(f"  Ground truth: {len(cv_gt)} CVs, {len(nc_gt)} NCs")

    # Get estimates for this time
    est_t = estimates[estimates["EoR_s"].round(3) == round(t_plot, 3)].copy()

    # For CVs, we use ground truth (known positions from CV trajectory data)
    cv_est = cv_gt.copy()  # CVs are known

    # For NCs, we use our estimates
    nc_est = est_t["est_pos_m"].values
    print(f"  Estimated: {len(cv_est)} CVs (known), {len(nc_est)} NCs (estimated)")

    # Create plot
    fig = create_figure7_plot(t_plot, cv_gt, nc_gt, cv_est, nc_est)

    # Save
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {OUTPUT_FILE}")

    # Also create a summary statistics figure
    print("\nGenerating summary statistics plot...")
    create_summary_plot(estimates, df_lane, eor_times)

    print("\n" + "="*60)

def create_summary_plot(estimates, df_lane, eor_times):
    """Create summary plot showing NC count over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Number of NCs over time
    est_counts = []
    gt_counts = []

    for t in eor_times:
        # Estimated count
        est_t = estimates[estimates["EoR_s"].round(3) == round(t, 3)]
        est_counts.append(len(est_t))

        # Ground truth count
        _, nc_gt = get_snapshot_at_time(df_lane, t, TIME_TOL)
        gt_counts.append(len(nc_gt))

    ax1.plot(eor_times, gt_counts, 'o-', label='Ground Truth', linewidth=2, markersize=6)
    ax1.plot(eor_times, est_counts, 's--', label='CVVL-S Estimate', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of NCs', fontsize=12, fontweight='bold')
    ax1.set_title('NC Count Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Estimation error
    errors = np.array(est_counts) - np.array(gt_counts)
    ax2.bar(eor_times, errors, width=10, alpha=0.7, color=['red' if e < 0 else 'green' for e in errors])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Estimation Error (Est - GT)', fontsize=12, fontweight='bold')
    ax2.set_title('NC Count Estimation Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig("figure7_summary.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary plot to figure7_summary.png")

if __name__ == "__main__":
    main()
