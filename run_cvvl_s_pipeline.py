"""
Main pipeline to run complete CVVL-S algorithm for source lane vehicle location estimation.
Executes all steps in sequence and combines results.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def run_script(script_name):
    """Run a Python script and check for errors."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_name} failed with return code {result.returncode}")
    print(f"✓ {script_name} completed successfully")

def combine_estimates():
    """Combine Type A and Type B NC position estimates."""
    print(f"\n{'='*60}")
    print("Combining Type A and Type B estimates")
    print('='*60)

    # Load Type A positions
    a_pos_file = Path("A_positions_ROUND.csv")
    b_pos_file = Path("B_positions.csv")

    all_positions = []

    if a_pos_file.exists():
        a_pos = pd.read_csv(a_pos_file)
        if not a_pos.empty:
            a_pos["segment_type"] = "A"
            a_pos = a_pos.rename(columns={"seg_idx_A": "seg_idx"})
            all_positions.append(a_pos)
            print(f"  Type A: {len(a_pos)} NC positions")

    if b_pos_file.exists():
        b_pos = pd.read_csv(b_pos_file)
        if not b_pos.empty:
            b_pos["segment_type"] = "B"
            b_pos = b_pos.rename(columns={"seg_idx_B": "seg_idx"})
            all_positions.append(b_pos)
            print(f"  Type B: {len(b_pos)} NC positions")

    if all_positions:
        combined = pd.concat(all_positions, ignore_index=True)
        combined = combined.sort_values(["EoR_s", "est_pos_m"], ascending=[True, False])

        # Save combined results
        out_file = "all_NC_positions_estimated.csv"
        combined.to_csv(out_file, index=False)
        print(f"\n✓ Saved combined estimates to {out_file}")
        print(f"  Total NC positions estimated: {len(combined)}")

        # Summary by time
        summary = combined.groupby("EoR_s").size().reset_index(name="total_NCs_estimated")
        print(f"\n  Summary by EoR time:")
        print(summary.to_string(index=False))

        return combined
    else:
        print("  Warning: No position estimates found!")
        return pd.DataFrame()

def main():
    """Run complete CVVL-S pipeline."""
    print("\n" + "="*60)
    print("CVVL-S PIPELINE - Baseline Case Replication")
    print("Paper: Real-time vehicle location estimation in signalized networks")
    print("="*60)

    # Check if SUMO simulation output exists
    fcd_file = Path("fcd_output.xml")
    if not fcd_file.exists():
        print("\n❌ Error: fcd_output.xml not found!")
        print("Please run SUMO simulation first:")
        print("  sumo -c simulation.sumocfg")
        return

    try:
        # Step 1: Estimate holding vehicles (R)
        run_script("recompute_R_CVHV_fixed.py")

        # Step 2: Estimate total number of NCs (Q) - Section 3.1
        run_script("estimate_Q_from_CV_only.py")

        # Step 3: Divide lane into Type A and Type B segments
        run_script("segments_AB_resegmented_v05_entr.py")

        # Step 4: Compute NC distribution in Type A segments
        run_script("compute_A_eq34_ROUND.py")

        # Step 5: Compute NC distribution in Type B segments
        run_script("compute_B_distribution.py")

        # Step 6: Combine all estimates
        combined = combine_estimates()

        print("\n" + "="*60)
        print("✓ CVVL-S PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

        if not combined.empty:
            print(f"\nOutput files generated:")
            print(f"  - holding_EoR_fixed.csv (holding vehicles)")
            print(f"  - eor_Q_estimated.csv (total NC count)")
            print(f"  - segments_AB_resegmented_v05_entr.csv (lane segments)")
            print(f"  - A_counts_ROUND.csv (Type A NC counts)")
            print(f"  - A_positions_ROUND.csv (Type A NC positions)")
            print(f"  - B_counts.csv (Type B NC counts)")
            print(f"  - B_positions.csv (Type B NC positions)")
            print(f"  - all_NC_positions_estimated.csv (combined estimates)")

            print(f"\nNext steps:")
            print(f"  1. Run evaluation: python evaluate_baseline.py")
            print(f"  2. Generate Table 1: python generate_table1.py")
            print(f"  3. Generate Figure 7: python generate_figure7.py")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
