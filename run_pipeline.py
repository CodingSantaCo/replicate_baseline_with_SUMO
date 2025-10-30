#!/usr/bin/env python3
"""
Pipeline Runner for Dynamic Q Estimation
=========================================

This script runs the complete pipeline for estimating the number of non-connected
vehicles (NCs) at End of Red (EoR) using dynamic, cycle-by-cycle parameters as
described in the paper.

Pipeline Steps:
1. estimate_dynamic_params.py - Calculate q, p, R per cycle with rolling window
2. estimate_Q_dynamic.py - Calculate Q using the dynamic parameters

Required Input:
- fcd_output.xml (SUMO FCD output)

Output Files:
- dynamic_params_per_cycle.csv - Dynamic q, p, R estimates per cycle
- eor_Q_estimated_dynamic.csv - Final Q estimates per cycle
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name: str, description: str):
    """Run a Python script and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"Script: {script_name}")
    print('='*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}")
        print(f"  Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_name}")
        return False

def check_files():
    """Check if required input files exist."""
    base = Path(".")
    required = ["fcd_output.xml"]
    
    missing = []
    for fname in required:
        if not (base / fname).exists():
            missing.append(fname)
    
    if missing:
        print("ERROR: Missing required input files:")
        for fname in missing:
            print(f"  - {fname}")
        return False
    
    return True

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║   Dynamic Q Estimation Pipeline                                  ║
║   Based on: "Real-time vehicle location estimation in           ║
║             signalized networks using partial CV trajectory data" ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check input files
    print("Checking input files...")
    if not check_files():
        print("\n✗ Pipeline cannot proceed. Please provide required files.")
        sys.exit(1)
    
    print("✓ All required input files found\n")
    
    # Step 1: Estimate dynamic parameters (q, p, R) per cycle
    success = run_script(
        "estimate_dynamic_params_improved.py",
        "Estimate q, p, R dynamically per cycle (rolling 3-cycle window)"
    )
    
    if not success:
        print("\n✗ Pipeline failed at Step 1")
        sys.exit(1)
    
    # Step 2: Estimate Q using dynamic parameters
    success = run_script(
        "estimate_Q_dynamic.py",
        "Estimate Q (number of NCs) using dynamic parameters"
    )
    
    if not success:
        print("\n✗ Pipeline failed at Step 2")
        sys.exit(1)
    
    # Success
    print(f"\n{'='*70}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print('='*70)
    print("\nOutput files:")
    print("  1. dynamic_params_per_cycle.csv - Dynamic q, p, R per cycle")
    print("  2. eor_Q_estimated_dynamic.csv - Final Q estimates")
    print("\nNext steps:")
    print("  - Compare with ground truth NC counts")
    print("  - Visualize trends across cycles")
    print("  - Analyze parameter dynamics")

if __name__ == "__main__":
    main()
