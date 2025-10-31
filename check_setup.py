"""
Check setup and provide clear next steps for running the CVVL-S baseline replication.
"""

import sys
import subprocess
from pathlib import Path

def check_sumo():
    """Check if SUMO is installed."""
    try:
        result = subprocess.run(['sumo', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ SUMO is installed")
            print(f"   Version: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("❌ SUMO is NOT installed")
    print("   Install with: sudo apt-get install sumo sumo-tools")
    print("   Or see: https://sumo.dlr.de/docs/Installing/index.html")
    return False

def check_python_packages():
    """Check if required Python packages are installed."""
    packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations',
        'matplotlib': 'Visualization'
    }

    all_ok = True
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"✅ {pkg:12} - {desc}")
        except ImportError:
            print(f"❌ {pkg:12} - {desc} (NOT INSTALLED)")
            all_ok = False

    if not all_ok:
        print("\n   Install missing packages with:")
        print("   pip install pandas numpy matplotlib")

    return all_ok

def check_files():
    """Check if all required files exist."""
    files = {
        'SUMO config': 'simulation.sumocfg',
        'Network': 'network_net.xml',
        'Routes': 'routes_rou.xml',
        'Holding estimation': 'recompute_R_CVHV_fixed.py',
        'Q estimation': 'estimate_Q_from_CV_only.py',
        'Segmentation': 'segments_AB_resegmented_v05_entr.py',
        'Type A distribution': 'compute_A_eq34_ROUND.py',
        'Type B distribution': 'compute_B_distribution.py',
        'Pipeline': 'run_cvvl_s_pipeline.py',
        'Evaluation': 'evaluate_baseline.py',
        'Visualization': 'generate_figure7.py'
    }

    all_ok = True
    for desc, filename in files.items():
        if Path(filename).exists():
            print(f"✅ {desc:20} - {filename}")
        else:
            print(f"❌ {desc:20} - {filename} (MISSING)")
            all_ok = False

    return all_ok

def check_output_exists():
    """Check if SUMO simulation has been run."""
    if Path('fcd_output.xml').exists():
        print("✅ SUMO output exists (fcd_output.xml)")
        size_mb = Path('fcd_output.xml').stat().st_size / 1024 / 1024
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print("⚠️  SUMO output NOT found (fcd_output.xml)")
        print("   Need to run: sumo -c simulation.sumocfg")
        return False

def main():
    print("="*70)
    print("CVVL-S BASELINE REPLICATION - SETUP CHECK")
    print("="*70)
    print()

    print("Checking Python packages...")
    print("-" * 70)
    python_ok = check_python_packages()
    print()

    print("Checking SUMO installation...")
    print("-" * 70)
    sumo_ok = check_sumo()
    print()

    print("Checking implementation files...")
    print("-" * 70)
    files_ok = check_files()
    print()

    print("Checking SUMO simulation output...")
    print("-" * 70)
    output_exists = check_output_exists()
    print()

    # Print summary and next steps
    print("="*70)
    print("SUMMARY")
    print("="*70)

    all_ready = python_ok and sumo_ok and files_ok

    if not python_ok:
        print("\n❌ MISSING PYTHON PACKAGES")
        print("   → Install with: pip install pandas numpy matplotlib")

    if not sumo_ok:
        print("\n❌ SUMO NOT INSTALLED")
        print("   → Install SUMO:")
        print("     Ubuntu/Debian: sudo apt-get install sumo sumo-tools")
        print("     Other: https://sumo.dlr.de/docs/Installing/index.html")

    if not files_ok:
        print("\n❌ MISSING IMPLEMENTATION FILES")
        print("   → Re-run the setup scripts")

    if all_ready:
        print("\n✅ All dependencies are installed!")

        if not output_exists:
            print("\n📋 NEXT STEP: Run SUMO simulation")
            print("   → Command: sumo -c simulation.sumocfg")
            print("   → Duration: ~1-2 minutes for 3600s simulation")
            print("   → Output: fcd_output.xml (~150-300 MB)")
        else:
            print("\n✅ SUMO simulation output exists!")
            print("\n📋 NEXT STEP: Run CVVL-S pipeline")
            print("   → Command: python run_cvvl_s_pipeline.py")
            print("   → Duration: ~30-60 seconds")
            print("\n   Then:")
            print("   → Evaluate: python evaluate_baseline.py")
            print("   → Visualize: python generate_figure7.py")

    print("\n" + "="*70)
    print("For detailed instructions, see: README_COMPLETE.md")
    print("For implementation details, see: IMPLEMENTATION_SUMMARY.md")
    print("="*70)

if __name__ == "__main__":
    main()
