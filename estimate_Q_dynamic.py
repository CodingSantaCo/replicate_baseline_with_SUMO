import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- Parameters ----------------
LANE_LEN = 1000.0         # m, stopline coordinate
V_FREE   = 13.89          # m/s, free-flow speed
CYCLE    = 60.0           # s
RED      = 30.0           # s (EoR offset)
EOR_TOL  = 0.5            # s, snapshot tolerance

# Input/Output files
FCD_FILE  = "fcd_output.xml"
PARAMS_FILE = "dynamic_params_per_cycle.csv"  # NEW: read dynamic params
OUT_CSV  = "eor_Q_estimated_dynamic.csv"
# -------------------------------------------

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
    """Select the main lane (prefer lane7_0 or most frequent)."""
    lanes = df["lane"].dropna().unique().tolist()
    return "lane7_0" if "lane7_0" in lanes else df["lane"].value_counts().idxmax()

def load_dynamic_params(path: Path) -> pd.DataFrame:
    """Load pre-computed dynamic parameters (q, p, R) per cycle."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Please run 'estimate_dynamic_params.py' first!"
        )
    
    df = pd.read_csv(path)
    required = {"EoR_s", "q_hat", "p_hat", "R", "R_CV"}
    missing = required - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    
    return df

def main():
    base = Path(".")
    fcd_path = base / FCD_FILE
    params_path = base / PARAMS_FILE
    
    # Check files
    if not fcd_path.exists():
        raise FileNotFoundError(f"Missing {FCD_FILE}")
    
    print("Loading FCD data...")
    df_all = load_fcd(fcd_path)
    lane_id = choose_lane(df_all)
    df_lane = df_all[df_all["lane"]==lane_id].copy()
    
    print(f"Selected lane: {lane_id}")
    
    # Load dynamic parameters
    print(f"Loading dynamic parameters from {PARAMS_FILE}...")
    df_params = load_dynamic_params(params_path)
    
    print(f"Found {len(df_params)} cycles with dynamic parameters")
    
    # Process each EoR time
    results = []
    
    for idx, row in df_params.iterrows():
        t = float(row['EoR_s'])
        q_hat = float(row['q_hat'])
        p_hat = float(row['p_hat'])
        R = float(row['R'])
        R_CV = float(row['R_CV'])
        
        # Calculate Q using Proposition 1 from the paper
        # Q = q(1-p) * (l/vf) + (R - R_CV)
        
        q_nc = q_hat * (1.0 - p_hat)
        tau_ff = LANE_LEN / V_FREE
        
        # Expected NCs in lane (free-flow travel time)
        Q_expected = q_nc * tau_ff
        
        # Holding NCs (additional due to signal)
        Q_holding = R - R_CV
        
        # Total estimated NCs
        Q_total = Q_expected + Q_holding
        
        print(f"\nCycle {idx+1}, EoR @ {t:.1f}s:")
        print(f"  q={q_hat:.4f} veh/s, p={p_hat:.3f}")
        print(f"  R={R:.0f}, R_CV={R_CV:.0f}, R_NC={Q_holding:.0f}")
        print(f"  Q_expected={Q_expected:.1f}, Q_holding={Q_holding:.1f}")
        print(f"  → Q_total={Q_total:.1f} NCs")
        
        results.append({
            'EoR_s': float(t),
            'cycle_id': int(row.get('cycle_id', idx + 1)),
            'lane': lane_id,
            
            # Input parameters (dynamic)
            'q_hat': float(q_hat),
            'p_hat': float(p_hat),
            'q_nc_hat': float(q_nc),
            'R': float(R),
            'R_CV': float(R_CV),
            
            # Components of Q
            'Q_expected': float(Q_expected),
            'Q_holding': float(Q_holding),
            
            # Final estimate
            'Q_total': float(Q_total),
            
            # Additional info
            'tau_ff': float(tau_ff),
            'window_cycles': int(row.get('window_cycles', 0))
        })
    
    # Save results
    df_out = pd.DataFrame(results)
    out_file = base / OUT_CSV
    df_out.to_csv(out_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Saved Q estimates to: {out_file}")
    print(f"  Total cycles: {len(results)}")
    print(f"{'='*60}")
    
    # Summary statistics
    print("\nQ Estimation Summary:")
    print(f"  Q_total: mean={df_out['Q_total'].mean():.1f}, "
          f"std={df_out['Q_total'].std():.1f}, "
          f"min={df_out['Q_total'].min():.1f}, "
          f"max={df_out['Q_total'].max():.1f}")
    print(f"  Q_expected: mean={df_out['Q_expected'].mean():.1f}")
    print(f"  Q_holding: mean={df_out['Q_holding'].mean():.1f}")
    
    # Check for consistency
    diff = (df_out['Q_total'] - (df_out['Q_expected'] + df_out['Q_holding'])).abs()
    if diff.max() > 1e-6:
        print(f"\n  Warning: Inconsistency detected (max diff={diff.max():.2e})")
    else:
        print(f"\n  ✓ Q = Q_expected + Q_holding (verified)")

if __name__ == "__main__":
    main()
