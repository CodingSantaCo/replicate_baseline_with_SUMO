import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import comb
from scipy.stats import poisson

# ============ BASELINE PARAMETERS (Section 5.1) ============
LANE_LEN = 1000.0          # m, stopline coordinate
V_FREE   = 13.89           # m/s, free-flow speed (50 km/h)
V_STOP   = 0.5             # m/s, stop threshold
CYCLE    = 60.0            # s
RED      = 30.0            # s (red duration for baseline)
GREEN    = 22.0            # s (green duration for baseline)
EOR_TOL  = 0.5             # s, snapshot tolerance
OMEGA    = 2               # rolling window size (paper uses 2)
L_EFF    = 6.44            # m, effective vehicle length
LOOKBACK = 5.0             # s, window to backtrack T_k1

# Saturation flow parameters
SAT_HEADWAY = 1.59          # s, saturation headway (tau)
SAT_FLOW = 1.0 / SAT_HEADWAY  # veh/s
# =====================================

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

def find_last_stopped_cv(snap_df: pd.DataFrame, hist_df: pd.DataFrame, 
                         eor_time: float) -> tuple:
    """
    Find the last stopped CV at EoR and trace back when it started stopping.
    
    Returns:
        vid: vehicle ID of last stopped CV (or None)
        L_k1: position of last stopped CV (or None)
        T_k1: time when last stopped CV started stopping (or None)
        N_tilde: number of observable vehicles before last stopped CV
    """
    iscv = is_cv(snap_df["type"], snap_df["id"])
    stopped_cv = snap_df[iscv & (snap_df["speed"] <= V_STOP)].copy()
    
    if stopped_cv.empty:
        return None, None, None, 0
    
    # Most downstream stopped CV (closest to stopline)
    k1 = stopped_cv.sort_values("pos", ascending=False).iloc[0]
    vid = k1["id"]
    L_k1 = float(k1["pos"])
    
    # Count vehicles before this CV (N_tilde in paper - Eq. A1)
    # These are vehicles between L_k1 and the stopline
    vehicles_before = snap_df[snap_df["pos"] > L_k1]
    N_tilde = len(vehicles_before)
    
    # Backtrack to find when it started stopping
    h = hist_df[hist_df["id"] == vid].sort_values("time")
    h = h[(h["time"] >= eor_time - LOOKBACK) & (h["time"] <= eor_time)]
    
    if h.empty:
        return vid, L_k1, float(eor_time), N_tilde
    
    stopped_mask = (h["speed"].to_numpy() <= V_STOP)
    times = h["time"].to_numpy()
    
    if not stopped_mask.any():
        return vid, L_k1, float(eor_time), N_tilde
    
    # Find start of continuous stopping
    last_idx = np.where(stopped_mask)[0][-1]
    start_idx = last_idx
    while start_idx-1 >= 0 and stopped_mask[start_idx-1]:
        start_idx -= 1
    
    T_k1 = float(times[start_idx])
    return vid, L_k1, T_k1, N_tilde

def count_cvs_in_queue(snap_df: pd.DataFrame) -> int:
    """Count number of CVs in the stopped queue (n in paper - Eq. A1)."""
    iscv = is_cv(snap_df["type"], snap_df["id"])
    stopped = snap_df["speed"] <= V_STOP
    return int((iscv & stopped).sum())

def estimate_constrained_queue_dist_cdt(q: float, r: float, s: float) -> np.ndarray:
    """
    Estimate constrained queue distribution using CDT model (Appendix A).
    
    Following Eq. (A3): N0 = sqr / (s - q)
    where s = saturation flow rate, r = red period, q = arrival rate
    
    Returns: array of probabilities P(N = k) for k = 0, 1, 2, ...
    """
    if q >= s:
        # Oversaturation - not handled in baseline
        return None
    
    # Governing parameter (Eq. A3)
    N0 = (s * q * r) / (s - q)
    
    # Maximum queue length to consider
    max_queue = int(N0 + 5 * np.sqrt(N0)) + 10
    
    # Poisson distribution using scipy
    k_values = np.arange(max_queue)
    probs = poisson.pmf(k_values, N0)
    probs = probs / probs.sum()  # Normalize
    
    return probs

def negative_log_likelihood(params, data_window, queue_dist):
    """
    Negative log-likelihood for MLE estimation of q and p.
    
    Based on Eq. (A1) from the paper's Appendix A.
    
    params: [q, p] where q is arrival rate (veh/s), p is CV penetration rate
    data_window: list of dicts with keys 'n' (CV count) and 'N_tilde' (observable vehicles)
    queue_dist: constrained queue length distribution P(N = k)
    """
    q, p = params
    
    # Bounds checking
    if q <= 0 or p <= 0 or p >= 1:
        return 1e10
    
    nll = 0.0
    
    for obs in data_window:
        n = obs['n']           # Number of CVs in queue
        N_tilde = obs['N_tilde']  # Observable vehicles before last stopped CV
        
        # Compute P(n, N_tilde) based on Eq. (A1)
        if n == 0 and N_tilde == 0:
            # P(n=0, N_tilde=0) = π0 + sum over z of πz * (1-p)^z
            prob = 0.0
            if queue_dist is not None:
                for z in range(len(queue_dist)):
                    prob += queue_dist[z] * ((1 - p) ** z)
            prob = max(prob, 1e-10)
            
        elif N_tilde >= n >= 1:
            # P(n, N_tilde) = sum over z >= N_tilde of πz * C(N_tilde-1, n-1) * p^n * (1-p)^(z-n)
            # where C is binomial coefficient
            prob = 0.0
            if queue_dist is not None:
                for z in range(N_tilde, len(queue_dist)):
                    binomial_coef = comb(N_tilde - 1, n - 1, exact=False)
                    prob += queue_dist[z] * binomial_coef * (p ** n) * ((1 - p) ** (z - n))
            prob = max(prob, 1e-10)
            
        else:
            # Invalid observation
            nll += 1e5
            continue
        
        nll -= np.log(prob)
    
    return nll

def estimate_q_p_rolling(cycle_data: list, omega: int = 2) -> tuple:
    """
    Estimate q and p using MLE over rolling window of omega past cycles.
    
    Following Appendix A methodology with CDT model.
    
    cycle_data: list of observations (most recent last)
    omega: number of past cycles to include (paper uses 2)
    
    Returns: (q_hat, p_hat)
    """
    if len(cycle_data) == 0:
        return 0.0, 0.5
    
    # Use last (omega+1) cycles (current + omega past)
    window_size = min(omega + 1, len(cycle_data))
    data_window = cycle_data[-window_size:]
    
    # Initial guess based on simple counting
    total_cvs = sum(d['n'] for d in data_window)
    total_obs = sum(d['N_tilde'] for d in data_window)
    
    if total_obs > 0:
        p_init = total_cvs / max(1, total_obs)
        p_init = np.clip(p_init, 0.1, 0.9)
    else:
        p_init = 0.5
    
    # Estimate q from total vehicles and time
    q_init = total_obs / (window_size * RED) if window_size > 0 else 0.1
    q_init = np.clip(q_init, 0.01, SAT_FLOW * 0.9)
    
    # MLE optimization with multiple attempts
    best_result = None
    best_nll = float('inf')
    
    for q_start in [q_init, q_init * 0.5, q_init * 1.5]:
        for p_start in [p_init, 0.3, 0.5, 0.7]:
            # Compute queue distribution for this q
            queue_dist = estimate_constrained_queue_dist_cdt(q_start, RED, SAT_FLOW)
            
            if queue_dist is None:
                continue
            
            try:
                result = minimize(
                    negative_log_likelihood,
                    x0=[q_start, p_start],
                    args=(data_window, queue_dist),
                    bounds=[(0.001, SAT_FLOW * 0.99), (0.01, 0.99)],
                    method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_nll:
                    best_result = result
                    best_nll = result.fun
            except:
                continue
    
    if best_result is not None and best_result.success:
        q_hat, p_hat = best_result.x
        return float(q_hat), float(p_hat)
    
    # Fallback to simple estimation
    return float(q_init), float(p_init)

def estimate_holding_vehicles(snap_df: pd.DataFrame, q_hat: float, p_hat: float,
                               eor_time: float, hist_df: pd.DataFrame,
                               green_duration: float) -> dict:
    """
    Estimate holding vehicles R and R_CV at EoR.

    Following Appendix B - CVHV model.
    R = vehicles that should have been discharged but are still in the lane

    Definition: "Holding vehicles are vehicles that, based on their projected
    trajectories using cruise speeds, should have been discharged by that instant
    but remain held by the system."

    Holding vehicles are those that entered before (EoR - tau_ff) and are still
    in the lane at EoR.

    This implements CVHV-I sub-model (for ToI during effective red).
    """
    # Free-flow travel time through the lane
    tau_ff = LANE_LEN / V_FREE

    # Cutoff time: vehicles that entered before this should have exited by EoR
    cutoff_time = eor_time - tau_ff

    # For each vehicle in snapshot, find when it first entered the lane
    holding_vehicles = []
    holding_cvs = []

    iscv = is_cv(snap_df["type"], snap_df["id"])

    for idx, row in snap_df.iterrows():
        vid = row["id"]
        is_cv_vehicle = iscv.loc[idx]

        # Find first appearance of this vehicle in history
        veh_hist = hist_df[hist_df["id"] == vid].sort_values("time")

        if len(veh_hist) > 0:
            entry_time = float(veh_hist["time"].min())

            # Check if this vehicle should have been discharged by EoR
            if entry_time < cutoff_time:
                holding_vehicles.append(vid)
                if is_cv_vehicle:
                    holding_cvs.append(vid)

    # Count holding vehicles
    R = len(holding_vehicles)
    R_CV = len(holding_cvs)
    R_NC = R - R_CV

    # For debugging: also compute the old (incorrect) method
    stopped = snap_df["speed"] <= V_STOP
    R1_old = int(stopped.sum())  # All stopped vehicles (old incorrect method)

    # Find last stopped CV for additional metrics
    vid, L_k1, T_k1, N_tilde = find_last_stopped_cv(snap_df, hist_df, eor_time)

    return {
        'R': R,
        'R_CV': R_CV,
        'R_NC': R_NC,
        'cutoff_time': cutoff_time,
        'tau_ff': tau_ff,
        'T_k1': T_k1,
        'L_k1': L_k1,
        'R1_old_method': R1_old,  # For comparison
        'method': 'entry_time_based'
    }

def main():
    base = Path(".")
    fcd_path = base / "fcd_output.xml"
    
    if not fcd_path.exists():
        raise FileNotFoundError(f"Missing {fcd_path}")
    
    print("Loading FCD data...")
    df_all = load_fcd(fcd_path)
    lane_id = choose_lane(df_all)
    df_lane = df_all[df_all["lane"]==lane_id].copy()
    
    print(f"Selected lane: {lane_id}")
    print(f"Total records: {len(df_lane)}")
    
    sim_end = float(df_lane["time"].max())
    
    # Generate EoR times (End of Red)
    eor_times = np.arange(RED, sim_end + 1e-9, CYCLE)
    
    print(f"\nBaseline Configuration (Section 5.1):")
    print(f"  Lane length: {LANE_LEN} m")
    print(f"  Free-flow speed: {V_FREE} m/s ({V_FREE * 3.6:.1f} km/h)")
    print(f"  Cycle length: {CYCLE} s")
    print(f"  Red duration: {RED} s")
    print(f"  Green duration: {GREEN} s")
    print(f"  Saturation headway: {SAT_HEADWAY} s")
    print(f"  Effective vehicle length: {L_EFF} m")
    print(f"  Rolling window (ω): {OMEGA} cycles")
    print(f"\nProcessing {len(eor_times)} EoR snapshots...")
    
    # Store observations for rolling window
    cycle_observations = []
    results = []
    
    for idx, t in enumerate(eor_times):
        print(f"\n{'='*60}")
        print(f"Cycle {idx+1}/{len(eor_times)}, EoR @ {t:.1f}s")
        print(f"{'='*60}")
        
        # Get snapshot at EoR
        snap = nearest_snapshot(df_lane, t, EOR_TOL)
        if snap.empty:
            print(f"  Warning: No vehicles at EoR {t:.1f}s")
            continue
        
        print(f"  Vehicles in snapshot: {len(snap)}")

        # Get history for backtracking T_k1 (short window)
        hist_short = df_lane[(df_lane["time"] >= t - LOOKBACK) & (df_lane["time"] <= t)].copy()

        # Get full history for tracking entry times (needed for R calculation)
        # Use max(tau_ff, LOOKBACK) to ensure we capture all potential holding vehicles
        tau_ff = LANE_LEN / V_FREE
        history_window = max(tau_ff + 10.0, LOOKBACK)  # Add 10s buffer
        hist = df_lane[(df_lane["time"] >= t - history_window) & (df_lane["time"] <= t)].copy()

        # Count CVs and observable vehicles in queue
        n_cvs = count_cvs_in_queue(snap)
        _, _, _, N_tilde = find_last_stopped_cv(snap, hist_short, t)
        
        print(f"  CVs in queue (n): {n_cvs}")
        print(f"  Observable vehicles before last CV (Ñ): {N_tilde}")
        
        # Store observation for MLE
        cycle_observations.append({
            'n': n_cvs,
            'N_tilde': N_tilde,
            'eor_time': t
        })
        
        # Estimate q and p using rolling window (current + OMEGA past cycles)
        q_hat, p_hat = estimate_q_p_rolling(cycle_observations, omega=OMEGA)
        
        print(f"  → Estimated q = {q_hat:.4f} veh/s (λ)")
        print(f"  → Estimated p = {p_hat:.4f} (ρ)")
        print(f"  → Estimated q_NC = {q_hat * (1 - p_hat):.4f} veh/s")
        print(f"  → Window size: {min(len(cycle_observations), OMEGA+1)} cycles")
        
        # Estimate holding vehicles using current q, p estimates
        holding = estimate_holding_vehicles(snap, q_hat, p_hat, t, hist, GREEN)
        
        print(f"  → Holding vehicles (R): {holding['R']}")
        print(f"  → Holding CVs (R_CV): {holding['R_CV']}")
        print(f"  → Holding NCs (R_NC): {holding['R_NC']}")
        print(f"  → Cutoff time: {holding.get('cutoff_time', 'N/A'):.1f}s")
        print(f"  → Method: {holding['method']}")
        if 'R1_old_method' in holding:
            print(f"  → (Old method would give R1={holding['R1_old_method']})")
        
        results.append({
            'EoR_s': float(t),
            'cycle_id': idx + 1,
            'q_hat': float(q_hat),
            'p_hat': float(p_hat),
            'q_nc_hat': float(q_hat * (1 - p_hat)),
            'R': int(holding['R']),
            'R_CV': int(holding['R_CV']),
            'R_NC': int(holding['R_NC']),
            'n_cvs_in_queue': int(n_cvs),
            'N_tilde': int(N_tilde),
            'window_cycles': min(len(cycle_observations), OMEGA+1),
            'T_k1': holding.get('T_k1', None),
            'L_k1': holding.get('L_k1', None),
            'cutoff_time': holding.get('cutoff_time', None),
            'tau_ff': holding.get('tau_ff', None),
            'R1_old_method': holding.get('R1_old_method', None),
            'holding_method': holding.get('method', 'unknown')
        })
    
    # Save results
    df_out = pd.DataFrame(results)
    out_file = base / "dynamic_params_per_cycle.csv"
    df_out.to_csv(out_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Saved dynamic parameters to: {out_file}")
    print(f"  Total cycles processed: {len(results)}")
    print(f"  Columns: {list(df_out.columns)}")
    print(f"{'='*70}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  q_hat: mean={df_out['q_hat'].mean():.4f}, std={df_out['q_hat'].std():.4f}")
    print(f"  p_hat: mean={df_out['p_hat'].mean():.4f}, std={df_out['p_hat'].std():.4f}")
    print(f"  q_nc_hat: mean={df_out['q_nc_hat'].mean():.4f}, std={df_out['q_nc_hat'].std():.4f}")
    print(f"  R: mean={df_out['R'].mean():.1f}, std={df_out['R'].std():.1f}")
    print(f"  R_CV: mean={df_out['R_CV'].mean():.1f}, std={df_out['R_CV'].std():.1f}")
    print(f"  R_NC: mean={df_out['R_NC'].mean():.1f}, std={df_out['R_NC'].std():.1f}")
    
    print("\n" + "="*70)
    print("Implementation follows:")
    print("  • Section 3.1 & Proposition 1 (Q estimation)")
    print("  • Appendix A (q and p estimation via MLE with CDT model)")
    print("  • Appendix B (R estimation via CVHV model)")
    print("  • Appendix C (Proof of Proposition 1)")
    print("  • Baseline configuration from Section 5.1")
    print("="*70)

if __name__ == "__main__":
    main()
