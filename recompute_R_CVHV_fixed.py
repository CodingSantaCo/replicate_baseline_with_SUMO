
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

# ---- Parameters you can tweak ----
LANE_LEN = 1000.0          # stopline coordinate (m)
V_FREE   = 13.89           # free-flow speed (m/s) on the approach
V_STOP   = 0.5             # stop threshold (m/s)
EOR_TOL  = 0.5             # time tolerance when sampling snapshots
CYCLE    = 60.0            # s
RED      = 30.0            # s
LOOKBACK = 5.0             # s window to backtrack T_k1
# Flows from your SUMO routes (per hour):
CV_VPH   = 166
NC_VPH   = 249
# ----------------------------------

def load_fcd(path: Path) -> pd.DataFrame:
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
    lanes = df["lane"].dropna().unique().tolist()
    return "lane7_0" if "lane7_0" in lanes else df["lane"].value_counts().idxmax()

def nearest_snapshot(df: pd.DataFrame, t_star: float, tol: float) -> pd.DataFrame:
    snap = df[df["time"].between(t_star - tol, t_star + tol)].copy()
    if snap.empty: return snap
    snap["dt"] = (snap["time"] - t_star).abs()
    snap = snap.sort_values(["id","dt"]).drop_duplicates("id", keep="first")
    return snap[(snap["pos"] <= LANE_LEN) & (snap["pos"] >= 0.0)].copy()

def is_cv(type_series: pd.Series, id_series: pd.Series) -> pd.Series:
    t = type_series.fillna("").astype(str).str.upper()
    iscv = (t == "CV")
    undecided = ~((t=="CV") | (t=="NC"))
    if undecided.any():
        ids = id_series[undecided].astype(str).str.lower()
        iscv.loc[undecided] = ids.str.contains("cv") | ids.str.contains("f_cv")
    return iscv.astype(bool)

def find_last_stopped_cv_at_eor(snap_df, hist_df, eor_time, v_stop=0.5):
    # CVs stopped at EoR
    iscv = is_cv(snap_df["type"], snap_df["id"])
    stopped_cv = snap_df[iscv & (snap_df["speed"] <= v_stop)].copy()
    if stopped_cv.empty:
        return None, None, None
    # most downstream stopped CV
    k1 = stopped_cv.sort_values("pos", ascending=False).iloc[0]
    vid = k1["id"]; L_k1 = float(k1["pos"])

    h = hist_df[hist_df["id"] == vid].sort_values("time")
    h = h[(h["time"] >= eor_time - LOOKBACK) & (h["time"] <= eor_time)]
    if h.empty:
        return vid, L_k1, float(eor_time)
    stopped_mask = (h["speed"].to_numpy() <= v_stop)
    times = h["time"].to_numpy()
    if not stopped_mask.any():
        return vid, L_k1, float(eor_time)
    last_idx = np.where(stopped_mask)[0][-1]
    start_idx = last_idx
    while start_idx-1 >= 0 and stopped_mask[start_idx-1]:
        start_idx -= 1
    T_k1 = float(times[start_idx])
    return vid, L_k1, T_k1

def recompute_R(fcd_path: Path, out_csv: Path):
    df_all = load_fcd(fcd_path)
    lane_id = choose_lane(df_all)
    df = df_all[df_all["lane"]==lane_id].copy()

    # flows & rates
    q_total = (CV_VPH + NC_VPH) / 3600.0   # veh/s
    p_cv    = CV_VPH / (CV_VPH + NC_VPH)   # penetration
    tau_ff  = LANE_LEN / V_FREE            # free-flow travel time (s)

    sim_end = float(df["time"].max())
    eor_times = np.arange(RED, sim_end + 1e-9, CYCLE)

    rows = []
    for t in eor_times:
        snap = nearest_snapshot(df, t, EOR_TOL)
        if snap.empty: 
            continue
        hist = df[(df["time"] >= t - LOOKBACK) & (df["time"] <= t)].copy()
        vid, L_k1, T_k1 = find_last_stopped_cv_at_eor(snap, hist, t, v_stop=V_STOP)

        # R1 from geometry around stopline requires l_eff; if你已有R1计算，亦可合并。
        # 这里只用于绘图对比R2修正效果 => 直接把R1设为可观测的“停止NC + 停止CV”的数量近似（窗口内）
        # 若你已有严格R1实现，请替换这几行。
        stop_all = snap[snap["speed"] <= V_STOP]
        # 真实holding近似（停住的所有车；你也可以只取NC）：
        R1_obs = int(stop_all.shape[0])

        # R2 (fixed): only arrivals that WOULD HAVE REACHED stopline by EoR
        if T_k1 is None:
            T_k1 = t
        effective_window = max(0.0, (t - tau_ff) - T_k1)
        R2 = q_total * (1.0 - p_cv) * effective_window

        R_est = int(np.floor(R1_obs + R2 + 1e-9))
        rows.append({"EoR_s": t, "R": R_est, "R1_obs": R1_obs, "R2_float": R2, "T_k1": T_k1, "tau_ff": tau_ff})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    base = Path(".")
    recompute_R(base / "fcd_output.xml", base / "holding_EoR_fixed.csv")
