
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- Tunables (you can change) ----------------
LANE_LEN = 1000.0         # m, stopline coordinate
V_FREE   = 13.89          # m/s, free-flow speed
CYCLE    = 60.0           # s
RED      = 30.0           # s (EoR offset)
EOR_TOL  = 0.5            # s, snapshot tolerance (not critical here)
HORIZON_H = 300.0         # s, horizon to estimate arrivals (CV-only)
P_MODE   = "fixed"        # "fixed" or "from_routes_csv" or "from_fcd_window"
P_FIXED  = 166.0/(166.0+249.0)  # default 0.4 if you used 166 CV / 249 NC per hour in routes
# If P_MODE == "from_routes_csv", set ROUTES_CV_VPH and ROUTES_NC_VPH below
ROUTES_CV_VPH = 166.0
ROUTES_NC_VPH = 249.0
# Input file names (same folder)
FCD_FILE  = "fcd_output.xml"
R_FILE_TRY = ["holding_EoR_breakdown_consistent.csv",
              "holding_EoR_breakdown.csv",
              "holding_EoR_fixed.csv"]
OUT_CSV  = "eor_Q_estimated.csv"
# -----------------------------------------------------------

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

def is_cv(type_series: pd.Series, id_series: pd.Series) -> pd.Series:
    t = type_series.fillna("").astype(str).str.upper()
    iscv = (t == "CV")
    undecided = ~((t=="CV") | (t=="NC"))
    if undecided.any():
        ids = id_series[undecided].astype(str).str.lower()
        iscv.loc[undecided] = ids.str.contains("cv") | ids.str.contains("f_cv")
    return iscv.astype(bool)

def detect_cv_entries(df_lane: pd.DataFrame, horizon_s: float, t_center: float) -> int:
    """Count CVs that ENTERED the lane during (t_center - horizon_s, t_center].
    We detect the first time each CV id appears on this lane and count it if within the window.
    """
    iscv_all = is_cv(df_lane["type"], df_lane["id"])
    df_cv = df_lane[iscv_all].copy()
    first_seen = df_cv.groupby("id", as_index=False)["time"].min()
    return int(((first_seen["time"] > t_center - horizon_s) & (first_seen["time"] <= t_center)).sum())

def estimate_pbar(df_lane_window: pd.DataFrame) -> float:
    iscv = is_cv(df_lane_window["type"], df_lane_window["id"])
    denom = len(iscv)
    if denom == 0:
        return P_FIXED
    return float(iscv.sum()) / float(denom)

def load_R_breakdown(base: Path) -> pd.DataFrame:
    for name in R_FILE_TRY:
        p = base / name
        if p.exists():
            df = pd.read_csv(p)
            required = {"EoR_s","R"}
            if not required.issubset(df.columns):
                continue
            if "R_CV" not in df.columns:
                if "R_NC" in df.columns:
                    df["R_CV"] = df["R"] - df["R_NC"]
                else:
                    df["R_CV"] = 0
            return df[["EoR_s","R","R_CV"]].copy()
    raise FileNotFoundError("Cannot find R breakdown file among: " + ", ".join(R_FILE_TRY))

def main():
    base = Path(".")
    fcd_path = base / FCD_FILE
    if not fcd_path.exists():
        raise FileNotFoundError(f"Missing {FCD_FILE}")
    df_all = load_fcd(fcd_path)
    lane_id = choose_lane(df_all)
    df_lane = df_all[df_all["lane"]==lane_id].copy()

    sim_end = float(df_lane["time"].max())
    eor_times = np.arange(RED, sim_end + 1e-9, CYCLE)

    dfR = load_R_breakdown(base)

    rows = []
    for t in eor_times:
        window = df_lane[(df_lane["time"] > t - HORIZON_H) & (df_lane["time"] <= t)].copy()

        if P_MODE == "fixed":
            p_hat = float(P_FIXED)
        elif P_MODE == "from_routes_csv":
            p_hat = float(ROUTES_CV_VPH / (ROUTES_CV_VPH + ROUTES_NC_VPH))
        elif P_MODE == "from_fcd_window":
            p_hat = estimate_pbar(window)
        else:
            p_hat = float(P_FIXED)

        n_cv_enter = detect_cv_entries(df_lane, HORIZON_H, t)
        q_cv_hat = n_cv_enter / HORIZON_H   # veh/s
        q_hat = q_cv_hat / max(p_hat, 1e-6) # total flow from CV-only
        q_nc_hat = q_hat * (1.0 - p_hat)

        r_row = dfR[dfR["EoR_s"].round(3) == round(float(t),3)]
        if r_row.empty:
            idx = (dfR["EoR_s"] - float(t)).abs().idxmin()
            r_row = dfR.loc[[idx]]
        R = float(r_row.iloc[0]["R"])
        R_CV = float(r_row.iloc[0]["R_CV"])

        Q = q_nc_hat * (LANE_LEN / V_FREE) + (R - R_CV)

        rows.append({
            "EoR_s": float(t),
            "lane": lane_id,
            "Q_est": float(Q),
            "q_hat_total_veh_per_s": float(q_hat),
            "p_hat": float(p_hat),
            "q_nc_hat": float(q_nc_hat),
            "travel_time_ff": float(LANE_LEN / V_FREE),
            "R": float(R),
            "R_CV": float(R_CV),
            "R_minus_Rc": float(R - R_CV),
            "n_cv_enter_in_H": int(n_cv_enter)
        })

    out = pd.DataFrame(rows).sort_values("EoR_s")
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

if __name__ == "__main__":
    main()
