
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

FCD_FILE  = "fcd_output.xml"
EOR_FILE  = "holding_EoR_fixed.csv"   # needs column EoR_s
OUT_FILE  = "segments_AB_resegmented_v05_entr.csv"

LANE_ID   = "lane7_0"
STOP_POS  = 1000.0
ENTR_POS  = 0.0
TOL_T     = 0.5

STOP_TH   = 0.5   # < 0.5 m/s => stopped

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
                "type":  a.get("type",""),
            })
        if event == "end":
            elem.clear()
    return pd.DataFrame(rows)

def nearest_snapshot(df: pd.DataFrame, t: float, tol: float) -> pd.DataFrame:
    snap = df[df["time"].between(t-tol, t+tol)].copy()
    if snap.empty: return snap
    snap["dt"] = (snap["time"]-t).abs()
    snap = snap.sort_values(["id","dt"]).drop_duplicates("id", keep="first")
    return snap

def main():
    base = Path(".")
    if not (base/"fcd_output.xml").exists() or not (base/"holding_EoR_fixed.csv").exists():
        raise FileNotFoundError("Missing fcd_output.xml or holding_EoR_fixed.csv")

    fcd  = load_fcd(base/"fcd_output.xml")
    eors = pd.read_csv(base/"holding_EoR_fixed.csv")
    if "EoR_s" not in eors.columns:
        raise ValueError("holding_EoR_fixed.csv must contain column EoR_s")

    fcd_lane = fcd[fcd["lane"]==LANE_ID].copy()

    out_rows = []
    for t in sorted(eors["EoR_s"].dropna().unique().tolist()):
        snap = nearest_snapshot(fcd_lane, float(t), TOL_T)
        if snap.empty:
            continue

        # CVs in downstream->upstream order
        is_cv = snap["type"].str.contains("CV", case=False, na=False) | snap["id"].astype(str).str.startswith("CV")
        cvs = snap[is_cv].copy().sort_values("pos", ascending=False).reset_index(drop=True)

        # anchors: stopline (stopped), CVs (stopped if v<0.5), entrance (treated as moving -> not stopped)
        anchors = [("stopline", STOP_POS, 0.0, True)]
        for _, r in cvs.iterrows():
            v = float(r["speed"])
            pos = float(r["pos"])
            is_stopped = (v < STOP_TH)
            anchors.append((str(r["id"]), pos, v, is_stopped))
        anchors.append(("entrance", ENTR_POS, 0.0, False))

        # build segments between consecutive anchors
        for i in range(len(anchors)-1):
            d_id, d_pos, d_v, d_stop = anchors[i]
            u_id, u_pos, u_v, u_stop = anchors[i+1]
            down_m = max(d_pos, u_pos)
            up_m   = min(d_pos, u_pos)
            seg_type = "A" if (d_stop and u_stop) else "B"
            out_rows.append({
                "EoR_s": float(t),
                "lane": LANE_ID,
                "seg_type": seg_type,
                "down_m": float(down_m),
                "up_m": float(up_m),
                "down_anchor": d_id,
                "down_speed": float(d_v),
                "up_anchor": u_id,
                "up_speed": float(u_v)
            })

    pd.DataFrame(out_rows, columns=[
        "EoR_s","lane","seg_type","down_m","up_m","down_anchor","down_speed","up_anchor","up_speed"
    ]).to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE}")

if __name__ == "__main__":
    main()
