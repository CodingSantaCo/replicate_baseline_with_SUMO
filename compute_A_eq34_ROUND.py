
import pandas as pd
import numpy as np
from math import floor
from pathlib import Path

IN_FILE = "segments_AB_resegmented_v05_entr.csv"
OUT_CNT = "A_counts_ROUND.csv"
OUT_POS = "A_positions_ROUND.csv"

STOP_POS = 1000.0   # stopline coordinate l
L_E      = 6.44      # average effective vehicle length
STOP_TH  = 0.5      # "stopped" threshold (only for sanity diagnostics)

def round_half_up(x: float) -> int:
    """Round to nearest, halves up (0.5 -> 1), matching the paper's ⌊·⌉ symbol."""
    return int(floor(x + 0.5))

def main():
    p = Path(IN_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{IN_FILE} not found")
    segs = pd.read_csv(p)

    # numeric coercion
    for c in ["EoR_s","up_m","down_m","up_speed","down_speed"]:
        if c in segs.columns:
            segs[c] = pd.to_numeric(segs[c], errors="coerce")

    # Use only A segments
    A = segs[segs["seg_type"].astype(str).str.upper()=="A"].copy()
    if A.empty:
        pd.DataFrame(columns=[
            "EoR_s","lane","seg_idx_A","Q_Ai","up_m","down_m","up_anchor","down_anchor","span_m","formula","note"
        ]).to_csv(OUT_CNT, index=False)
        pd.DataFrame(columns=["EoR_s","lane","seg_idx_A","j_in_seg","est_pos_m"]).to_csv(OUT_POS, index=False)
        print(f"Saved empty {OUT_CNT} and {OUT_POS} (no A segments)")
        return

    group_keys = ["EoR_s"] + (["lane"] if "lane" in A.columns else [])
    rows_cnt, rows_pos = [], []

    for key, g in A.groupby(group_keys, as_index=False):
        if isinstance(key, tuple):
            t = float(key[0])
            lane_id = key[1] if len(key)>1 else (g["lane"].iloc[0] if "lane" in g.columns else "lane7_0")
        else:
            t = float(key)
            lane_id = g["lane"].iloc[0] if "lane" in g.columns else "lane7_0"

        # sort downstream->upstream by down_m desc
        g = g.sort_values("down_m", ascending=False).reset_index(drop=True)

        # vectors
        L_down = g["down_m"].to_numpy(float)  # L_i
        L_up   = g["up_m"].to_numpy(float)    # L_{i+1}

        # sanity: ensure down_m >= up_m per segment
        for i in range(len(g)):
            if L_down[i] < L_up[i]:
                L_down[i], L_up[i] = L_up[i], L_down[i]

        for i in range(len(g)):
            seg = g.iloc[i]
            up_m = float(L_up[i]); down_m = float(L_down[i])
            up_a = seg.get("up_anchor",""); down_a = seg.get("down_anchor","")
            up_v = float(seg.get("up_speed",0.0)); down_v = float(seg.get("down_speed",0.0))

            # Eq.(D1)/(D2) — ROUND (half up)
            if i == 0:
                # i=0: Q = ⌊(l - L1)/le⌉
                span = (STOP_POS - up_m)
                Q = round_half_up(max(0.0, span / L_E))
                formula = "i=0: round((l - L1)/le)"
            else:
                # i>0: Q = ⌊(Li - le - Li+1)/le⌉
                span = (down_m - L_E - up_m)
                Q = round_half_up(max(0.0, span / L_E))
                formula = "i>0: round((Li - le - Li+1)/le)"

            # emit positions only if Q>0 (Eq.(D3)(D4))
            if Q > 0:
                if i == 0:
                    step = (STOP_POS - up_m) / Q
                    for j in range(1, Q+1):
                        x = up_m + j*step
                        rows_pos.append({"EoR_s": t, "lane": lane_id, "seg_idx_A": i, "j_in_seg": j, "est_pos_m": float(x)})
                else:
                    # positions uniformly between Li+1 and Li - le
                    numerator = (down_m - L_E - up_m)
                    step = numerator / Q
                    for j in range(1, Q+1):
                        x = up_m + j*step
                        rows_pos.append({"EoR_s": t, "lane": lane_id, "seg_idx_A": i, "j_in_seg": j, "est_pos_m": float(x)})

            note = ""
            # diagnostics: anchors-stopped status (does not gate formulas)
            if str(down_a).lower()!="stopline" and down_v >= 0.5:
                note += "down_m anchor not fully stopped; "
            if up_v >= 0.5:
                note += "up_m anchor not fully stopped; "

            rows_cnt.append({
                "EoR_s": t, "lane": lane_id, "seg_idx_A": i, "Q_Ai": int(Q),
                "up_m": up_m, "down_m": down_m, "up_anchor": up_a, "down_anchor": down_a,
                "span_m": float(span), "formula": formula, "note": note.strip()
            })

    pd.DataFrame(rows_cnt).to_csv(OUT_CNT, index=False)
    pd.DataFrame(rows_pos).to_csv(OUT_POS, index=False)
    print(f"Saved {OUT_CNT} and {OUT_POS}")

if __name__ == "__main__":
    main()
