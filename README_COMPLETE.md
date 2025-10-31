# CVVL-S Baseline Replication Using SUMO

This repository replicates the **baseline case from Table 1** and the **corresponding subfigure in Figure 7** from Section 5 of the paper:

**"Real-time vehicle location estimation in signalized networks using partial connected vehicle trajectory data"**
by Shaocheng Jia, S.C. Wong, and Wai Wong (Transportation Research Part B, 2025)

## Baseline Case Parameters

- **Lane Type**: Source lane (lane7_0)
- **Red period (r)**: 30s
- **V/C ratio**: 0.5
- **CV penetration rate (p)**: 0.4
- **Time of Interest (ToI)**: End of Red (EoR)
- **Simulation duration**: 3600s (1 hour)

## Implementation Status

### ✅ Completed Components

1. **Holding Vehicle Estimation** (`recompute_R_CVHV_fixed.py`)
   - Implements CVHV model to estimate R (holding vehicles)

2. **Total NC Count Estimation** (`estimate_Q_from_CV_only.py`)
   - Implements Section 3.1, Proposition 1
   - Estimates Q = q̄(1-p̄) × l/vf + R - RC

3. **Lane Segmentation** (`segments_AB_resegmented_v05_entr.py`)
   - Divides lane into Type A (stopped) and Type B (moving) segments
   - Based on CV locations and speeds

4. **Type A NC Distribution** (`compute_A_eq34_ROUND.py`)
   - Implements Section 3.2, Proposition 2 Part A
   - Computes QA_i and NC positions in stopped segments (Eqs. 3-4)

5. **Type B NC Distribution** (`compute_B_distribution.py`)
   - Implements Section 3.2, Proposition 2 Part B
   - Computes QB_i and NC positions in moving segments (Eqs. 5-11)
   - Includes safe headway constraints and feasible space calculation

6. **Main Pipeline** (`run_cvvl_s_pipeline.py`)
   - Integrates all components in correct sequence
   - Combines Type A and Type B estimates

7. **Evaluation** (`evaluate_baseline.py`)
   - Computes Precision, Recall, and F1 score
   - Compares against ground truth from SUMO

8. **Visualization** (`generate_figure7.py`)
   - Generates Figure 7 style plots
   - Shows ground truth vs estimated positions

## Quick Start

### 1. Run SUMO Simulation

```bash
sumo -c simulation.sumocfg
```

This generates `fcd_output.xml` with vehicle trajectories for 3600 seconds.

### 2. Run Complete CVVL-S Pipeline

```bash
python run_cvvl_s_pipeline.py
```

This executes all steps in sequence:
- Estimates holding vehicles (R)
- Estimates total NC count (Q)
- Divides lane into segments
- Distributes NCs in Type A segments
- Distributes NCs in Type B segments
- Combines all estimates

### 3. Evaluate Results

```bash
python evaluate_baseline.py
```

Computes metrics and compares with Table 1 baseline:
- Precision
- Recall
- F1 score

### 4. Generate Visualizations

```bash
python generate_figure7.py
```

Creates:
- `figure7_baseline.png` - Main visualization
- `figure7_summary.png` - Summary statistics

## Expected Results (from Paper Table 1 Baseline)

| Method | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| EVLS   | 47%       | 71%    | 57%      |
| **CVVL-S** | **58%** | **76%** | **66%** |

Our implementation should achieve results close to the CVVL-S row.

## Output Files

| File | Description |
|------|-------------|
| `holding_EoR_fixed.csv` | Estimated holding vehicles at each EoR |
| `eor_Q_estimated.csv` | Total NC count estimates |
| `segments_AB_resegmented_v05_entr.csv` | Lane segments (Type A and B) |
| `A_counts_ROUND.csv` | NC counts in Type A segments |
| `A_positions_ROUND.csv` | NC positions in Type A segments |
| `B_counts.csv` | NC counts in Type B segments |
| `B_positions.csv` | NC positions in Type B segments |
| `all_NC_positions_estimated.csv` | **Combined final estimates** |
| `evaluation_results.csv` | Detailed evaluation metrics |
| `figure7_baseline.png` | Visualization of estimates |

## File Descriptions

### Core Algorithm Files

- **`recompute_R_CVHV_fixed.py`**: Estimates R (holding vehicles) using CV trajectory data
- **`estimate_Q_from_CV_only.py`**: Estimates total number of NCs using flow and penetration rate
- **`segments_AB_resegmented_v05_entr.py`**: Divides lane into Type A/B segments based on CV speeds
- **`compute_A_eq34_ROUND.py`**: Distributes NCs in Type A (stopped) segments
- **`compute_B_distribution.py`**: Distributes NCs in Type B (moving) segments

### Pipeline and Evaluation

- **`run_cvvl_s_pipeline.py`**: Main script to run complete pipeline
- **`evaluate_baseline.py`**: Computes precision, recall, F1 vs ground truth
- **`generate_figure7.py`**: Creates visualizations

### SUMO Files

- **`simulation.sumocfg`**: SUMO configuration (3600s simulation)
- **`network_*.xml`**: Network definition files
- **`routes_rou.xml`**: Vehicle routes with CV/NC designation

## Key Parameters

```python
# From paper Section 5.1
LANE_LEN = 1000.0  # meters
V_FREE = 13.89     # m/s (50 km/h cruise speed)
CYCLE = 60.0       # seconds
RED = 30.0         # seconds (baseline case)
L_E = 6.44         # meters (effective vehicle length)
DELTA_T = 2.04     # seconds (saturation headway)
TOLERANCE = 10.0   # meters (evaluation threshold)

# Baseline case
CV_VPH = 166       # CVs per hour
NC_VPH = 249       # NCs per hour
P_CV = 0.4         # Penetration rate
V/C = 0.5          # Volume-to-capacity ratio
```

## Algorithm Implementation Details

### Section 3.1: Total NC Count (Proposition 1)

```
Q = q̄(1-p̄) × l/vf + R - RC
```

where:
- q̄ = estimated arrival rate
- p̄ = estimated CV penetration rate
- l/vf = free-flow travel time
- R = holding vehicles
- RC = holding CVs

### Section 3.2: NC Distribution (Proposition 2)

#### Type A Segments (Stopped Traffic)
```
QA_i = ⌊(Li - le - Li+1) / le⌉  (for i > 0)
```
NCs uniformly distributed with spacing le.

#### Type B Segments (Moving Traffic)
1. Compute max capacity: `Q̃B_i` using safe headway Δt
2. Scale allocation: `QB_i = ⌊ρQ̃B_i⌉ + excess`
3. Compute feasible space: `[L(l)_i, L(u)_i]`
4. Distribute uniformly within feasible space

## Troubleshooting

### SUMO not found
```bash
# Install SUMO
sudo apt-get install sumo sumo-tools sumo-doc
```

### Missing Python packages
```bash
pip install pandas numpy matplotlib
```

### No CVs detected
Check `routes_rou.xml` - vehicles should have `type="CV"` or IDs containing "cv".

## References

Jia, S., Wong, S.C., Wong, W., 2025. Real-time vehicle location estimation in signalized networks using partial connected vehicle trajectory data. Transportation Research Part B 200, 103292.

## Contact

For questions about the paper: shaocjia@connect.hku.hk

For questions about this implementation: See paper repository
