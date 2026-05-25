# 03 — Simulation Experiment (paper §4.2 + §5.2)

Reproduces **Table 3** (per-pattern SSR / RandI of all 10 algorithms on 150 simulated grids).

## Datasets — `datasets/`

150 synthetic 25×25 grids generated from a piecewise-linear regime data-generating process (paper §4.2.1):

- 50 grids per spatial pattern × 3 patterns = 150 grids
- Patterns: **Rectangular**, **Voronoi**, **Arbitrary**
- Each file `grid_{N}h.txt` contains the (X, y, true labels) for one grid at the high-noise level

The three training grids used during evolution (NOT used during held-out evaluation) are `grid_5h.txt`, `grid_75h.txt`, `grid_149h.txt` — one per pattern.

## Runners

| Script | What it does |
|---|---|
| `run_simulation_baselines.py` | Runs the 4 classical baselines (AZP, Region-K-Models, GWR-Skater, Skater-reg) on all 150 grids; writes `results/baseline_per_dataset.csv` + `baseline_per_pattern.csv` |
| `run_simulation_evolved.py` | Runs the 6 K-Models variants (seed + 5 evolved) on all 150 grids; writes `results/evolved_per_dataset.csv` + `evolved_per_pattern.csv` |
| `smoke_test_baselines.py` | Sanity check on a small subset of grids |

## Outputs — `results/`

| File | Content |
|---|---|
| `baseline_per_pattern.csv` | SSR, RandI, NMI, MAE, time means per pattern × 4 baselines |
| `evolved_per_pattern.csv` | SSR, RandI, NMI, MAE, time means per pattern × 6 K-Models variants |
| `baseline_per_dataset.csv` | Per-grid raw numbers (450 rows = 150 grids × 3 metrics-tracked) |
| `evolved_per_dataset.csv` | Per-grid raw numbers for K-Models variants |
| `ModelTime.xlsx` | Compiled Excel summary |
| `sim_*_run.log` | Run logs |

To reproduce **Table 3** in the paper, merge `baseline_per_pattern.csv` + `evolved_per_pattern.csv` and pivot by algorithm × pattern. The exact summary script is in `05_analysis_and_figures/_paper_summary.py`.
