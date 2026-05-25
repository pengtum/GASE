# 04 — Real-World Experiment (paper §4.3 + §5.3)

Reproduces **Table 4** (cross-dataset means for the 6-algorithm summary), **Figure 6** (the compact 10-algorithm, 4-domain SSR Reduction figure), **Figure S1** (full per-dataset breakdown), **Figure S2** (p-sensitivity), and **Figures 7 + S4 + S5 + S6** (regime / coefficient / dominant-covariate maps).

## Datasets — `all_datasets/`

Eleven real-world datasets across four application domains:

| Domain | Dataset | Target | Source |
|---|---|---|---|
| Climate | ERA5 | 2-m air temperature | Muñoz Sabater et al. 2021 |
| Health | Arthritis, BPHigh, Cancer, Asthma, Depression, Diabetes, Obesity, Stroke (8 datasets) | Prevalence of each disease | CDC PLACES (Greenlund 2022) |
| Hydrology | CAMELS | Mean daily discharge | Addor 2017 |
| Politics | US Voting 2021 | Democratic vote share | Fotheringham 2023 |

Each dataset is a folder containing the cleaned (X, y, geometry) data plus a `metadata.json`. The US administrative boundary file is `us_states_conus_5070.gpkg`.

## Runners

| Script | What it does |
|---|---|
| `run_real_world_baselines.py` | Runs the 4 classical baselines on all 11 datasets |
| `run_real_world_evolved.py` | Runs the 6 K-Models variants (frozen, no further evolution) on all 11 datasets |
| `run_p_sensitivity.py` | Sweeps the number of regions p ∈ {3..12}; produces Figure S2 input |
| `run_real_world_analysis.py` / `run_analysis_v2.py` | Post-processing helpers |
| `comprehensive_ranking.py` | Per-dataset win counting |

## Outputs

- `results_baselines/algorithm_comparison.csv` — 4 baselines × 11 datasets × 4 metrics
- `results_evolved/algorithm_comparison.csv` — 6 K-Models variants × 11 datasets × 4 metrics
- per-dataset diagnostic dumps inside the same folders

These two CSVs are the canonical sources of all numbers in **Table 4** and **Figure 6**.

## Reproducibility

The evolved algorithms in `02_case_study_algorithms/evolved/` are **frozen** — no further evolution happens on real data. Each variant is imported as a Python module and its `run_algorithm()` function is called once per dataset, using the same target number of regions p inferred from the administrative-unit structure (typically p = 4..6) and the same 6-nearest-neighbor spatial weights matrix for all algorithms.

---

## Note on `US_Forest_FIA.csv`

This file is included for reproducibility of preliminary experiments but is **not used in the final paper** (paper uses 11 datasets; FIA was excluded after pre-screening). Skip it for the published results.
