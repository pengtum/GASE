# SEGA: Self-Evolving Geospatial Algorithm — Code & Data Release

Companion code and data for the manuscript:

> **Toward Self-Evolving Geospatial Algorithms: An LLM-Driven Framework with a Case Study on Spatial Regime Discovery**

This release contains the full SEGA (Self-Evolving Geospatial Algorithm) framework implementation, all seed and evolved algorithm variants, both simulation and real-world experimental code, and the scripts to regenerate every figure and table in the paper.

---

## Quick start

```bash
# 1. Set up environment (Python 3.10+)
pip install -r requirements.txt

# 2. Reproduce the simulation experiment (~hours; uses local CPUs)
cd 03_simulation_experiment
python run_simulation_baselines.py        # 4 classical baselines × 150 grids
python run_simulation_evolved.py          # 6 K-Models variants × 150 grids

# 3. Reproduce the real-world experiment (~30 min)
cd ../04_real_world_experiment
python run_real_world_baselines.py        # 4 classical baselines × 11 datasets
python run_real_world_evolved.py          # 6 K-Models variants × 11 datasets

# 4. Generate all paper figures
cd ../05_analysis_and_figures
python generate_combined_bar_chart.py     # Figure 6 (main)
python generate_baseline_map_matrix.py    # Figure 7 (regime maps)
python generate_evolved_map_matrix.py     # Figure S4 (evolved variants)
# ... see 05_analysis_and_figures/README.md for the full mapping
```

---

## Folder structure

| Folder | Paper section | Contents |
|---|---|---|
| `01_framework/` | §3 (Methods) | SEGA's three modules: (1) two-loop architecture (via `geoevolve` Python package), (2) automatic code formatter, (3) simulation-driven training protocol |
| `02_case_study_algorithms/` | §4.4 (Comparison Methods) | The 10 algorithms compared in the paper: the 2KModels seed, 5 evolved K-Models variants, and 4 classical regionalization baselines, plus the round-by-round trajectory snapshots |
| `03_simulation_experiment/` | §4.2 + §5.2 | 150 simulated grids (Rectangular / Voronoi / Arbitrary) and the runner scripts that produce Table 3 |
| `04_real_world_experiment/` | §4.3 + §5.3 | 11 real-world datasets (ERA5 climate, 8 health datasets, CAMELS hydrology, US voting) and the runner scripts that produce Table 4 + Figure 6 + p-sensitivity (Figure S2) |
| `05_analysis_and_figures/` | All figures + tables | Scripts that turn `results/*.csv` into every figure and LaTeX table in the paper |
| `06_trajectory_web/` | §5.1 (referenced) | Interactive web page tracking the 10-round evolution of the algorithm, source for [https://web-production-8126.up.railway.app/](https://web-production-8126.up.railway.app/) |

---

## Mapping from paper to code

| Paper item | Code location |
|---|---|
| Table 3 (simulation, per-pattern) | `03_simulation_experiment/results/{baseline,evolved}_per_pattern.csv` |
| Table 4 (real-world, cross-dataset means) | `04_real_world_experiment/results_*/algorithm_comparison.csv` → `05_analysis_and_figures/_paper_summary.py` |
| Figure 4 (trajectory) | `02_case_study_algorithms/trajectory/best_program_r{0,1,...,10}.py` + `06_trajectory_web/` |
| Figure 5 (simulation examples) | `03_simulation_experiment/run_simulation_evolved.py` (writes maps), `05_analysis_and_figures/generate_combined_bar_chart.py` |
| Figure 6 (10-algorithm domain summary) | `05_analysis_and_figures/generate_combined_bar_chart.py` |
| Figure 7 (baseline regime maps) | `05_analysis_and_figures/generate_baseline_map_matrix.py` |
| Figure S1 (full 10×11×4 breakdown) | `05_analysis_and_figures/generate_combined_bar_chart.py` (per-dataset variant) |
| Figure S2 (p-sensitivity) | `04_real_world_experiment/run_p_sensitivity.py` |
| Figure S4 (6 K-Models variant regime maps) | `05_analysis_and_figures/generate_evolved_map_matrix.py` |
| Figure S5 (dominant covariate maps) | `05_analysis_and_figures/generate_map_matrix_coeff.py` + `generate_map_matrix_extra.py` |
| Figure S6 (regression coefficient maps) | `05_analysis_and_figures/generate_map_matrix_r2.py` |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{lou2026sega,
  title   = {Toward Self-Evolving Geospatial Algorithms: An LLM-Driven Framework with a Case Study on Spatial Regime Discovery},
  author  = {Anonymous Authors (redacted for double-blind review)},
  journal = {Annals of the American Association of Geographers},
  year    = {2026}
}
```

This work builds on the GeoEvolve framework (also distributed as a Python package):

```bibtex
@article{luo2025geoevolve,
  title  = {GeoEvolve: An LLM-Powered Multi-Agent System for Evolving Geospatial Algorithms with Embedded Geographic Knowledge},
  author = {Luo, Peng and ...},
  year   = {2025}
}
```

---

## License

This code is released under the MIT License (see `LICENSE`).  The 11 real-world datasets in `04_real_world_experiment/all_datasets/` come from publicly available sources (ERA5, PLACES, CAMELS, US Election 2021); their original licenses apply and are noted in the metadata file.

---

## Contact

For questions about the code, please use the figshare comment system or contact the corresponding author once the paper is published (authors are currently anonymized for double-blind review).
