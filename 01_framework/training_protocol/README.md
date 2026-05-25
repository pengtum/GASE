# Module 3 — Simulation-Driven Training Protocol with a Controlled DGP

This module specifies what dataset the inner loop is allowed to see during evolution and what metric drives the search. SEGA prescribes three principles:

1. **Run the inner loop only on synthetic data.** Evolution happens on a synthetic dataset $\mathcal{D}_{\text{sim}}$ whose ground-truth structure is known by construction. Real-world data is never seen by the mutation, selection, or knowledge-retrieval steps.

2. **Apply the discovered algorithm unchanged to real data.** After $T$ rounds, the algorithm $\mathcal{A}^*$ is frozen and applied to each real-world dataset with no hyperparameter retuning and no additional evolution. Real-world results are therefore out-of-sample.

3. **Use different metrics for driving the search and for reporting results.** A bounded, dimensionless *fitness signal* (e.g., Rand Index) drives the inner loop; a separate set of *evaluation metrics* (e.g., SSR Reduction, RMSE) is computed post-hoc.

## How this maps to the case study

| Principle | Spatial regime case study implementation |
|---|---|
| 1. Synthetic only | 150 synthetic 25×25 grids (Rectangular / Voronoi / Arbitrary patterns) — see `03_simulation_experiment/datasets/` |
| 2. Frozen + applied to real data | Same `SEGA_2kmodels_final.py` is run on 11 real-world datasets — see `04_real_world_experiment/run_real_world_evolved.py` |
| 3. Separate fitness vs. evaluation | Fitness = Rand Index against ground-truth partition. Evaluation = SSR, SSR Reduction, RMSE, Variance Ratio |

No additional Python module is needed for this protocol — it is enforced by *how the runner scripts in 03_/04_ are wired*.
