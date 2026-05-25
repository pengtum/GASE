# 02 — Case-Study Algorithms (10 total)

The 10 algorithms compared throughout the paper (Tables 3 + 4 + Figure 6), grouped by family. See paper §4.4 for the full taxonomy.

## `seed/`

| File | Paper name | Description |
|---|---|---|
| `2kmodels_seed.py` | 2KModels (seed) | The Two-stage K-Models algorithm of Guo & Wang (2023). Acts as $\mathcal{A}_0$ — the starting point for all 5 evolved variants. |

## `evolved/` — 5 evolved K-Models variants

These are the **final discovered programs** at the end of evolution. They were produced by running the `geoevolve` Python package on the seed algorithm with different knowledge-injection settings (paper Table 2 / §4.4.1).

| File | Paper name | What changes vs. the seed |
|---|---|---|
| `openevolve_no_geoknow.py` | OpenEvolve (No GeoKnow.) | OpenEvolve flat loop with no geographic knowledge prompt |
| `openevolve_general_geoknow.py` | OpenEvolve (General GeoKnow.) | + general geographic-knowledge prompt (Tobler, autocorrelation, scale) |
| `openevolve_specific_geoknow.py` | OpenEvolve (Specific GeoKnow.) | + spatial-regime-specific hints (contiguity penalty, size balancing) |
| `geoevolve_no_rag.py` | GeoEvolve (No GeoKnowRAG) | Full GeoEvolve multi-agent dialogue, retrieval module disabled |
| `SEGA_2kmodels_final.py` | **GeoEvolve / SEGA-2KModels** (the final) | Full GeoEvolve with GeoKnowRAG enabled — **this is the algorithm reported as "GeoEvolve" / "SEGA-2KModels" in the paper** |

Each file is the output of an evolution run, with the original `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers preserved so its provenance is visible. You can run any of them directly:

```python
from SEGA_2kmodels_final import run_algorithm
labels, predictions = run_algorithm(X, y, spatial_weights, p=5)
```

## `classical_baselines/` — 4 classical regionalization methods

| File | Paper name | Reference |
|---|---|---|
| `azp.py` | AZP | Openshaw (1977, 1995) |
| `region_k_models.py` | Region-K-Models | Guo & Wang (2023) |
| `gwr_skater.py` | GWR-Skater | Helbich et al. (2013) |
| `skater_reg.py` | Skater-reg | Anselin (2024) |
| `_common.py`, `__init__.py` | (utilities) | – |

These are independent of `geoevolve` and only depend on the standard PyData + libpysal + spopt stack.

## `trajectory/` — round-by-round snapshots

The 11 files `best_program_r0.py` … `best_program_r10.py` are snapshots of the evolving algorithm after each of the 10 inner-loop rounds reported in paper Figure 4 (and in the interactive web view in `06_trajectory_web/`):

- `r0` = the seed (== `seed/2kmodels_seed.py`)
- `r4`, `r5`, `r6`, `r10` = the four score-changing rounds
- `r10` = the final discovered algorithm (== `evolved/SEGA_2kmodels_final.py`)

These are useful if you want to trace exactly which mutation was introduced when.
