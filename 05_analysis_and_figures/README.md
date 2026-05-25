# 05 — Analysis & Figures

Scripts that turn the result CSVs in `03_/04_/results*/` into the figures and tables in the paper.

## Master cross-check

| Script | Output |
|---|---|
| `_paper_summary.py` | Prints the canonical real-world means + win counts to stdout (matches Table 4 exactly) |
| `_print_xlsx.py` | Pretty-prints the simulation Excel summary |

Run `python _paper_summary.py` after `03_/run_*.py` and `04_/run_*.py` are done to verify your numbers match Table 4.

## Bar charts (Figure 6 + Figure S1)

| Script | Paper figure |
|---|---|
| `generate_combined_bar_chart.py` | **Figure 6** — compact 10-algorithm SSR-Reduction summary by domain (4 columns) |
| `generate_bar_chart.py` | Original 3-algorithm bar chart |
| `generate_baseline_bar_chart.py` | Variant: classical baselines only |

## Regime / map matrices (Figure 7 + Figure S4 + Figure S5 + Figure S6)

| Script | Paper figure |
|---|---|
| `generate_baseline_map_matrix.py` | **Figure 7** — 4 classical baselines × 4 representative datasets |
| `generate_evolved_map_matrix.py` | **Figure S4** — 6 K-Models variants × 4 representative datasets |
| `generate_map_matrix.py` | Generic 3-variant map matrix |
| `generate_map_matrix_extra.py` | Additional map breakdowns |
| `generate_map_matrix_coeff.py` | **Figure S5** — dominant covariate maps |
| `generate_map_matrix_r2.py` | **Figure S6** — regression coefficient maps |
| `generate_baseline_r2_map_matrix.py` | Baseline-only $R^2$ maps |

## LaTeX tables

| Script | Output |
|---|---|
| `generate_latex_table.py` | Produces a ready-to-paste LaTeX `tabular` for Table 4 |

## Where the figures land

After running any `generate_*.py` script, output PDFs are written to `paper_figures/`. Some figures are pre-generated and shipped here for convenience.
