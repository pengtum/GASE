# 07 — Equal-Budget Baseline and Component Ablations (paper Appendix D)

Reproduces **Table D.6**: the matched-budget comparison between the evolved algorithm and
its unmodified seed, and the one-component ablations of the evolved algorithm.

Added in the second review round, which asked whether the evolved algorithm's advantage
comes from the extra computation it spends at inference or from its algorithmic content.

## How to read the two metric columns

Under Principle 4 (paper §3.2.3) no ground-truth information may reach a candidate
program, so **every** selection rule in this study — inside the evolved algorithm and
inside the matched-budget wrapper alike — ranks candidate partitions by post-merge SSR.
SSR is therefore the one quantity additional search can buy directly, and a search-based
baseline will narrow the SSR gap whether or not it has recovered any more of the regime
structure.

**ARI is the comparison of record.** It is reachable by no selection rule here, and it is
the quantity spatial regime discovery is defined on. Closeness on SSR is weak evidence of
equivalence.

## Programs — `programs/`

Each ablation differs from `evolved_plain.py` by exactly one change, marked `# ABLATION:`
in the source, so `diff` makes the change auditable.

| Variant in Table D.6 | File | What it is |
|---|---|---|
| `2kmodels (seed)` | `seed_plain.py` | The seed, byte-for-byte unmodified; same-batch reference |
| `Seed + equal-compute search` | `seed_gridsearch.py` | The seed kernel unchanged, wrapped so that it searches the same six initial cluster counts the evolved algorithm searches, `{⌊2.5p⌋, 4p, ⌊√n⌋, p, p+1, p+2}` (candidate-set code copied verbatim), gives each candidate `max_iter//2` in stage one (as the evolved algorithm does), draws 3 random restarts per candidate, and selects by the seed's own label-free post-merge SSR |
| `Seed + matched-budget search` | `seed_gridsearch_budget.py` | Strict budget match with **zero** edits to the seed: loads `seed_plain.py` through `importlib` and loops `(k, restart)` combinations within the evolved algorithm's measured mean per-grid runtime (7.47 s, same batch, same machine) |
| `GeoEvolve A* (full)` | `evolved_plain.py` | The frozen discovered algorithm reported in the paper |
| `A* w/o initial-k search` | `evolved_no_ksearch.py` | `candidate_ks` → `[20]`, the seed's fixed stage-one count |
| `A* w/o border refinement` | `evolved_no_refine.py` | Skips `batch_border_refinement_early_stop` |
| `A* w/o spatial penalty` | `evolved_no_penalty.py` | Refinement kept, `spatial_penalty = 0` (jitter disabled with it) |
| `A* w/o KMeans++ init` | `evolved_random_init.py` | `init_zones` reverted to the seed's plain random assignment |

Two asymmetries deliberately favour the baseline: it always completes at least one full
pass over the six cluster counts, and it checks its budget only between passes, so it
actually runs **1.11×** as long as the evolved algorithm.

## Runner

```bash
python run_eval.py --workers 8      # all 150 grids × 8 variants
```

Adapted from `Round-2/reanalysis/run_clean_simulation_eval.py` and using the same
label-free harness: a program receives only `(X, Y, w, p, min_size)` and returns a label
vector; every metric is computed outside it. Settings: 150 grids, `p = 5`,
`min_size = 10`, `max_iter = 10000`, one machine, one batch for the timing column.
Re-runs de-duplicate on `(algorithm, grid)`, so an interrupted run resumes.

Inputs are read from elsewhere in this repository:
`../03_simulation_experiment/datasets/` and
`../02_case_study_algorithms/classical_baselines/`.

## Outputs — `results/`

| File | Content |
|---|---|
| `summary_equal_compute_ablation.csv` | The numbers behind Table D.6: mean ± SD SSR / ARI, mean time, and paired Wilcoxon ΔSSR / ΔARI against the full evolved algorithm |
| `clean_overall.csv` | Cross-grid means of every metric (SSR, RandI, ARI, NMI, MAE, SSR reduction, variance ratio, time) per variant |
| `clean_per_pattern.csv` | The same, split by Rectangular / Voronoi / Arbitrary |
| `clean_per_dataset.csv` | Per-grid raw numbers (1200 rows = 8 variants × 150 grids) |
| `wilcoxon_vs_seed.csv` | Paired Wilcoxon signed-rank tests against the unevolved seed |
| `comparison_vs_paper.csv` | Seed numbers re-measured here against the published values, as a harness check |
| `labels_*.npz` | Predicted labels per variant per grid, so the partitions can be inspected without re-running |

## Table D.6 as reported

| Variant | SSR ↓ | ARI ↑ | Rel. time | ΔSSR (median) | p | ΔARI (median) | p |
|---|---|---|---|---|---|---|---|
| Evolved algorithm (reference) | 58.96 ± 9.46 | 0.894 ± 0.096 | 1.00 | | | | |
| *(a) Seed under a matched budget* | | | | | | | |
| Seed, single run | 80.52 ± 20.11 | 0.801 ± 0.105 | 0.10 | −17.87 | 1.1×10⁻²⁴ | +0.084 | 6.0×10⁻²¹ |
| Seed + matched-budget search | 60.60 ± 8.64 | 0.854 ± 0.091 | 1.11 | −1.90 | 2.1×10⁻⁵ | +0.043 | 2.3×10⁻¹⁰ |
| *(b) Evolved algorithm with one component removed* | | | | | | | |
| w/o initial-k search | 62.35 ± 12.08 | 0.873 ± 0.114 | 0.56 | −0.33 | 4.3×10⁻⁵ | +0.004 | 6.3×10⁻³ |
| w/o border refinement | 68.72 ± 13.68 | 0.830 ± 0.094 | 0.98 | −7.09 | 2.1×10⁻²⁵ | +0.063 | 2.4×10⁻²⁶ |
| w/o spatial penalty | 56.54 ± 8.53 | 0.852 ± 0.092 | 0.99 | +1.80 | 3.9×10⁻¹⁹ | +0.039 | 1.1×10⁻²⁴ |
| w/o KMeans++ initialization | 57.90 ± 7.09 | 0.902 ± 0.092 | 0.57 | 0.00 | 0.68 | 0.000 | 0.68 |

Negative ΔSSR and positive ΔARI favour the evolved algorithm. Paired medians are used
because the per-grid SSR distribution is right-skewed, so they need not match the
difference of the means in the first column. The two identical p-values in the last row
come from the signed-rank statistic taking mirror-image values on the two metrics; they
are a coincidence, not a transcription error.

What the table shows: additional computation accounts for most of the SSR advantage but
not for the recovery of the regime structure itself. The component that is purely a
matter of search effort — the initial-cluster-count search — contributes the least of the
three that contribute at all, while border refinement contributes an order of magnitude
more. The spatial penalty contributes +0.039 ARI even though removing it *lowers* SSR;
because its value is negative in the algorithm's own selection objective, no tuning of
the seed's parameters against that objective could have found it.
