"""Matched-budget grid search around the UNMODIFIED 2KModels seed.

The seed algorithm is loaded from `seed_plain.py` via importlib and is not
edited in any way (`seed_plain.py` is character-identical to
`simulation_study/algorithms/initial_program.py`; the only difference is
CRLF -> LF line endings).

Per grid, the wrapper repeatedly runs the seed's own stage-1 + merge
pipeline over the same initial-cluster-count candidate set the evolved
algorithm searches internally, {int(2.5p), 4p, sqrt(n), p, p+1, p+2},
cycling through the candidates with a fresh random restart each pass,
until a wall-clock budget is exhausted.  The budget equals the evolved
algorithm's mean per-grid runtime measured in the same batch on the same
machine (7.47 s).  Two deliberate choices favour the baseline:

  * at least one full sweep over the candidate set is always completed,
    even if it exceeds the budget;
  * the budget is checked between runs, so a run that starts inside the
    budget may finish beyond it.

Selection among candidates uses the seed's own label-free post-merge SSR
(computed with the seed's `compute_ssr_for_units`), exactly as the evolved
algorithm selects internally.  No ground-truth information is used.
"""
import importlib.util
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "seed2k_plain", os.path.join(_HERE, "seed_plain.py"))
_seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seed)

# Evolved A*'s mean per-grid runtime, same machine / same batch
# (results/clean_per_dataset.csv, "GeoEvolve A* (full)").
BUDGET_SECONDS = 7.47


def run_two_stage_kmeans(X, Y, p, w, min_size=None, max_iter=1000,
                         init_stoc_step=True, verbose=False):
    t0 = time.time()
    if min_size is None:
        min_size = X.shape[1]
    n = X.shape[0]

    # Candidate set: verbatim construction from the evolved program.
    candidate_ks = [
        min(max(int(2.5 * p), 2 * p), n // max(min_size, 1)),
        min(max(int(4 * p), 2 * p), n // max(min_size, 1)),
        min(max(int(np.sqrt(n)), p), n // max(min_size, 1)),
    ]
    if p not in candidate_ks:
        candidate_ks.append(p)
    for addk in (p + 1, p + 2):
        if addk <= n // max(min_size, 1):
            candidate_ks.append(addk)
    candidate_ks = list(sorted(set(candidate_ks)))

    best_partition = None
    best_ssr = np.inf
    run_idx = 0
    # Guaranteed first sweep over all candidates, then keep adding random
    # restarts (cycling candidates) while within the time budget.
    while run_idx < len(candidate_ks) or (time.time() - t0) < BUDGET_SECONDS:
        k = candidate_ks[run_idx % len(candidate_ks)]
        np.random.seed(7919 * run_idx + k)
        labels_partition = _seed.run_kmodels(
            X, Y, n_regions=k, w=w, min_size=min_size,
            max_iter=max_iter // 2, init_stoc_step=init_stoc_step,
            verbose=verbose)
        labels_merged = _seed.merge_stage(
            X, Y, labels_partition, w, p=p, min_size=min_size,
            verbose=verbose)
        ssr = sum(_seed.compute_ssr_for_units(X, Y, np.where(labels_merged == r)[0])
                  for r in np.unique(labels_merged))
        if ssr < best_ssr:
            best_ssr = ssr
            best_partition = labels_partition
        run_idx += 1

    return _seed.merge_stage(X, Y, best_partition, w, p=p,
                             min_size=min_size, verbose=verbose)
