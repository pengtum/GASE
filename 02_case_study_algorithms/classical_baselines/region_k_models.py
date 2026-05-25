"""Region K-Models (RKM): k-models style regionalization that strictly maintains
adjacency at every move (each unit's new region must be a spatial neighbor's
region, and the donor region must stay connected & above min_size).
"""
import numpy as np

from . import _common as C


def _closest_region(X, Y, coeffs, allowed=None):
    """For each unit, the index of the region whose OLS fit gives lowest squared residual.
    `allowed[i]` (optional) restricts candidate regions for unit i.
    """
    n = len(Y)
    k = len(coeffs)
    ssr = np.full((n, k), np.inf)
    for j in range(k):
        beta = coeffs[j]
        if beta is None:
            continue
        resid = Y - X.dot(beta)
        ssr[:, j] = resid ** 2
    if allowed is not None:
        mask = np.ones((n, k), dtype=bool)
        for i in range(n):
            for r in allowed[i]:
                mask[i, r] = False
        ssr[mask] = np.inf
    return np.argmin(ssr, axis=1)


def run(X, Y, n_regions, w, min_size=10, max_iter=300,
        init_stoc_step=True, verbose=False, coords=None, seed=None):
    rng = np.random.default_rng(seed)
    if min_size is None:
        min_size = max(X.shape[1], 5)

    labels = C.init_zones_seeded(w, n_regions, min_size, rng=rng)
    units = np.arange(w.n)

    for it in range(max_iter):
        regions = [units[labels == r].tolist() for r in range(n_regions)]
        coeffs = C.fit_equations(X, Y, regions)

        # Compute desired (best-fit) region for each unit
        new_labels = _closest_region(X, Y, coeffs)
        moves = np.where(new_labels != labels)[0]
        if len(moves) == 0:
            break

        rng.shuffle(moves)
        applied = 0
        for u in moves:
            old_r = int(labels[u])
            new_r = int(new_labels[u])
            if old_r == new_r:
                continue
            # Adjacency constraint: target region must be among neighbor labels
            neigh_labels = {labels[v] for v in w.neighbors[u]}
            if new_r not in neigh_labels:
                continue
            # Re-fetch donor units (mutated as we apply moves)
            donor_units = np.where(labels == old_r)[0].tolist()
            if len(donor_units) <= min_size:
                continue
            if not C.is_connected_after_removal(donor_units, u, w):
                continue
            labels[u] = new_r
            applied += 1
        if verbose:
            print(f"  RKM iter {it}: {applied} moves")
        if applied == 0:
            break
    return C.relabel_sequential(labels)
