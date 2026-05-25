"""Helpers shared across regression-based regionalization baselines."""
import numpy as np
from collections import deque


def neighbors_of_label(labels, w, target_label):
    """Indices NOT in target_label that are adjacent to a unit in target_label."""
    inside = np.where(labels == target_label)[0]
    cand = set()
    inside_set = set(inside.tolist())
    for u in inside:
        for v in w.neighbors[u]:
            if v not in inside_set:
                cand.add(int(v))
    return cand


def is_connected_after_removal(units, removed_unit, w):
    """Check that the region defined by `units` minus `removed_unit` is still connected."""
    remaining = [u for u in units if u != removed_unit]
    if len(remaining) <= 1:
        return True
    in_set = set(remaining)
    start = remaining[0]
    visited = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in w.neighbors[u]:
            if v in in_set and v not in visited:
                visited.add(v)
                queue.append(v)
    return len(visited) == len(in_set)


def fit_equations(X, Y, regions):
    """OLS coefficients per region (list of arrays of length n_features)."""
    coeffs = []
    for units in regions:
        if len(units) == 0:
            coeffs.append(None)
            continue
        Xr = X[units, :]
        Yr = Y[units]
        beta = np.linalg.pinv(Xr).dot(Yr)
        coeffs.append(beta)
    return coeffs


def region_ssr(X, Y, units):
    if len(units) == 0:
        return 0.0
    Xr = X[units, :]
    Yr = Y[units]
    beta = np.linalg.pinv(Xr).dot(Yr)
    resid = Yr - Xr.dot(beta)
    return float(np.sum(resid ** 2))


def relabel_sequential(labels):
    uniq = np.unique(labels)
    mapping = {old: new for new, old in enumerate(uniq)}
    return np.array([mapping[l] for l in labels], dtype=int)


def init_zones_seeded(w, n_regions, min_size=None, rng=None):
    """Grow `n_regions` spatially-contiguous regions from random seeds via BFS."""
    if rng is None:
        rng = np.random.default_rng()
    n = w.n
    labels = -np.ones(n, dtype=int)
    seed_pool = list(range(n))
    rng.shuffle(seed_pool)
    seeds = seed_pool[:n_regions]

    queues = []
    for r, s in enumerate(seeds):
        labels[s] = r
        q = deque(w.neighbors[s])
        queues.append(q)

    # BFS round-robin
    remaining = n - n_regions
    while remaining > 0:
        progressed = False
        for r in range(n_regions):
            q = queues[r]
            while q:
                u = q.popleft()
                if labels[u] == -1:
                    labels[u] = r
                    remaining -= 1
                    progressed = True
                    for v in w.neighbors[u]:
                        if labels[v] == -1:
                            q.append(v)
                    break
        if not progressed:
            # Some isolated cells: assign to nearest assigned neighbor; if none, to random
            for u in range(n):
                if labels[u] == -1:
                    for v in w.neighbors[u]:
                        if labels[v] >= 0:
                            labels[u] = labels[v]
                            break
                    if labels[u] == -1:
                        labels[u] = rng.integers(0, n_regions)
            break
    return labels
