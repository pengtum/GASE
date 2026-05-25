"""Regression-based AZP (Automatic Zoning Procedure).

Classic AZP move: pick a border unit and move it to an adjacent region if doing
so reduces the total within-region SSR while preserving (a) min_size and
(b) connectivity of the donor region.
"""
import numpy as np

from . import _common as C


def _swap_delta(X, Y, donor_units, recv_units, u):
    """Change in total SSR if u moves from donor -> recv. Negative = improvement."""
    donor_after = [v for v in donor_units if v != u]
    recv_after = recv_units + [u]
    cur = C.region_ssr(X, Y, donor_units) + C.region_ssr(X, Y, recv_units)
    new = C.region_ssr(X, Y, donor_after) + C.region_ssr(X, Y, recv_after)
    return new - cur


def run(X, Y, n_regions, w, min_size=10, max_iter=300,
        init_stoc_step=True, verbose=False, coords=None, seed=None):
    rng = np.random.default_rng(seed)
    if min_size is None:
        min_size = max(X.shape[1], 5)

    labels = C.init_zones_seeded(w, n_regions, min_size, rng=rng)

    for it in range(max_iter):
        changed = False
        order = list(range(n_regions))
        rng.shuffle(order)
        for r in order:
            cand_units = C.neighbors_of_label(labels, w, r)
            if not cand_units:
                continue
            cand_units = list(cand_units)
            rng.shuffle(cand_units)
            recv_units = np.where(labels == r)[0].tolist()
            for u in cand_units:
                old_r = int(labels[u])
                if old_r == r:
                    continue
                donor_units = np.where(labels == old_r)[0].tolist()
                if len(donor_units) <= min_size:
                    continue
                if not C.is_connected_after_removal(donor_units, u, w):
                    continue
                delta = _swap_delta(X, Y, donor_units, recv_units, u)
                if delta < -1e-9:
                    labels[u] = r
                    recv_units.append(u)
                    changed = True
                    if verbose:
                        print(f"  iter {it}: moved {u} from {old_r} -> {r}, delta={delta:.4f}")
        if not changed:
            break
    return C.relabel_sequential(labels)
