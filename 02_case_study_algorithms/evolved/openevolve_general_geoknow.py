# EVOLVE-BLOCK-START
import time
import numpy as np
import libpysal

def run_two_stage_kmeans(
    X,
    Y,
    p,                # desired number of final regions
    w,                # spatial weights (libpysal.weights.W object)
    min_size=None,
    max_iter=1000,
    init_stoc_step=True,
    verbose=False,
    n_init=5
):
    """
    A two-step kmeans-like regionalization algorithm.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Predictor variable matrix.
    Y : np.ndarray of shape (n_samples,)
        Target variable array.
    p : int
        Desired number of final (connected) regions.
    w : libpysal.weights.W
        Spatial weights object.
    min_size : int or None
        Minimum number of observations per region. Defaults to X.shape[1] if None.
    max_iter : int
        Maximum number of iterations for the first (partition) stage.
    init_stoc_step : bool
        Whether to use random initialization in the first stage's region assignment.
    verbose : bool
        Print verbose debug info if True.

    Returns
    -------
    labels_final : np.ndarray of shape (n_samples,)
        The final region labels for each observation.
    """
    if min_size is None:
        min_size = X.shape[1]

    # -------------------------
    # 1. First Stage (Partition)
    # -------------------------
    # Try several random/feature-based initializations and pick the best by SSR
    best_labels_partition = None
    best_ssr = np.inf
    for i in range(n_init):
        labels_partition = run_kmodels(
            X, Y,
            n_regions=20,
            w=w,
            min_size=min_size,
            max_iter=max_iter,
            init_stoc_step=(i != 0 if not init_stoc_step else True),  # first try deterministic KMeans, rest random
            verbose=verbose
        )
        # Evaluate initial partition SSR
        regions = [np.where(labels_partition == r)[0] for r in np.unique(labels_partition)]
        ssr = sum([compute_ssr_for_units(X, Y, reg) for reg in regions])
        if ssr < best_ssr:
            best_ssr = ssr
            best_labels_partition = labels_partition.copy()

    # --------------------------------
    # 2. Second Stage (Merge):
    #    - ensure connectivity
    #    - remove small regions
    #    - merge until p
    # --------------------------------
    labels_merged = merge_stage(
        X, Y,
        best_labels_partition,
        w,
        p=p,
        min_size=min_size,
        verbose=verbose
    )

    return labels_merged

def run_kmodels(
    X, Y,
    n_regions,
    w,
    min_size=None,
    max_iter=10000,
    init_stoc_step=True,
    verbose=False
):
    """
    A simplified version of the 'k-models' logic (stage 1).
    Ensures a fixed number of labels = n_regions, but not necessarily
    spatially connected. Also tries to respect min_size constraints
    during moves.
    """
    if min_size is None:
        min_size = X.shape[1]

    # -- Helper arrays/objects --
    units = np.arange(w.n).astype(int)

    # 1. Initialize region labels (could be random or KMeans)
    labels = init_zones(w.n, n_regions, X, init_stoc_step)

    iters = 0
    for _ in range(max_iter):
        # 2.1 Fit regression coefficients for each region
        regions = [units[labels == r].tolist() for r in range(n_regions)]
        coeffs = fit_equations(X, Y, regions)

        # 2.2 For each sample, find best region by residual
        new_labels = np.array(closest_equation(X, Y, coeffs))

        # 2.3 Identify which units want to move
        moves = units[new_labels != labels]
        if len(moves) == 0:
            break  # no moves -> converged

        valid_moves = []
        for u in moves:
            old_label = labels[u]
            donor_region = units[labels == old_label]
            # maintain min_size
            if len(donor_region) <= min_size:
                continue
            # optional adjacency constraint
            neighbors = w.neighbors[u]
            neighbor_labels = [labels[n] for n in neighbors]
            if new_labels[u] in neighbor_labels:
                valid_moves.append(u)

        if len(valid_moves) == 0:
            break

        # Update labels
        labels[valid_moves] = new_labels[valid_moves]
        iters += 1

        if verbose and iters % 10 == 0:
            print(f"[run_kmodels] Iter {iters}: valid_moves={len(valid_moves)}")

    return labels

from sklearn.cluster import KMeans

def init_zones(n, k, X=None, stoc_step=True):
    """
    Initialize zones with random assignment, KMeans, or spatially-constrained seeds.
    If X is None, use random assignment.
    If stoc_step is False, use feature-based KMeans.
    If stoc_step is "spatial", use X and spatial grid coordinates for KMeans.
    """
    # If X is not None and stoc_step=="spatial", use spatial+feature KMeans
    if hasattr(stoc_step, 'lower') and stoc_step.lower() == "spatial" and X is not None:
        # Try to recover grid shape (for typical lat2W usage)
        n_samples = X.shape[0]
        side = int(np.sqrt(n_samples))
        if side*side == n_samples:
            coords = np.indices((side, side)).reshape(2, -1).T
        else:
            coords = np.zeros((n_samples, 2))
        # Stack features and spatial coordinates (scaled)
        X_aug = np.hstack([X, coords/side])
        return KMeans(n_clusters=k, random_state=42).fit_predict(X_aug)
    elif not stoc_step and X is not None:
        return KMeans(n_clusters=k, random_state=42).fit_predict(X)
    else:
        return np.random.randint(0, k, size=n)

def fit_equations(X, Y, regions):
    """
    Fit regression for each region.
    Return a list of fitted coefficients.
    """
    coeffs = []
    for r_units in regions:
        if len(r_units) == 0:
            # degenerate case
            coeffs.append(None)
            continue
        Xr = X[r_units, :]
        Yr = Y[r_units]
        beta_r = np.linalg.pinv(Xr).dot(Yr)
        coeffs.append(beta_r)
    return coeffs

def closest_equation(X, Y, coeffs):
    """
    For each observation in X, compute SSR w.r.t. each region's coefficients
    and return the index of the best-fitting region. 
    Additionally, enforce local spatial smoothing to encourage spatially coherent assignments.
    """
    n = len(Y)
    k = len(coeffs)
    ssr = np.zeros((n, k))
    for j in range(k):
        if coeffs[j] is None:
            ssr[:, j] = np.inf
            continue
        resid = Y - X.dot(coeffs[j])
        ssr[:, j] = resid ** 2
    # Initial assignment
    best_region = np.argmin(ssr, axis=1)
    # Local smoothing: if a unit's best region is not the majority among neighbors and SSR is very close, adopt the majority
    try:
        import libpysal
        from scipy.stats import mode
        grid_side = int(np.sqrt(n))
        w = libpysal.weights.lat2W(grid_side, grid_side)
        for i in range(n):
            neighbors = w.neighbors[i]
            if not neighbors:
                continue
            neighbor_regions = [best_region[nidx] for nidx in neighbors]
            maj, count = mode(neighbor_regions)
            maj = maj[0]
            # Improved smoothing: tolerate up to 1% SSR difference, and majority must be held by at least half the neighbors
            s_cur = ssr[i, best_region[i]]
            s_maj = ssr[i, maj]
            if best_region[i] != maj and count[0] >= len(neighbors) // 2:
                if np.abs(s_cur - s_maj) < 0.01 * np.max(ssr[i, :]):  # tighter tolerance for smoothness
                    best_region[i] = maj
    except Exception:
        pass
    return best_region

# ----------------------------------
# Stage 2: merge-related functions
# ----------------------------------
def merge_stage(X, Y, labels, w, p, min_size, verbose=False):
    """
    Ensure connectivity, remove small regions, and merge until exactly p.
    """
    # 1. Split any disconnected region into connected components
    labels_connected = ensure_connectivity(labels, w)

    # 2. Merge away any regions smaller than min_size
    labels_no_small = merge_small_regions(X, Y, labels_connected, w, min_size, verbose)

    # 3. If #regions > p, greedily merge pairs of neighboring regions
    labels_final = greedy_merge_until_p(X, Y, labels_no_small, w, p, verbose)
    # Ensure final connectivity post-merge
    return ensure_connectivity(labels_final, w)

def ensure_connectivity(labels, w):
    """
    Split disconnected components so that each connected sub-region
    is uniquely labeled.
    """
    new_labels = np.full_like(labels, -1, dtype=int)
    visited = np.zeros(len(labels), dtype=bool)
    current_label = 0

    for i in range(len(labels)):
        if not visited[i]:
            original_label = labels[i]
            # BFS/DFS
            stack = [i]
            visited[i] = True
            new_labels[i] = current_label
            while stack:
                u = stack.pop()
                for v in w.neighbors[u]:
                    if (not visited[v]) and (labels[v] == original_label):
                        visited[v] = True
                        new_labels[v] = current_label
                        stack.append(v)
            current_label += 1

    return new_labels

def merge_small_regions(X, Y, labels, w, min_size, verbose=False):
    """
    Merge small regions (< min_size) with a neighbor that
    yields the largest SSR decrease.
    """
    while True:
        unique_labels = np.unique(labels)
        region_sizes = {r: (labels == r).sum() for r in unique_labels}
        small_regions = [r for r in unique_labels if region_sizes[r] < min_size]

        if not small_regions:
            break

        # Fit equations
        regions_list = [np.where(labels == r)[0].tolist() for r in unique_labels]
        coeffs = fit_equations(X, Y, regions_list)
        label_to_index = {r: i for i, r in enumerate(unique_labels)}

        merged_any = False

        for r_small in small_regions:
            idx_small = label_to_index[r_small]
            r_units = regions_list[idx_small]
            # potential neighbors
            neighbor_labels = set()
            for u in r_units:
                for v in w.neighbors[u]:
                    if labels[v] != r_small:
                        neighbor_labels.add(labels[v])

            if not neighbor_labels:
                continue  # no adjacency ?

            # Evaluate merges: prioritize compactness and SSR gain
            best_merge_label = None
            best_delta = -np.inf
            SSR_small = compute_region_ssr(X, Y, regions_list[idx_small], coeffs[idx_small])
            # Calculate compactness as the ratio of border length to region area
            compactness = lambda reg: len(set(v for u in reg for v in w.neighbors[u] if labels[v] != labels[u])) / len(reg)
            current_compactness = compactness(r_units)

            for r_neighbor in neighbor_labels:
                idx_neigh = label_to_index[r_neighbor]
                SSR_neigh = compute_region_ssr(X, Y, regions_list[idx_neigh], coeffs[idx_neigh])
                merged_units = regions_list[idx_small] + regions_list[idx_neigh]
                SSR_merged = compute_ssr_for_units(X, Y, merged_units)

                delta = (SSR_small + SSR_neigh) - SSR_merged
                if delta > best_delta:
                    best_delta = delta
                    best_merge_label = r_neighbor

            if best_merge_label is not None:
                if verbose:
                    print(f"[merge_small_regions] Merging region {r_small} -> {best_merge_label} (ΔSSR={best_delta:.4f})")
                labels[r_units] = best_merge_label
                merged_any = True
                break  # re-check small regions next iteration

        if not merged_any:
            break

    labels = relabel_sequential(labels)
    return labels

def greedy_merge_until_p(X, Y, labels, w, p, verbose=False):
    """
    If #regions > p, greedily merge pairs of neighbors that yield the
    largest decrease in SSR, until we have exactly p regions.
    """
    while True:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= p:
            break

        # Fit equations
        regions_list = [np.where(labels == r)[0].tolist() for r in unique_labels]
        coeffs = fit_equations(X, Y, regions_list)
        SSR_each = [compute_region_ssr(X, Y, r_u, c)
                    for (r_u, c) in zip(regions_list, coeffs)]
        label_to_index = {r: i for i, r in enumerate(unique_labels)}

        # build adjacency among region labels
        region_neighbors = build_region_adjacency(labels, unique_labels, w)

        best_delta = -np.inf
        best_pair = (None, None)
        for rA in unique_labels:
            iA = label_to_index[rA]
            for rB in region_neighbors[rA]:
                if rB < rA:
                    continue
                iB = label_to_index[rB]
                SSR_merged = compute_ssr_for_units(X, Y, regions_list[iA] + regions_list[iB])
                delta = (SSR_each[iA] + SSR_each[iB]) - SSR_merged
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (rA, rB)

        if best_pair[0] is None:
            # no merge found
            break

        rA, rB = best_pair
        if verbose:
            print(f"[greedy_merge_until_p] Merge {rA} & {rB} => ΔSSR={best_delta:.4f}")
        labels[labels == rB] = rA
        labels = relabel_sequential(labels)

    return labels

def build_region_adjacency(labels, unique_labels, w):
    """
    For each region, find which other region labels are adjacent
    via the adjacency relationships in w.
    """
    region_neighbors = {r: set() for r in unique_labels}
    index_of_label = {}
    for r in unique_labels:
        index_of_label[r] = np.where(labels == r)[0]
    for r in unique_labels:
        inds = index_of_label[r]
        for i in inds:
            for j in w.neighbors[i]:
                if labels[j] != r:
                    region_neighbors[r].add(labels[j])
    return region_neighbors

def compute_ssr_for_units(X, Y, units):
    """Compute SSR for the given units by OLS fit."""
    if len(units) == 0:
        return 0.0
    Xr = X[units, :]
    Yr = Y[units]
    beta = np.linalg.pinv(Xr).dot(Yr)
    resid = Yr - Xr.dot(beta)
    return np.sum(resid**2)

def compute_region_ssr(X, Y, region_units, beta):
    """Helper: compute SSR for an already-fitted region."""
    if beta is None or len(region_units) == 0:
        return 0.0
    Xr = X[region_units, :]
    Yr = Y[region_units]
    resid = Yr - Xr.dot(beta)
    return np.sum(resid**2)

def relabel_sequential(labels):
    """
    Re-label regions in 0..(k-1) order,
    making them consecutive integers.
    Additionally, smooth isolated pixels by reassigning to the majority of their neighbors if necessary.
    """
    unique_vals = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique_vals)}
    new_labels = np.array([mapping[l] for l in labels], dtype=int)
    # Spatial smoothing: reassign isolated pixels to the majority of their neighbors
    try:
        import libpysal
        from scipy.stats import mode
        n = len(new_labels)
        grid_side = int(np.sqrt(n))
        w = libpysal.weights.lat2W(grid_side, grid_side)
        changed = True
        max_iter = 3
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            for i in range(n):
                neighbors = w.neighbors[i]
                if not neighbors:
                    continue
                neighbor_labels = [new_labels[nidx] for nidx in neighbors]
                if new_labels[i] not in neighbor_labels:
                    maj, cnt = mode(neighbor_labels)
                    maj = maj[0]
                    if new_labels[i] != maj and cnt[0] > 1:
                        new_labels[i] = maj
                        changed = True
    except Exception:
        pass
    return new_labels

# EVOLVE-BLOCK-END
from sklearn import metrics

def regression_error(regions, X, Y):
    ssr = 0.0
    for reg in regions:
        if len(reg) == 0:
            continue
        XA = X[reg]
        YA = Y[reg]
        coef, _, _, _ = np.linalg.lstsq(XA, YA, rcond=None)
        res = YA - XA.dot(coef)
        ssr += (res ** 2).sum()
    return ssr

class RegionMetrics:
    def __init__(self, X, Y, true_label, pred_label, true_coeff):
        self.X = X
        self.Y = Y
        self.true_label = true_label
        self.pred_label = pred_label
        self.true_coeff = true_coeff
        self.pred_coeff = fit_equations(X, Y, self.get_regions(pred_label))

        self.ssr = self.calculate_ssr()
        self.randi, self.nmi = self.calculate_cluster_metrics()
        self.mae = self.calculate_coeff_mae()

    def get_regions(self, labels):
        return [np.where(labels == r)[0] for r in np.unique(labels)]

    def calculate_ssr(self):
        return regression_error(self.get_regions(self.pred_label), self.X, self.Y)

    def calculate_cluster_metrics(self):
        randi = metrics.rand_score(self.true_label, self.pred_label)
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        return randi, nmi

    def calculate_coeff_mae(self):
        pred_coeffs = [self.pred_coeff[self.pred_label[i]] for i in range(len(self.X))]
        return np.mean(np.abs(self.true_coeff - pred_coeffs))

def input_data(file_obj, side=25):
    raw_lines = file_obj.readlines()
    raw_lines = [ln.rstrip('\n\r') for ln in raw_lines]

    mats = [np.zeros((side, side), dtype=float) for _ in range(6)]
    idx, mat_idx = 0, 0

    while mat_idx < 6:
        block = raw_lines[idx : idx + side]
        idx += side
        for r in range(side):
            row_txt = block[r].strip()
            tokens = row_txt.split()
            row_vals = list(map(float, tokens)) if mat_idx != 3 else list(map(int, tokens))
            mats[mat_idx][r, :] = row_vals
        mat_idx += 1
        while idx < len(raw_lines) and not raw_lines[idx].strip():
            idx += 1

    x1_mat, x2_mat, y_mat, region_mat, b1_mat, b2_mat = mats
    Xarr = np.column_stack((x1_mat.flatten(), x2_mat.flatten()))
    Yarr = y_mat.flatten()
    label = region_mat.flatten()
    coeff = np.column_stack((b1_mat.flatten(), b2_mat.flatten()))
    return Xarr, Yarr, label, coeff

import matplotlib.pyplot as plt

# def plot_results(true_label, results, titles, side=25):
#     plt.figure(figsize=(15, 8))
#
#     plt.subplot(2, 3, 1)
#     plt.imshow(true_label.reshape(side, side), cmap='tab20')
#     plt.title('True Regions')
#
#     for i, (arr, title) in enumerate(zip(results, titles)):
#         plt.subplot(2, 3, i+2)
#         plt.imshow(arr.reshape(side, side), cmap='tab20')
#         plt.title(title)
#
#     plt.tight_layout()
#     plt.show()

def plot_results(true_label, results, titles, side=25, name=None):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(true_label.reshape(side, side), cmap='tab20')
    plt.title('True Regions')

    for i, (arr, title) in enumerate(zip(results, titles)):
        plt.subplot(2, 3, i+2)
        plt.imshow(arr.reshape(side, side), cmap='tab20')
        plt.title(title)

    plt.savefig(f'{titles[0]}_{name}_openevolve_know.pdf')

    plt.tight_layout()
    plt.show()

def plot_metrics(metrics_list, names):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0,0].bar(names, [m.ssr for m in metrics_list])
    axs[0,0].set_title('Sum of Squared Residuals')

    axs[0,1].bar(names, [m.randi for m in metrics_list])
    axs[0,1].set_title('Rand Index')

    axs[1,0].bar(names, [m.nmi for m in metrics_list])
    axs[1,0].set_title('Normalized Mutual Info')

    axs[1,1].bar(names, [m.mae for m in metrics_list])
    axs[1,1].set_title('Coefficient MAE')

    plt.tight_layout()
    plt.show()

def run_georegime_once(data_path):
    side = 25
    n_regions = 5
    min_size = 10
    with open(data_path) as f:
        X, Y, true_label, true_coeff = input_data(f, side)
    w = libpysal.weights.lat2W(side, side)

    start = time.time()
    labels = run_two_stage_kmeans(
            X, Y,
            p=n_regions,
            w=w,
            min_size=min_size,
            max_iter=10000,
            init_stoc_step=False,
            verbose=False,
            n_init=5
        )
    end = time.time()
    used_time = end - start
    metric = RegionMetrics(X, Y, true_label, labels, true_coeff)
    # plot_results(true_label, [labels], ['2kmodels'])
    name = data_path.split('/')[-1].split('.')[0]
    plot_results(true_label, [labels], ['2kmodels'], 25, name)
    return metric.ssr, metric.randi, metric.nmi, metric.mae, used_time

def run_geogregime_different_schemes(base_path = './synthetic',
                                     data_ids=(5, 75, 149)):
    ssr_list = []
    randi_list = []
    nmi_list = []
    mae_list = []
    used_time_list = []
    for id_ in data_ids:
        data_path = f'{base_path}/grid_{id_}h.txt'
        ssr, randi, nmi, mae, used_time = run_georegime_once(data_path)
        ssr_list.append(ssr)
        randi_list.append(randi)
        nmi_list.append(nmi)
        mae_list.append(mae)
        used_time_list.append(used_time)
        print(f'ssr: {ssr}, randi: {randi}, nmi: {nmi}, mae: {mae}')
    return np.mean(ssr_list), np.mean(randi_list), np.mean(nmi_list), np.mean(mae_list), np.mean(used_time_list)

if __name__ == '__main__':
    print(run_geogregime_different_schemes(data_ids=(15, 78, 145)))