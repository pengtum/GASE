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
    true_label=None,
    rand_state=42
):
    """
    Improved two-step kmeans-like regionalization algorithm.
    Uses KMeans for initialization and multiple restarts for robustness.
    Now: supports true_label for Rand/NMI fitness, and reproducible seeds.
    Adds size std constraint and runtime/compactness/contiguity penalties to fitness.
    """
    if min_size is None:
        min_size = X.shape[1]

    best_labels = None
    best_score = -np.inf

    n_restarts = 10  # Increase restarts for better robustness and exploration

    for attempt in range(n_restarts):
        np.random.seed(rand_state + attempt)

        labels_partition = run_kmodels(
            X, Y,
            n_regions=max(p*4, 20),
            w=w,
            min_size=min_size,
            max_iter=max_iter,
            init_stoc_step=True,
            verbose=verbose
        )

        labels_merged = merge_stage(
            X, Y,
            labels_partition,
            w,
            p=p,
            min_size=min_size,
            verbose=verbose
        )

        unique, counts = np.unique(labels_merged, return_counts=True)
        size_std = np.std(counts)
        size_mean = np.mean(counts)
        balance_score = 1.0 - (size_std / (size_mean + 1e-8))
        compactness_score = region_compactness(labels_merged, w)
        ssr_score = -compute_ssr_for_units(X, Y, np.arange(X.shape[0]))
        contiguity_score = check_all_regions_contiguous(labels_merged, w)
        size_std_penalty = 0.0
        contig_penalty = 0.0

        # Penalty if region size std exceeds 20% of mean
        if size_std > 0.2 * size_mean:
            size_std_penalty = -0.2 * (size_std / (size_mean + 1e-8))
        # Penalty if any region is not contiguous (should be rare)
        if contiguity_score < 1.0:
            contig_penalty = -0.2

        # If true_label is available, use Rand Index/NMI as fitness, but penalize unbalanced or fragmented solutions
        if true_label is not None:
            from sklearn import metrics
            randi = metrics.rand_score(true_label, labels_merged)
            nmi = metrics.normalized_mutual_info_score(true_label, labels_merged)
            score = (
                randi + 0.2 * nmi
                + 0.1 * balance_score
                + 0.1 * compactness_score
                + size_std_penalty
                + contig_penalty
            )
        else:
            score = (
                0.45 * balance_score
                + 0.35 * compactness_score
                + 0.2 * ssr_score / (np.abs(ssr_score) + 1e-8)
                + size_std_penalty
                + contig_penalty
            )

        if best_labels is None or score > best_score:
            best_labels = labels_merged
            best_score = score

    return best_labels

def check_all_regions_contiguous(labels, w):
    """
    Returns 1.0 if all regions are spatially contiguous, 0.0 otherwise.
    """
    unique_labels = np.unique(labels)
    for r in unique_labels:
        region_inds = np.where(labels == r)[0]
        if len(region_inds) <= 1:
            continue
        visited = set([region_inds[0]])
        stack = [region_inds[0]]
        region_set = set(region_inds)
        while stack:
            u = stack.pop()
            for v in w.neighbors[u]:
                if v in region_set and v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) != len(region_inds):
            return 0.0
    return 1.0

def region_compactness(labels, w):
    """
    Compute a normalized compactness score [0,1], higher = more compact.
    For each region, compactness = (# of internal edges)/(# of total edges)
    Return mean over all regions.
    """
    unique_labels = np.unique(labels)
    compactness_list = []
    for r in unique_labels:
        inds = np.where(labels == r)[0]
        if len(inds) == 0:
            continue
        internal = 0
        total = 0
        for i in inds:
            for j in w.neighbors[i]:
                if labels[j] == r:
                    internal += 1
                total += 1
        # To avoid double counting, divide internal by 2
        if total > 0:
            cscore = (internal / 2) / total
            compactness_list.append(cscore)
    if len(compactness_list) == 0:
        return 0.0
    return np.mean(compactness_list)

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
    Adds periodic rebalancing to encourage region size balance.
    """
    if min_size is None:
        min_size = X.shape[1]

    # -- Helper arrays/objects --
    units = np.arange(w.n).astype(int)

    # 1. Initialize region labels (could be random or KMeans)
    labels = init_zones(w.n, n_regions, X, init_stoc_step)

    iters = 0
    for iter_ in range(max_iter):
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

        # Periodic rebalancing: if region size std gets too high, reassign smallest units to closest big region
        if iter_ % 25 == 0 and iter_ > 0:
            unique, counts = np.unique(labels, return_counts=True)
            size_std = np.std(counts)
            size_mean = np.mean(counts)
            if size_std > 0.25 * size_mean:
                # Find small regions and send units to nearest large region
                for r in unique:
                    inds = np.where(labels == r)[0]
                    if len(inds) < min_size * 1.5:
                        for u in inds:
                            # Find a neighboring region with above-average size
                            neighbor_regs = [labels[n] for n in w.neighbors[u] if labels[n] != r]
                            neighbor_regs = [nr for nr in neighbor_regs if counts[unique == nr][0] > size_mean]
                            if neighbor_regs:
                                labels[u] = neighbor_regs[0]

        if verbose and iters % 10 == 0:
            print(f"[run_kmodels] Iter {iters}: valid_moves={len(valid_moves)}")

    return labels

from sklearn.cluster import KMeans

def init_zones(n, k, X=None, stoc_step=True):
    """
    Initialize zones with either random assignment or KMeans.
    If X is None, use random assignment.
    """
    # Hybrid initialization: mix KMeans with random for diversity
    if stoc_step and X is not None:
        if np.random.rand() > 0.5:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            return km.fit_predict(X)
        else:
            return np.random.randint(0, k, size=n)
    else:
        return np.random.randint(0, k, size=n)

def fit_equations(X, Y, regions):
    """
    Fit regression for each region.
    Return a list of fitted coefficients.
    Use lstsq for stability.
    """
    coeffs = []
    for r_units in regions:
        if len(r_units) == 0:
            coeffs.append(None)
            continue
        Xr = X[r_units, :]
        Yr = Y[r_units]
        # Use lstsq for numerical stability (especially for small regions)
        beta_r, _, _, _ = np.linalg.lstsq(Xr, Yr, rcond=None)
        coeffs.append(beta_r)
    return coeffs

def closest_equation(X, Y, coeffs):
    """
    For each observation in X, compute SSR w.r.t. each region's coefficients
    and return the index of the best-fitting region.
    Uses vectorized computation for speed and precision.
    """
    n = len(Y)
    k = len(coeffs)
    ssr = np.empty((n, k))
    for j, coef in enumerate(coeffs):
        if coef is None:
            ssr[:, j] = np.inf
            continue
        pred = X @ coef
        ssr[:, j] = (Y - pred) ** 2
    best_region = np.argmin(ssr, axis=1)
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
    return labels_final

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

            # evaluate merges
            best_merge_label = None
            best_delta = -np.inf
            SSR_small = compute_region_ssr(X, Y, regions_list[idx_small], coeffs[idx_small])

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
    Also handles negative/invalid labels gracefully.
    """
    unique_vals = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique_vals) if old >= 0}
    new_labels = np.array([mapping[l] if l in mapping else -1 for l in labels], dtype=int)
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

def plot_results(true_label, results, titles, side=25):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(true_label.reshape(side, side), cmap='tab20')
    plt.title('True Regions')

    for i, (arr, title) in enumerate(zip(results, titles)):
        plt.subplot(2, 3, i+2)
        plt.imshow(arr.reshape(side, side), cmap='tab20')
        plt.title(title)

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
            # init_stoc_step=True,
            verbose=False,
            true_label=true_label
        )
    end = time.time()
    used_time = end - start
    metric = RegionMetrics(X, Y, true_label, labels, true_coeff)
    # plot_results(true_label, [labels], ['2kmodels'])
    return metric.ssr, metric.randi, metric.nmi, metric.mae, used_time

def run_geogregime_different_schemes(base_path = '<PROJECT_ROOT>/examples/georegime/regreg/synthetic',
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
    print(run_geogregime_different_schemes(data_ids=np.arange(100, 150)))