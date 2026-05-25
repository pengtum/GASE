# EVOLVE-BLOCK-START
from scipy.sparse import csgraph as cg
from scipy.optimize import OptimizeWarning
from collections import namedtuple
from warnings import warn
from libpysal.weights import w_subset
from utils import set_endog
import time
import numpy as np
import copy
from sklearn.metrics import euclidean_distances
from sklearn import metrics
import spreg
import libpysal

deletion = namedtuple("deletion", ("in_node", "out_node", "score"))

class Skater_reg(object):
    def __init__(
            self,
            dissimilarity=euclidean_distances,
            affinity=None,
            reduction=np.sum,
            center=np.mean,
    ):
        if affinity is not None:
            metric = lambda x: -np.log(affinity(x))
        else:
            metric = dissimilarity
        self.metric = metric
        self.reduction = reduction
        self.center = center

    def fit(
            self,
            n_clusters,
            W,
            data=None,
            data_reg=None,
            quorum=-np.inf,
            trace=True,
            islands="increase",
            verbose=False,
            model_family="spreg",
    ):
        if trace:
            self._trace = []
        if data is None:
            attribute_kernel = np.ones((W.n, W.n))
            data = np.ones((W.n, 1))
        else:
            attribute_kernel = self.metric(data)

        if data.shape[0] <= n_clusters * quorum:
            raise ValueError("The number of observations is less than the number of clusters times the quorum.")

        W.transform = "b"
        W = W.sparse
        start = time.time()

        super_verbose = verbose > 1
        start_W = time.time()
        dissim = W.multiply(attribute_kernel)
        dissim.eliminate_zeros()
        end_W = time.time() - start_W

        if super_verbose:
            print("Computing Affinity Kernel took {:.2f}s".format(end_W))

        tree_time = time.time()
        MSF = cg.minimum_spanning_tree(dissim)
        tree_time = time.time() - tree_time
        if super_verbose:
            print("Computing initial MST took {:.2f}s".format(tree_time))

        initial_component_time = time.time()
        current_n_subtrees, current_labels = cg.connected_components(
            MSF, directed=False
        )
        initial_component_time = time.time() - initial_component_time

        if super_verbose:
            print(
                "Computing connected components took {:.2f}s.".format(
                    initial_component_time
                )
            )

        if current_n_subtrees > 1:
            island_warnings = [
                "Increasing `n_clusters` from {} to {} in order to account for islands.".format(
                    n_clusters, n_clusters + current_n_subtrees
                ),
                "Counting islands towards the remaining {} clusters.".format(
                    n_clusters - (current_n_subtrees)
                ),
            ]
            ignoring_islands = int(islands.lower() == "ignore")
            chosen_warning = island_warnings[ignoring_islands]
            warn(
                "By default, the graph is disconnected! {}".format(chosen_warning),
                OptimizeWarning,
                stacklevel=2,
            )
            if not ignoring_islands:
                n_clusters += current_n_subtrees
            _, island_populations = np.unique(current_labels, return_counts=True)
            if (island_populations < quorum).any():
                raise ValueError(
                    "Islands must be larger than the quorum. If not, drop the small islands and solve for"
                    " clusters in the remaining field."
                )
        if trace:
            self._trace.append(([], deletion(np.nan, np.nan, np.inf)))
            if super_verbose:
                print(self._trace[-1])
        trees_scores = None
        prev_score = np.inf
        while current_n_subtrees < n_clusters:  # while we don't have enough regions
            (
                best_deletion,
                trees_scores,
                new_MSF,
                current_n_subtrees,
                current_labels,
            ) = self.find_cut(
                MSF,
                data,
                data_reg,
                current_n_subtrees,
                current_labels,
                quorum=quorum,
                trees_scores=trees_scores,
                labels=None,
                target_label=None,
                verbose=verbose,
                model_family=model_family,
            )

            if np.isfinite(best_deletion.score):  # if our search succeeds
                # accept the best move as *the* move
                if super_verbose:
                    print("cut made {}...".format(best_deletion))
                if best_deletion.score > prev_score:
                    raise ValueError(
                        ("The score increased with the number of clusters. "
                         "Please check your data.\nquorum: {}; n_clusters: {}"
                         ).format(quorum, n_clusters)
                    )
                prev_score = best_deletion.score
                MSF = new_MSF
            else:  # otherwise, it means the MSF admits no further cuts
                prev_n_subtrees, _ = cg.connected_components(MSF, directed=False)
                warn(
                    "MSF contains no valid moves after finding {} subtrees. "
                    "Decrease the size of your quorum to find the remaining {} subtrees.".format(
                        prev_n_subtrees, n_clusters - prev_n_subtrees
                    ),
                    OptimizeWarning,
                    stacklevel=2,
                )
            if trace:
                self._trace.append((current_labels, best_deletion))

        self.current_labels_ = current_labels
        self.minimum_spanning_forest_ = MSF
        self._elapsed_time = time.time() - start
        return self

    def score_spreg(
            self,
            data=None,
            data_reg=None,
            all_labels=None,
            quorum=-np.inf,
            current_labels=None,
            current_tree=None,
    ):
        labels, subtree_quorums = self._prep_score(all_labels, current_tree, current_labels)
        if (subtree_quorums < quorum).any():
            return np.inf, None

        set_labels = set(labels)
        if data_reg is not None:
            kargs = {k: v for k, v in data_reg.items() if k not in ["reg", "y", "x", "w", "x_nd", "yend", "q"]}
            trees_scores = {}

            if data_reg["reg"].__name__ in {"GM_Lag", "BaseGM_Lag"}:
                try:
                    x = np.hstack((np.ones((data_reg["x"].shape[0], 1)), data_reg["x"]))
                except np.linalg.LinAlgError:
                    x = _const_x(data_reg["x"])
                from .twosls_regimes import TSLS_Regimes
                reg = TSLS_Regimes(
                    y=data_reg["y"],
                    x=x,
                    yend=data_reg.get("yend"),
                    q=data_reg.get("q"),
                    regimes=all_labels,
                )
                score = np.dot(reg.u.T, reg.u)[0][0]
            else:
                label_indices = {l: np.where(all_labels == l)[0] for l in set_labels}

                for l in set_labels:
                    regi_ids = label_indices[l]

                    if "w" in data_reg:
                        w_ids = list(map(data_reg["w"].id_order.__getitem__, regi_ids))
                        kargs["w"] = w_subset(data_reg["w"], w_ids, silence_warnings=True)

                    x = data_reg["x"][regi_ids]
                    y = data_reg["y"][regi_ids]
                    yend = data_reg["yend"][regi_ids] if "yend" in data_reg else None
                    q = data_reg["q"][regi_ids] if "q" in data_reg else None

                    temp_vars = {"x": x, "yend": yend, "q": q}
                    for key in ["x", "yend", "q"]:
                        mat = temp_vars[key]
                        if mat is not None and mat.size > 0 and np.linalg.matrix_rank(mat) < mat.shape[1]:
                            _, r = np.linalg.qr(mat)
                            small_diag_indices = np.abs(np.diag(r)) < 1e-10
                            temp_vars[key] = mat[:, ~small_diag_indices]

                    x = temp_vars["x"]
                    try:
                        x = np.hstack((np.ones((x.shape[0], 1)), x))
                    except np.linalg.LinAlgError:
                        x = _const_x(x)

                    if temp_vars["yend"] is not None:
                        kargs["yend"] = temp_vars["yend"]
                        kargs["q"] = temp_vars["q"]

                    reg = data_reg["reg"](y=y, x=x, **kargs)
                    trees_scores[l] = np.dot(reg.u.T, reg.u)[0][0]

                score = sum(trees_scores.values())
        else:
            part_scores, score, trees_scores = self._data_reg_none(data, all_labels, set_labels)

        return score, trees_scores

    def score_stats(
            self,
            data=None,
            data_reg=None,
            all_labels=None,
            quorum=-np.inf,
            current_labels=None,
            current_tree=None,
    ):
        labels, subtree_quorums = self._prep_score(
            all_labels, current_tree, current_labels
        )
        if (subtree_quorums < quorum).any():
            return np.inf, None
        set_labels = set(labels)
        if data_reg is not None:
            kargs = {
                k: v
                for k, v in data_reg.items()
                if k not in ["reg", "y", "x", "w", "x_nd"]
            }
            trees_scores = {}
            for l in set_labels:
                x = data_reg["x"][all_labels == l]
                if np.linalg.matrix_rank(x) < x.shape[1]:
                    small_diag_indices = np.abs(np.diag(np.linalg.qr(x)[1])) < 1e-10
                    x = x[:, ~small_diag_indices]

                try:
                    x = np.hstack((np.ones((x.shape[0], 1)), x))
                    reg = data_reg["reg"](
                        data_reg["y"][all_labels == l], x, **kargs
                    ).fit()
                except np.linalg.LinAlgError:
                    x = _const_x(x)
                    reg = data_reg["reg"](
                        data_reg["y"][all_labels == l], x, **kargs
                    ).fit()

                trees_scores[l] = np.sum(reg.resid ** 2)
            score = sum(trees_scores.values())
        else:
            part_scores, score, trees_scores = self._data_reg_none(
                data, all_labels, set_labels
            )
        return score, trees_scores

    def _prep_score(self, all_labels, current_tree, current_labels):
        if all_labels is None:
            try:
                labels = self.current_labels_
            except AttributeError:
                raise ValueError(
                    "Labels not provided and MSF_Prune object has not been fit to data yet."
                )
        if current_tree is not None:
            labels = all_labels[current_labels == current_tree]
        _, subtree_quorums = np.unique(labels, return_counts=True)
        return labels, subtree_quorums

    def _data_reg_none(self, data, all_labels, set_labels):
        assert data.shape[0] == len(
            all_labels
        ), "Length of label array ({}) does not match " "length of data ({})! ".format(
            all_labels.shape[0], data.shape[0]
        )
        part_scores = [
            self.reduction(
                self.metric(
                    X=data[all_labels == l],
                    Y=self.center(data[all_labels == l], axis=0).reshape(1, -1),
                )
            )
            for l in set_labels
        ]

        score = self.reduction(part_scores).item()
        trees_scores = {l: part_scores[i] for i, l in enumerate(set_labels)}
        return part_scores, score, trees_scores

    def _prep_lag(self, data_reg):
        # if the model is a spatial lag, add the lagged dependent variable to the model
        data_reg['yend'], data_reg['q'] = set_endog(data_reg["y"], data_reg["x"][:, 1:], data_reg["w"], yend=None,
                                                    q=None, w_lags=1, lag_q=True)
        return data_reg

    def find_cut(
            self,
            MSF,
            data=None,
            data_reg=None,
            current_n_subtrees=None,
            current_labels=None,
            quorum=-np.inf,
            trees_scores=None,
            labels=None,
            target_label=None,
            make=False,
            verbose=False,
            model_family="spreg",
    ):
        if data is None:
            data = np.ones(MSF.shape)

        if (labels is None) != (target_label is None):
            raise ValueError(
                "Both labels and target_label must be supplied! Only {} provided.".format(
                    ["labels", "target_label"][int(target_label is None)]
                )
            )
        if verbose:
            try:
                from tqdm import tqdm
            except ImportError:

                def tqdm(noop, desc=""):
                    return noop

        else:

            def tqdm(noop, desc=""):
                return noop

        zero_in = (labels is not None) and (target_label is not None)
        best_deletion = deletion(np.nan, np.nan, np.inf)
        best_d_score = -np.inf

        try:
            if data_reg["reg"].__name__ == "GM_Lag" or data_reg["reg"].__name__ == "BaseGM_Lag":
                data_reg = self._prep_lag(data_reg)
        except:
            pass

        try:
            old_score = sum(trees_scores.values())
        except:
            pass
        best_scores = {}
        current_list = current_labels.tolist()
        for in_node, out_node in tqdm(
                np.vstack(MSF.nonzero()).T, desc="finding cut..."
        ):  # iterate over MSF edges
            if zero_in:
                if labels[in_node] != target_label:
                    continue

            local_MSF = copy.deepcopy(MSF)
            # delete a candidate edge
            local_MSF[in_node, out_node] = 0
            local_MSF.eliminate_zeros()
            current_tree = current_labels[in_node]

            # get the connected components
            local_n_subtrees, local_labels = cg.connected_components(
                local_MSF, directed=False
            )

            if local_n_subtrees <= current_n_subtrees:
                raise Exception("Malformed MSF!")

            # compute the score of these components
            if model_family == "spreg":
                new_score, new_trees_scores = self.score_spreg(
                    data, data_reg, local_labels, quorum, current_labels, current_tree
                )
            elif model_family == "statsmodels":
                new_score, new_trees_scores = self.score_stats(
                    data, data_reg, local_labels, quorum, current_labels, current_tree
                )
            else:
                raise ValueError("Model family must be either spreg or statsmodels.")

            if np.isfinite(new_score):
                try:
                    d_score = trees_scores[current_tree] - new_score
                    score = old_score - d_score
                except:
                    d_score = -new_score
                    score = new_score
                # if the d_score is greater than the best score and quorum is met
                if d_score > best_d_score:
                    best_deletion = deletion(in_node, out_node, score)
                    best_d_score = d_score
                    try:
                        for i in set(current_labels):
                            best_scores[
                                local_labels[current_list.index(i)]
                            ] = trees_scores[i]
                        for i in new_trees_scores:
                            best_scores[i] = new_trees_scores[i]
                    except:
                        best_scores = new_trees_scores
                    best_MSF = local_MSF
                    best_labels = local_labels
        try:
            return best_deletion, best_scores, best_MSF, local_n_subtrees, best_labels
        except UnboundLocalError:  # in case no solution is found
            return deletion(None, None, np.inf), np.inf, None, np.inf, None


def _const_x(x):
    x = x[:, np.ptp(x, axis=0) != 0]
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    return x

def fit_equations(X, Y, regions_list):
    coeffs = []
    for units in regions_list:
        if len(units) == 0:
            coeffs.append(None)
            continue
        Xr = X[units, :]
        Yr = Y[units]
        beta = np.linalg.pinv(Xr).dot(Yr)
        coeffs.append(beta)
    return coeffs

def skater_reg(Xarr, Yarr, n_regions, w, min_size=None):
    nobs = Xarr.shape[0]
    nvar = Xarr.shape[1] - 1

    Xreg = np.asarray([Xarr[u, 1:] for u in range(nobs)])
    Xreg = Xreg.reshape((nobs, nvar))
    Yreg = np.asarray([Yarr[u] for u in range(nobs)])
    Yreg = Yreg.reshape((nobs, 1))
    results = Skater_reg().fit(n_clusters=n_regions, W=w, data=Xreg,
              data_reg={'reg': spreg.OLS, 'y': Yreg, 'x': Xreg}, quorum=min_size)

    label = results.current_labels_
    units = np.arange(w.n).astype(int)
    regions = [units[label == r].tolist() for r in range(n_regions)]
    coeffs = fit_equations(Xarr, Yarr, regions)
    return label, coeffs

def run_skater_reg(X, Y, n_regions, w, min_size=10, max_iter=10000, init_stoc_step=True, verbose=False):
    label, coeffs = skater_reg(
        X, Y, n_regions, w,
        min_size=min_size
    )
    return label

# EVOLVE-BLOCK-END
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

def run_georegime_once(data_path):
    side = 25
    n_regions = 5
    min_size = 10
    with open(data_path) as f:
        X, Y, true_label, true_coeff = input_data(f, side)
    w = libpysal.weights.lat2W(side, side)

    start = time.time()
    labels = run_skater_reg(
            X, Y,
            n_regions=n_regions,
            w=w,
            min_size=min_size,
            max_iter=10000,
            # init_stoc_step=True,
            verbose=False
        )
    end = time.time()
    used_time = end - start
    metric = RegionMetrics(X, Y, true_label, labels, true_coeff)
    return metric.ssr, metric.randi, metric.nmi, metric.mae, used_time

def run_geogregime_different_schemes(base_path  = 'D:/Research/GeoEvolve/GeoEvolveWithDynamicRAG/examples/georegime/regreg/synthetic',
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


