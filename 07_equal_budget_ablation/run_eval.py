"""Equal-compute baseline + component ablations (Round-2, Reviewer 2 comment 2).

Adapted from Round-1/reanalysis/run_clean_simulation_eval.py — same label-free
harness, external metrics, and 150-grid protocol.

Responds to AAAG Round-1 reviewer concerns about true-label leakage:
  - NO algorithm ever receives `true_label` (neither as a kwarg nor via the
    `run_two_stage_kmeans._true_label` attribute hack found in some evolved programs).
  - All metrics (SSR, RandI, ARI, NMI, MAE, SSR reduction, variance ratio) are
    computed EXTERNALLY by this script, never inside candidate code.
  - Adds Adjusted Rand Index (ARI) per Reviewer 1, and per-pattern mean +/- SD plus
    Wilcoxon signed-rank tests vs the seed per Reviewer 1/2.

Protocol matches the paper runs (GASE/03_simulation_experiment):
  - 150 grids (grid_{0..149}h.txt), 3 patterns x 50 grids, p=5, min_size=10
  - evolved variants: max_iter=10000, init_stoc_step=True
  - classical baselines: max_iter=300, per-dataset seeds from default_rng(42)

Outputs (in ./results_clean/):
  clean_per_dataset.csv     one row per (algorithm, grid)
  clean_per_pattern.csv     mean/SD per (algorithm, pattern)
  clean_overall.csv         mean/SD per algorithm over all 150 grids
  wilcoxon_vs_seed.csv      paired tests (per-grid SSR and ARI) vs the 2KModels seed
  labels_{algo}.npz         predicted labels per grid (for future metric recomputation)
  comparison_vs_paper.csv   clean numbers next to the paper's Table 3 values
"""
import argparse
import os
import sys
import time
import warnings
import importlib.util
import inspect
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATASETS_DIR = os.path.join(REPO, "03_simulation_experiment", "datasets")
EVOLVED_DIR = os.path.join(HERE, "programs")
BASELINES_DIR = os.path.join(REPO, "02_case_study_algorithms", "classical_baselines")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _safe(name):
    """File-name-safe form of an algorithm label.

    The labels contain "*" (as in "A* w/o border refinement"), which Windows
    forbids in file names -- leaving it in makes the repository impossible to
    check out there. Also drops the other characters Windows rejects.
    """
    out = name.replace("*", "star").replace("/", "_").replace(" ", "_")
    return out.translate(str.maketrans("", "", '?<>:"|\\'))

SIDE = 25
N_REGIONS = 5
MIN_SIZE = 10
EVOLVED_MAX_ITER = 10000
BASELINE_MAX_ITER = 300

PATTERNS = [
    ("Rectangular", list(range(0, 50))),
    ("Voronoi", list(range(50, 100))),
    ("Arbitrary", list(range(100, 150))),
]

# (display_name, kind, file)
ALGORITHMS = [
    ("2kmodels (seed)",               "evolved", "seed_plain.py"),
    ("Seed + equal-compute search",   "evolved", "seed_gridsearch.py"),
    ("Seed + matched-budget search",  "evolved", "seed_gridsearch_budget.py"),
    ("GeoEvolve A* (full)",           "evolved", "evolved_plain.py"),
    ("A* w/o initial-k search",       "evolved", "evolved_no_ksearch.py"),
    ("A* w/o border refinement",      "evolved", "evolved_no_refine.py"),
    ("A* w/o spatial penalty",        "evolved", "evolved_no_penalty.py"),
    ("A* w/o KMeans init",            "evolved", "evolved_random_init.py"),
]

# Paper Table 3 values (leaky-era runs) for side-by-side comparison.
PAPER_TABLE3 = {
    "SKATER-reg":                  {"ssr": 403.22, "randi": 0.715},
    "RegionKModels":               {"ssr": 348.12, "randi": 0.800},
    "GWR+SKATER":                  {"ssr": 236.17, "randi": 0.865},
    "AZP":                         {"ssr": 212.15, "randi": 0.824},
    "OpenEvolve_NoGeoKnow":        {"ssr": 97.12,  "randi": 0.932},
    "2kmodels (seed)":             {"ssr": 81.03,  "randi": 0.929},
    "GeoEvolve_NoRAG":             {"ssr": 78.82,  "randi": 0.932},
    "OpenEvolve_SpecificGeoKnow":  {"ssr": 78.53,  "randi": 0.935},
    "OpenEvolve_GeneralGeoKnow":   {"ssr": 71.09,  "randi": 0.939},
    "GeoEvolve_StaticRAG":         {"ssr": 59.16,  "randi": 0.963},
}


# ------------------------------------------------------------------
# Data loading (same format as the paper runners)
# ------------------------------------------------------------------
def input_data(path, side=SIDE):
    with open(path) as f:
        raw_lines = [ln.rstrip("\n\r") for ln in f.readlines()]
    mats = [np.zeros((side, side), dtype=float) for _ in range(6)]
    idx, mat_idx = 0, 0
    while mat_idx < 6:
        block = raw_lines[idx: idx + side]
        idx += side
        for r in range(side):
            tokens = block[r].strip().split()
            row_vals = list(map(float, tokens)) if mat_idx != 3 else list(map(int, tokens))
            mats[mat_idx][r, :] = row_vals
        mat_idx += 1
        while idx < len(raw_lines) and not raw_lines[idx].strip():
            idx += 1
    x1, x2, y, region, b1, b2 = mats
    X = np.column_stack((x1.flatten(), x2.flatten()))
    Y = y.flatten()
    label = region.flatten().astype(int)
    coeff = np.column_stack((b1.flatten(), b2.flatten()))
    return X, Y, label, coeff


# ------------------------------------------------------------------
# External metrics (candidate code never touches these)
# ------------------------------------------------------------------
def external_metrics(X, Y, true_label, pred_label, true_coeff):
    from sklearn import metrics as skm
    pred_label = np.asarray(pred_label).astype(int)

    ssr = 0.0
    region_y_means, total_n = [], len(Y)
    within_var_sum = 0.0
    coeffs_per_region = {}
    for r in np.unique(pred_label):
        mask = pred_label == r
        Xr, Yr = X[mask], Y[mask]
        region_y_means.append(np.mean(Yr))
        within_var_sum += len(Yr) * np.var(Yr)
        if Xr.shape[0] >= Xr.shape[1]:
            beta = np.linalg.pinv(Xr).dot(Yr)
            resid = Yr - Xr.dot(beta)
            ssr += float(np.sum(resid ** 2))
            coeffs_per_region[r] = beta
        else:
            coeffs_per_region[r] = np.zeros(Xr.shape[1])

    beta_g = np.linalg.pinv(X).dot(Y)
    ssr_global = float(np.sum((Y - X.dot(beta_g)) ** 2))
    ssr_reduction = 1.0 - ssr / ssr_global if ssr_global > 0 else 0.0

    within_var = within_var_sum / max(total_n, 1)
    between_var = np.var(region_y_means) if len(region_y_means) > 1 else 0.0
    var_ratio = between_var / (within_var + between_var) if (within_var + between_var) > 0 else 0.0

    pred_coeff = np.array([coeffs_per_region[pred_label[i]] for i in range(len(pred_label))])
    mae = float(np.mean(np.abs(true_coeff - pred_coeff)))

    return {
        "ssr": ssr,
        "randi": float(skm.rand_score(true_label, pred_label)),
        "ari": float(skm.adjusted_rand_score(true_label, pred_label)),
        "nmi": float(skm.normalized_mutual_info_score(true_label, pred_label)),
        "mae": mae,
        "ssr_reduction": ssr_reduction,
        "var_ratio": var_ratio,
        "n_regions_found": int(len(np.unique(pred_label))),
    }


# ------------------------------------------------------------------
# Worker: loads one algorithm module, runs one grid, label-free
# ------------------------------------------------------------------
_WORKER_CACHE = {}


def _load_module(kind, fname):
    key = (kind, fname)
    if key in _WORKER_CACHE:
        return _WORKER_CACHE[key]
    if kind == "baseline":
        # skater_reg.py does `from utils import set_endog`; alias spreg.utils
        if "utils" not in sys.modules:
            import spreg.utils as _spreg_utils
            sys.modules["utils"] = _spreg_utils
        # classical_baselines is a package with relative imports (_common.py)
        pkg_parent = os.path.dirname(BASELINES_DIR)
        if pkg_parent not in sys.path:
            sys.path.insert(0, pkg_parent)
        import importlib as _il
        mod = _il.import_module(f"classical_baselines.{fname[:-3]}")
    else:
        path = os.path.join(EVOLVED_DIR, fname)
        spec = importlib.util.spec_from_file_location(f"clean_{fname[:-3]}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    _WORKER_CACHE[key] = mod
    return mod


def run_one(task):
    """task = (algo_name, kind, fname, pattern, dataset_id, baseline_seed)"""
    algo_name, kind, fname, pattern, did, bseed = task
    data_path = os.path.join(DATASETS_DIR, f"grid_{did}h.txt")
    X, Y, true_label, true_coeff = input_data(data_path)

    mod = _load_module(kind, fname)

    t0 = time.time()
    try:
        if kind == "evolved":
            fn = mod.run_two_stage_kmeans
            # --- anti-leak guards ---
            if hasattr(fn, "_true_label"):
                del fn._true_label
            import libpysal
            w = libpysal.weights.lat2W(SIDE, SIDE)
            kwargs = dict(X=X, Y=Y, p=N_REGIONS, w=w, min_size=MIN_SIZE,
                          max_iter=EVOLVED_MAX_ITER, init_stoc_step=True, verbose=False)
            # Explicitly never pass true_label; if the signature accepts it, its
            # default (None) forces the label-free code path.
            assert "true_label" not in kwargs
            pred = fn(**kwargs)
        else:
            fn = getattr(mod, "run", None) or getattr(mod, "run_skater_reg")
            import libpysal
            w = libpysal.weights.lat2W(SIDE, SIDE)
            sig = inspect.signature(fn)
            kwargs = dict(X=X, Y=Y, n_regions=N_REGIONS, w=w,
                          min_size=MIN_SIZE, max_iter=BASELINE_MAX_ITER)
            if "seed" in sig.parameters:
                kwargs["seed"] = bseed
            if "coords" in sig.parameters:
                coords = np.array([[i % SIDE, i // SIDE] for i in range(SIDE * SIDE)], dtype=float)
                kwargs["coords"] = coords
            if "init_stoc_step" in sig.parameters and "init_stoc_step" not in kwargs:
                kwargs["init_stoc_step"] = True
            pred = fn(**kwargs)
        elapsed = time.time() - t0
    except Exception as e:
        return {"algorithm": algo_name, "pattern": pattern, "dataset_id": did,
                "error": f"{type(e).__name__}: {e}", "time": time.time() - t0}

    m = external_metrics(X, Y, true_label, np.asarray(pred), true_coeff)
    m.update({"algorithm": algo_name, "pattern": pattern, "dataset_id": did,
              "time": elapsed, "error": ""})
    m["pred_label"] = np.asarray(pred).astype(int)
    return m


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    import pandas as pd
    from scipy.stats import wilcoxon

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--algorithms", nargs="+", default=None,
                    help="Subset of algorithm display names")
    ap.add_argument("--limit-per-pattern", type=int, default=None)
    args = ap.parse_args()

    selected = [a for a in ALGORITHMS
                if args.algorithms is None or a[0] in args.algorithms]

    per_dataset_csv = os.path.join(RESULTS_DIR, "clean_per_dataset.csv")
    done = set()
    if os.path.exists(per_dataset_csv):
        prev = pd.read_csv(per_dataset_csv)
        done = set(zip(prev["algorithm"], prev["dataset_id"]))
        print(f"Resuming: {len(done)} (algo, grid) pairs already recorded")

    all_rows = []
    t_start = time.time()
    for algo_name, kind, fname in selected:
        # reproduce the paper's per-dataset baseline seed sequence
        rng = np.random.default_rng(42)
        tasks = []
        for pat, ids in PATTERNS:
            ids2 = ids[:args.limit_per_pattern] if args.limit_per_pattern else ids
            for did in ids2:
                bseed = int(rng.integers(0, 1_000_000))
                if (algo_name, did) in done:
                    continue
                tasks.append((algo_name, kind, fname, pat, did, bseed))
        if not tasks:
            print(f"=== {algo_name}: all grids already done, skipping")
            continue
        print(f"=== {algo_name}: {len(tasks)} grids on {args.workers} workers", flush=True)
        t0 = time.time()
        label_store = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(run_one, tasks)):
                if res.get("error"):
                    print(f"  FAIL grid {res['dataset_id']}: {res['error']}", flush=True)
                else:
                    label_store[str(res["dataset_id"])] = res.pop("pred_label")
                all_rows.append(res)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(tasks)} grids", flush=True)
        # persist incrementally per algorithm
        df_new = pd.DataFrame([r for r in all_rows if r["algorithm"] == algo_name])
        header = not os.path.exists(per_dataset_csv)
        df_new.to_csv(per_dataset_csv, mode="a", header=header, index=False)
        if label_store:
            np.savez_compressed(
                os.path.join(RESULTS_DIR, f"labels_{_safe(algo_name)}.npz"),
                **label_store)
        print(f"=== {algo_name} done in {time.time()-t0:.0f}s", flush=True)

    # ---------------- aggregation ----------------
    df = pd.read_csv(per_dataset_csv)
    df = df[df["error"].isna() | (df["error"] == "")]
    metric_cols = ["ssr", "randi", "ari", "nmi", "mae", "ssr_reduction", "var_ratio", "time"]

    per_pattern = (df.groupby(["algorithm", "pattern"])[metric_cols]
                     .agg(["mean", "std"]).round(4))
    per_pattern.columns = ["_".join(c) for c in per_pattern.columns]
    per_pattern.reset_index().to_csv(os.path.join(RESULTS_DIR, "clean_per_pattern.csv"), index=False)

    overall = (df.groupby("algorithm")[metric_cols]
                 .agg(["mean", "std"]).round(4))
    overall.columns = ["_".join(c) for c in overall.columns]
    overall = overall.sort_values("ssr_mean", ascending=False)
    overall.reset_index().to_csv(os.path.join(RESULTS_DIR, "clean_overall.csv"), index=False)

    # Wilcoxon signed-rank vs seed (paired per grid)
    seed_name = "2kmodels (seed)"
    rows = []
    if seed_name in set(df["algorithm"]):
        seed_df = df[df["algorithm"] == seed_name].set_index("dataset_id")
        for algo in df["algorithm"].unique():
            if algo == seed_name:
                continue
            a = df[df["algorithm"] == algo].set_index("dataset_id")
            common = seed_df.index.intersection(a.index)
            if len(common) < 10:
                continue
            for col in ("ssr", "ari"):
                stat, p = wilcoxon(a.loc[common, col], seed_df.loc[common, col])
                rows.append({"algorithm": algo, "metric": col, "n": len(common),
                             "median_diff": float((a.loc[common, col] - seed_df.loc[common, col]).median()),
                             "wilcoxon_p": p})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "wilcoxon_vs_seed.csv"), index=False)

    # side-by-side with paper Table 3
    comp = []
    for algo, paper in PAPER_TABLE3.items():
        if algo in overall.index:
            comp.append({
                "algorithm": algo,
                "paper_ssr": paper["ssr"], "clean_ssr": overall.loc[algo, "ssr_mean"],
                "paper_randi": paper["randi"], "clean_randi": overall.loc[algo, "randi_mean"],
                "clean_ari": overall.loc[algo, "ari_mean"],
            })
    pd.DataFrame(comp).to_csv(os.path.join(RESULTS_DIR, "comparison_vs_paper.csv"), index=False)

    print("\n===== CLEAN OVERALL (sorted worst->best by SSR) =====")
    print(overall[["ssr_mean", "ssr_std", "randi_mean", "randi_std",
                   "ari_mean", "ari_std", "nmi_mean", "time_mean"]].to_string())
    print(f"\nTotal wall-clock: {(time.time()-t_start)/60:.1f} min")
    print(f"Outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
