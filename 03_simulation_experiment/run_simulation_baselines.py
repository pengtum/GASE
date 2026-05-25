"""Run all 5 baseline algorithms on every grid_*h.txt simulation dataset.

Per-dataset metrics: SSR (within-region OLS), RandIndex, NMI, MAE on coefficients,
and runtime. Per-spatial-pattern means are computed at the end.

Spatial patterns:
    - Rectangular: dataset ids 0-49
    - Voronoi:     dataset ids 50-99
    - Arbitrary:   dataset ids 100-149

Results are written to:
    simulation_study/results/baseline_per_dataset.csv      (one row per (algo, dataset))
    simulation_study/results/baseline_per_pattern.csv      (averaged per pattern x algo)
And the 'performance' sheet of ModelTime.xlsx is updated with new rows.
"""
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import libpysal
from sklearn import metrics

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "codes"))

from baselines import azp, region_k_models, gwr_skater, skater_reg, shap_based  # noqa: E402

DATASETS_DIR = os.path.join(BASE, "simulation_study", "datasets")
RESULTS_DIR = os.path.join(BASE, "simulation_study", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PER_DATASET_CSV = os.path.join(RESULTS_DIR, "baseline_per_dataset.csv")
PER_PATTERN_CSV = os.path.join(RESULTS_DIR, "baseline_per_pattern.csv")

PATTERNS = [
    ("Rectangular", range(0, 50)),
    ("Voronoi", range(50, 100)),
    ("Arbitrary", range(100, 150)),
]

ALGORITHMS = {
    "AZP": (azp.run, dict()),
    "RegionKModels": (region_k_models.run, dict()),
    "SKATER-reg": (skater_reg.run, dict()),
    "GWR+SKATER": (gwr_skater.run, dict()),  # coords injected per call
    "SHAP-based": (shap_based.run, dict()),  # coords injected per call
}

NEEDS_COORDS = {"GWR+SKATER", "SHAP-based"}

SIDE = 25
N_REGIONS = 5
MIN_SIZE = 10
MAX_ITER = 300


def input_data(file_obj, side=SIDE):
    raw_lines = [ln.rstrip('\n\r') for ln in file_obj.readlines()]
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
    Xarr = np.column_stack((x1.flatten(), x2.flatten()))
    Yarr = y.flatten()
    label = region.flatten().astype(int)
    coeff = np.column_stack((b1.flatten(), b2.flatten()))
    return Xarr, Yarr, label, coeff


def fit_coeffs(X, Y, labels):
    """Per-region OLS coefficients (one row per unit, broadcast)."""
    pred = np.zeros_like(np.column_stack([X[:, 0], X[:, 0]]) if X.shape[1] == 2 else X[:, :2])
    coeffs_per_region = {}
    for r in np.unique(labels):
        mask = labels == r
        Xr, Yr = X[mask], Y[mask]
        if Xr.shape[0] < Xr.shape[1]:
            beta = np.zeros(Xr.shape[1])
        else:
            beta = np.linalg.pinv(Xr).dot(Yr)
        coeffs_per_region[r] = beta
    pred_coeff = np.array([coeffs_per_region[labels[i]] for i in range(len(labels))])
    return pred_coeff, coeffs_per_region


def metrics_for_pred(X, Y, true_label, pred_label, true_coeff):
    pred_coeff, _ = fit_coeffs(X, Y, pred_label)
    # SSR
    ssr = 0.0
    for r in np.unique(pred_label):
        mask = pred_label == r
        Xr, Yr = X[mask], Y[mask]
        if Xr.shape[0] < Xr.shape[1]:
            continue
        beta = np.linalg.pinv(Xr).dot(Yr)
        resid = Yr - Xr.dot(beta)
        ssr += float(np.sum(resid ** 2))
    randi = float(metrics.rand_score(true_label, pred_label))
    nmi = float(metrics.normalized_mutual_info_score(true_label, pred_label))
    mae = float(np.mean(np.abs(true_coeff - pred_coeff)))
    return ssr, randi, nmi, mae


def load_existing(csv_path):
    """Return set of (algorithm, dataset_id) already recorded."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["algorithm"], int(row["dataset_id"])))
    return done


def append_row(csv_path, row):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "pattern", "dataset_id",
            "ssr", "randi", "nmi", "mae", "time"
        ])
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS.keys()),
                   help="Subset of algorithms to run")
    p.add_argument("--limit-per-pattern", type=int, default=None,
                   help="If set, only run on the first N datasets of each pattern")
    return p.parse_args()


def main():
    args = parse_args()
    selected = [a for a in args.algorithms if a in ALGORITHMS]
    print(f"Running algorithms: {selected}")
    print(f"Resuming from: {PER_DATASET_CSV}")
    done = load_existing(PER_DATASET_CSV)
    print(f"Already-completed (algo,dataset) pairs: {len(done)}")

    coords = np.array([[i % SIDE, i // SIDE] for i in range(SIDE * SIDE)], dtype=float)
    w = libpysal.weights.lat2W(SIDE, SIDE)

    rng = np.random.default_rng(42)

    total_start = time.time()
    for algo_name in selected:
        fn, _ = ALGORITHMS[algo_name]
        print(f"\n=== {algo_name} ===", flush=True)
        algo_start = time.time()
        for pat_name, ids in PATTERNS:
            ids_list = list(ids)
            if args.limit_per_pattern:
                ids_list = ids_list[:args.limit_per_pattern]
            for did in ids_list:
                key = (algo_name, did)
                if key in done:
                    continue
                path = os.path.join(DATASETS_DIR, f"grid_{did}h.txt")
                if not os.path.exists(path):
                    print(f"  [skip] missing dataset {did}")
                    continue
                with open(path) as f:
                    X, Y, true_label, true_coeff = input_data(f)

                kwargs = dict(coords=coords) if algo_name in NEEDS_COORDS else {}
                seed = int(rng.integers(0, 1_000_000))
                t0 = time.time()
                try:
                    labels = fn(X, Y, n_regions=N_REGIONS, w=w,
                                min_size=MIN_SIZE, max_iter=MAX_ITER,
                                seed=seed, **kwargs)
                    elapsed = time.time() - t0
                    ssr, randi, nmi, mae = metrics_for_pred(
                        X, Y, true_label, labels, true_coeff)
                    row = {
                        "algorithm": algo_name, "pattern": pat_name,
                        "dataset_id": did, "ssr": ssr, "randi": randi,
                        "nmi": nmi, "mae": mae, "time": elapsed
                    }
                    append_row(PER_DATASET_CSV, row)
                    done.add(key)
                    print(f"  {pat_name} d={did}: time={elapsed:.2f}s "
                          f"ssr={ssr:.2f} randi={randi:.3f} nmi={nmi:.3f} mae={mae:.3f}",
                          flush=True)
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {pat_name} d={did}: FAIL ({elapsed:.2f}s): {e}", flush=True)
                    row = {
                        "algorithm": algo_name, "pattern": pat_name,
                        "dataset_id": did, "ssr": float('nan'), "randi": float('nan'),
                        "nmi": float('nan'), "mae": float('nan'), "time": elapsed
                    }
                    append_row(PER_DATASET_CSV, row)
                    done.add(key)
        print(f"=== {algo_name} done in {time.time() - algo_start:.1f}s ===", flush=True)

    # Aggregate
    print("\nAggregating per-pattern averages...")
    summary = {}  # algo -> pattern -> list of metric dicts
    with open(PER_DATASET_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row["algorithm"]
            pat = row["pattern"]
            if algo not in selected:
                continue
            try:
                rec = {k: float(row[k]) for k in ("ssr", "randi", "nmi", "mae", "time")}
            except ValueError:
                continue
            if any(np.isnan(v) for v in rec.values()):
                continue
            summary.setdefault(algo, {}).setdefault(pat, []).append(rec)

    with open(PER_PATTERN_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "pattern", "n_runs",
                         "ssr_mean", "randi_mean", "nmi_mean", "mae_mean", "time_mean"])
        for algo in selected:
            if algo not in summary:
                continue
            for pat_name, _ in PATTERNS:
                recs = summary[algo].get(pat_name, [])
                if not recs:
                    continue
                writer.writerow([
                    algo, pat_name, len(recs),
                    np.mean([r["ssr"] for r in recs]),
                    np.mean([r["randi"] for r in recs]),
                    np.mean([r["nmi"] for r in recs]),
                    np.mean([r["mae"] for r in recs]),
                    np.mean([r["time"] for r in recs]),
                ])
    print(f"Saved: {PER_PATTERN_CSV}")
    print(f"Total elapsed: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
