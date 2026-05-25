"""Run the 7 two-stage-kmeans variants (the 2k-models seed + 3 OpenEvolve +
3 GeoEvolve evolved versions) under the *same* simulation protocol used for
baselines: 50 datasets per spatial pattern, h-noise only.

Each module exposes `run_two_stage_kmeans(X, Y, p, w, min_size, max_iter,
init_stoc_step, verbose)`.

Outputs:
  simulation_study/results/evolved_per_dataset.csv
  simulation_study/results/evolved_per_pattern.csv
"""
import argparse
import csv
import importlib.util
import os
import sys
import time
import warnings

import numpy as np
import libpysal
from sklearn import metrics

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODES = os.path.join(BASE, "codes")
DATASETS_DIR = os.path.join(BASE, "simulation_study", "datasets")
RESULTS_DIR = os.path.join(BASE, "simulation_study", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PER_DATASET_CSV = os.path.join(RESULTS_DIR, "evolved_per_dataset.csv")
PER_PATTERN_CSV = os.path.join(RESULTS_DIR, "evolved_per_pattern.csv")

PATTERNS = [
    ("Rectangular", range(0, 50)),
    ("Voronoi", range(50, 100)),
    ("Arbitrary", range(100, 150)),
]

# (display_name, source_file)
ALGORITHMS = [
    ("2kmodels (initial)",            "initial_program.py"),
    ("OpenEvolve_NoGeoKnow",          "best_program_openevolve_without_geo_know.py"),
    ("OpenEvolve_SimpleGeoKnow",      "best_program_openevolve_with_simple_geo_know.py"),
    ("OpenEvolve_SpecificGeoKnow",    "best_program_openevolve_with_specific_geo_know.py"),
    ("GeoEvolve_NoRAG",               "best_program_geoevolve_without_rag.py"),
    ("GeoEvolve_StaticRAG",           "best_program_geoevolve_static_rag.py"),
    ("GeoEvolve_DynamicRAG",          "best_program_geoevolve_dynamic_rag.py"),
]

SIDE = 25
N_REGIONS = 5
MIN_SIZE = 10
MAX_ITER = 10000
INIT_STOC_STEP = True


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    return pred_coeff


def metrics_for_pred(X, Y, true_label, pred_label, true_coeff):
    pred_coeff = fit_coeffs(X, Y, pred_label)
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
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
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
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="Subset of algorithm display names to run")
    p.add_argument("--limit-per-pattern", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    selected = args.algorithms or [name for name, _ in ALGORITHMS]
    print(f"Running algorithms: {selected}")

    modules = {}
    for display, fname in ALGORITHMS:
        if display not in selected:
            continue
        path = os.path.join(CODES, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        try:
            modules[display] = load_module(f"alg_{display}", path)
            print(f"  loaded {display}: {fname}")
        except Exception as e:
            print(f"  FAILED to load {display}: {e}")

    print(f"Resuming from: {PER_DATASET_CSV}")
    done = load_existing(PER_DATASET_CSV)
    print(f"Already-completed (algo,dataset) pairs: {len(done)}")

    w = libpysal.weights.lat2W(SIDE, SIDE)
    rng = np.random.default_rng(42)

    total_start = time.time()
    for algo_name in selected:
        if algo_name not in modules:
            continue
        mod = modules[algo_name]
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
                    continue
                with open(path) as f:
                    X, Y, true_label, true_coeff = input_data(f)
                t0 = time.time()
                try:
                    labels = mod.run_two_stage_kmeans(
                        X, Y, p=N_REGIONS, w=w,
                        min_size=MIN_SIZE, max_iter=MAX_ITER,
                        init_stoc_step=INIT_STOC_STEP, verbose=False)
                    elapsed = time.time() - t0
                    ssr, randi, nmi, mae = metrics_for_pred(
                        X, Y, true_label, labels, true_coeff)
                    row = {"algorithm": algo_name, "pattern": pat_name,
                           "dataset_id": did, "ssr": ssr, "randi": randi,
                           "nmi": nmi, "mae": mae, "time": elapsed}
                    append_row(PER_DATASET_CSV, row)
                    done.add(key)
                    print(f"  {pat_name} d={did}: time={elapsed:.2f}s "
                          f"ssr={ssr:.2f} randi={randi:.3f} nmi={nmi:.3f} mae={mae:.3f}",
                          flush=True)
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {pat_name} d={did}: FAIL ({elapsed:.2f}s): {e}", flush=True)
                    row = {"algorithm": algo_name, "pattern": pat_name,
                           "dataset_id": did, "ssr": float('nan'),
                           "randi": float('nan'), "nmi": float('nan'),
                           "mae": float('nan'), "time": elapsed}
                    append_row(PER_DATASET_CSV, row)
                    done.add(key)
        print(f"=== {algo_name} done in {time.time() - algo_start:.1f}s ===",
              flush=True)

    # Aggregate
    print("\nAggregating per-pattern averages...")
    summary = {}
    with open(PER_DATASET_CSV) as f:
        for row in csv.DictReader(f):
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
