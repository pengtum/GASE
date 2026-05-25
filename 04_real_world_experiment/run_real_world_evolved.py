"""Run the 7 two-stage-kmeans variants (initial 2k-models seed + 6 evolved
variants) on the real-world datasets, mirroring run_real_world_baselines.py.

Outputs are written under results_evolved/ and labels are saved per dataset
in <ds>_labels.npz so subsequent figure scripts can reuse them.
"""
import importlib.util
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import libpysal
from libpysal.weights import KNN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "all_datasets")
OUTPUT_DIR = os.path.join(BASE, "results_evolved")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROOT = os.path.dirname(BASE)
CODES = os.path.join(ROOT, "codes")

ALGO_FILES = [
    ("2kmodels (initial)",         os.path.join(CODES, "initial_program.py")),
    ("OpenEvolve_NoGeoKnow",       os.path.join(CODES, "best_program_openevolve_without_geo_know.py")),
    ("OpenEvolve_SimpleGeoKnow",   os.path.join(CODES, "best_program_openevolve_with_simple_geo_know.py")),
    ("OpenEvolve_SpecificGeoKnow", os.path.join(CODES, "best_program_openevolve_with_specific_geo_know.py")),
    ("GeoEvolve_NoRAG",            os.path.join(CODES, "best_program_geoevolve_without_rag.py")),
    ("GeoEvolve_StaticRAG",        os.path.join(CODES, "best_program_geoevolve_static_rag.py")),
    ("GeoEvolve_DynamicRAG",       os.path.join(CODES, "best_program_geoevolve_dynamic_rag.py")),
]

MAX_SAMPLES = 2000
K_NEIGHBORS = 6
SKIP_DATASETS = {"US_Forest_FIA", "US_Climate_ERA5_CENSUS", "US_Climate_ERA5_STATE"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    is_proj = False
    if "proj_x" in df.columns:
        df = df.rename(columns={"proj_x": "coord_x", "proj_y": "coord_y"}); is_proj = True
    elif "proj_X" in df.columns:
        df = df.rename(columns={"proj_X": "coord_x", "proj_Y": "coord_y"}); is_proj = True
    elif "lat" in df.columns and "lon" in df.columns:
        df["coord_x"] = df["lon"]; df["coord_y"] = df["lat"]
    else:
        raise ValueError(f"no coord cols in {csv_path}")
    x_cols = sorted([c for c in df.columns if c.startswith("X") and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
    return df, x_cols, is_proj


def identify_continuous(df, x_cols, min_unique=15):
    return [c for c in x_cols
            if df[c].dtype != "object" and df[c].nunique() >= min_unique]


def prepare(df, x_cols, is_proj):
    x_cols = identify_continuous(df, x_cols)
    if not x_cols:
        raise ValueError("no continuous X cols")
    key = ["Y"] + x_cols + ["coord_x", "coord_y"]
    d = df.dropna(subset=key).copy()
    if not is_proj:
        cx, cy = d["coord_x"], d["coord_y"]
        if (cx.min() < -130 or cy.max() > 55) and (cx.min() > -200 and cx.max() < 0):
            d = d[(cx > -130) & (cx < -60) & (cy > 23) & (cy < 52)].reset_index(drop=True)
    if len(d) > MAX_SAMPLES:
        d = d.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)
    Y = d["Y"].values
    coords = d[["coord_x", "coord_y"]].values.astype(float)
    regions_orig = d["region"].values if "region" in d.columns else None
    Xraw = d[x_cols].values.astype(float)
    Xs = StandardScaler().fit_transform(Xraw)
    X = np.column_stack([np.ones(len(Xs)), Xs])
    return X, Y, coords, regions_orig, x_cols


def determine_p(regions_orig):
    if regions_orig is None:
        return 5
    n = len(np.unique(regions_orig))
    if n <= 15:
        return n
    if n <= 30:
        return min(n, 10)
    return 8


def build_w(coords):
    w = KNN.from_array(coords, k=K_NEIGHBORS)
    w.transform = "r"
    return w


def compute_summary(X, Y, labels):
    intra = {}
    total_ssr = 0.0
    for r in np.unique(labels):
        mask = labels == r
        Xr, Yr = X[mask], Y[mask]
        n = len(Yr)
        if n < X.shape[1]:
            intra[r] = {"n": n, "y_mean": float(np.mean(Yr)) if n > 0 else np.nan,
                        "y_std": float(np.std(Yr)) if n > 0 else np.nan,
                        "ssr": np.nan, "r2": np.nan, "rmse": np.nan}
            continue
        beta = np.linalg.pinv(Xr).dot(Yr)
        resid = Yr - Xr.dot(beta)
        ssr = float(np.sum(resid ** 2))
        sst = float(np.sum((Yr - np.mean(Yr)) ** 2))
        r2 = 1 - ssr / sst if sst > 0 else np.nan
        intra[r] = {"n": n, "y_mean": float(np.mean(Yr)),
                    "y_std": float(np.std(Yr)), "ssr": ssr,
                    "r2": float(r2) if not np.isnan(r2) else np.nan,
                    "rmse": float(np.sqrt(ssr / n))}
        total_ssr += ssr
    beta_g = np.linalg.pinv(X).dot(Y)
    resid_g = Y - X.dot(beta_g)
    ssr_g = float(np.sum(resid_g ** 2))
    sst_g = float(np.sum((Y - np.mean(Y)) ** 2))
    summary = {
        "n_regions": int(len(np.unique(labels))),
        "global_ssr": ssr_g,
        "total_ssr": total_ssr,
        "ssr_reduction": 1 - total_ssr / ssr_g if ssr_g > 0 else np.nan,
        "global_r2": 1 - ssr_g / sst_g if sst_g > 0 else np.nan,
    }
    return intra, summary


def main():
    csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")])
    print(f"Found {len(csv_files)} datasets")

    modules = []
    for name, path in ALGO_FILES:
        try:
            mod = load_module(f"alg_{name}", path)
            modules.append((name, mod))
            print(f"  loaded {name}")
        except Exception as e:
            print(f"  FAILED to load {name}: {e}")

    rows = []
    saved = {}
    for f in csv_files:
        ds = os.path.splitext(f)[0]
        if ds in SKIP_DATASETS:
            continue
        print(f"\n=== {ds} ===", flush=True)
        try:
            df, xcols, is_proj = load_dataset(os.path.join(DATASET_DIR, f))
            X, Y, coords, regions_orig, xcols = prepare(df, xcols, is_proj)
            print(f"  shape X={X.shape}, features={len(xcols)}")
            w = build_w(coords)
            p = determine_p(regions_orig)
            print(f"  p={p}")

            ds_results = {}
            for algo, mod in modules:
                t0 = time.time()
                try:
                    min_size = max(X.shape[1], 10)
                    labels = mod.run_two_stage_kmeans(
                        X, Y, p=p, w=w, min_size=min_size, max_iter=300,
                        init_stoc_step=True, verbose=False)
                    elapsed = time.time() - t0
                    intra, summary = compute_summary(X, Y, labels)
                    ds_results[algo] = {"labels": labels, "runtime": elapsed,
                                         "intra": intra, "summary": summary}
                    r2_vals = [intra[r]["r2"] for r in intra if not np.isnan(intra[r]["r2"])]
                    rmse_vals = [intra[r]["rmse"] for r in intra if not np.isnan(intra[r]["rmse"])]
                    avg_r2 = float(np.mean(r2_vals)) if r2_vals else np.nan
                    avg_rmse = float(np.mean(rmse_vals)) if rmse_vals else np.nan
                    total_n = sum(intra[r]["n"] for r in intra)
                    within = sum(intra[r]["n"] * intra[r]["y_std"] ** 2 for r in intra
                                 if not np.isnan(intra[r]["y_std"])) / max(total_n, 1)
                    means = [intra[r]["y_mean"] for r in intra if not np.isnan(intra[r]["y_mean"])]
                    between = float(np.var(means)) if len(means) > 1 else 0.0
                    vr = between / (within + between) if (within + between) > 0 else 0.0
                    rows.append({
                        "Dataset": ds, "Algorithm": algo,
                        "N_Regions": summary["n_regions"],
                        "SSR_Global": summary["global_ssr"],
                        "SSR_Regime": summary["total_ssr"],
                        "SSR_Reduction": summary["ssr_reduction"],
                        "Avg_R2": avg_r2, "Avg_RMSE": avg_rmse,
                        "Variance_Ratio": vr, "Global_R2": summary["global_r2"],
                        "Runtime_s": elapsed,
                    })
                    print(f"  {algo}: time={elapsed:.1f}s, k={summary['n_regions']}, "
                          f"SSR_red={summary['ssr_reduction']:.3f}, AvgR2={avg_r2:.3f}",
                          flush=True)
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {algo}: FAIL ({elapsed:.1f}s): {e}", flush=True)
                    rows.append({
                        "Dataset": ds, "Algorithm": algo,
                        "N_Regions": np.nan, "SSR_Global": np.nan, "SSR_Regime": np.nan,
                        "SSR_Reduction": np.nan, "Avg_R2": np.nan, "Avg_RMSE": np.nan,
                        "Variance_Ratio": np.nan, "Global_R2": np.nan,
                        "Runtime_s": elapsed,
                    })

            saved[ds] = ds_results
            pd.DataFrame(rows).to_csv(
                os.path.join(OUTPUT_DIR, "algorithm_comparison.csv"), index=False)
            np.savez(os.path.join(OUTPUT_DIR, f"{ds}_labels.npz"),
                     coords=coords,
                     **{algo: ds_results[algo]["labels"] for algo in ds_results})
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    to_pickle = {ds: {a: {"runtime": ar["runtime"], "summary": ar["summary"]}
                      for a, ar in dr.items()} for ds, dr in saved.items()}
    with open(os.path.join(OUTPUT_DIR, "results_summary.pkl"), "wb") as f:
        pickle.dump(to_pickle, f)
    print(f"\nWrote {OUTPUT_DIR}/algorithm_comparison.csv with {len(rows)} rows")


if __name__ == "__main__":
    main()
