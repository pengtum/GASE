"""
P-sensitivity analysis: Test different numbers of regions (p) for each dataset
and each algorithm, then plot line charts showing how metrics change with p.
"""

import os
import sys
import time
import warnings
import pickle
import importlib.util

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import libpysal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'all_datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'p_sensitivity')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGO_FILES = {
    'Initial': 'initial_program.py',
    'OpenEvolve': 'best_program_openevolve.py',
    'GeoEvolve': 'best_program_geoevolve.py',
}
ALGO_COLORS = {'Initial': '#1f77b4', 'OpenEvolve': '#ff7f0e', 'GeoEvolve': '#2ca02c'}
ALGO_MARKERS = {'Initial': 'o', 'OpenEvolve': 's', 'GeoEvolve': '^'}

MAX_SAMPLES = 2000
K_NEIGHBORS = 6
SKIP_DATASETS = ['US_Forest_FIA']

# Range of p values to test
P_VALUES = [3, 4, 5, 6, 7, 8, 10, 12]

US_STATES_PATH = os.path.join(BASE_DIR, 'us_states_conus.gpkg')


# ============================================================
# Utility
# ============================================================
def save_fig(output_dir, name, dpi=150):
    plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.close()


def load_us_basemap():
    if os.path.exists(US_STATES_PATH):
        return gpd.read_file(US_STATES_PATH)
    try:
        us_states = gpd.read_file(
            'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip')
        exclude = ['AK', 'HI', 'AS', 'GU', 'MP', 'PR', 'VI']
        conus = us_states[~us_states['STUSPS'].isin(exclude)]
        conus.to_file(US_STATES_PATH, driver='GPKG')
        return conus
    except Exception:
        return None


def add_basemap(ax, us_states, is_latlon=True):
    if us_states is not None and is_latlon:
        us_states.boundary.plot(ax=ax, color='#666666', linewidth=0.4, alpha=0.5)
        us_states.plot(ax=ax, color='#f0f0f0', edgecolor='#999999', linewidth=0.3, alpha=0.2)


def is_geographic(coords):
    return (coords[:, 0].min() > -130 and coords[:, 0].max() < -60 and
            coords[:, 1].min() > 23 and coords[:, 1].max() < 52)


# ============================================================
# Load algorithms
# ============================================================
def load_algorithm(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_all_algorithms():
    algos = {}
    for name, fname in ALGO_FILES.items():
        fpath = os.path.join(BASE_DIR, fname)
        algos[name] = load_algorithm(f'algo_{name}', fpath)
    return algos


# ============================================================
# Data loading
# ============================================================
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    if 'proj_x' in df.columns:
        df = df.rename(columns={'proj_x': 'coord_x', 'proj_y': 'coord_y'})
    elif 'proj_X' in df.columns:
        df = df.rename(columns={'proj_X': 'coord_x', 'proj_Y': 'coord_y'})
    elif 'lat' in df.columns and 'lon' in df.columns:
        df['coord_x'] = df['lon']
        df['coord_y'] = df['lat']
    else:
        raise ValueError(f"Cannot find coordinate columns in {csv_path}")
    x_cols = sorted([c for c in df.columns if c.startswith('X') and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
    return df, x_cols


def prepare_data(df, x_cols, max_samples=MAX_SAMPLES):
    key_cols = ['Y'] + x_cols + ['coord_x', 'coord_y']
    df_clean = df.dropna(subset=key_cols).copy()

    # Filter Alaska/Hawaii
    if 'coord_x' in df_clean.columns and 'coord_y' in df_clean.columns:
        cx, cy = df_clean['coord_x'], df_clean['coord_y']
        looks_like_latlon = (cx.min() > -200 and cx.max() < 0 and
                             cy.min() > 0 and cy.max() < 80)
        if looks_like_latlon and (cx.min() < -130 or cy.max() > 55):
            conus_mask = (cx > -130) & (cx < -60) & (cy > 23) & (cy < 52)
            n_before = len(df_clean)
            df_clean = df_clean[conus_mask].reset_index(drop=True)
            n_removed = n_before - len(df_clean)
            if n_removed > 0:
                print(f"  Filtered {n_removed} non-CONUS points")

    sampled = False
    if len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, random_state=42).reset_index(drop=True)
        sampled = True

    Y = df_clean['Y'].values
    coords = df_clean[['coord_x', 'coord_y']].values

    for col in x_cols:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'str':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    X_raw = df_clean[x_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    return X, Y, coords, df_clean, sampled


def build_spatial_weights(coords, k=K_NEIGHBORS):
    from libpysal.weights import KNN
    w = KNN.from_array(coords, k=k)
    w.transform = 'r'
    return w


# ============================================================
# Metrics
# ============================================================
def compute_metrics(X, Y, labels):
    """Compute SSR reduction, avg R², within/between variance ratio, avg RMSE."""
    unique_labels = np.unique(labels)
    n_regions = len(unique_labels)

    # Global OLS
    beta_g = np.linalg.pinv(X).dot(Y)
    resid_g = Y - X.dot(beta_g)
    ssr_global = np.sum(resid_g ** 2)

    # Per-region OLS
    total_ssr = 0
    r2_list = []
    rmse_list = []
    y_means = []
    region_sizes = []

    for r in unique_labels:
        mask = labels == r
        Xr, Yr = X[mask], Y[mask]
        n_r = len(Yr)
        region_sizes.append(n_r)
        y_means.append(np.mean(Yr))

        if n_r < X.shape[1]:
            continue
        beta = np.linalg.pinv(Xr).dot(Yr)
        resid = Yr - Xr.dot(beta)
        ssr = np.sum(resid ** 2)
        sst = np.sum((Yr - np.mean(Yr)) ** 2)
        r2 = 1 - ssr / sst if sst > 0 else 0
        total_ssr += ssr
        r2_list.append(r2)
        rmse_list.append(np.sqrt(ssr / n_r))

    ssr_reduction = 1 - total_ssr / ssr_global if ssr_global > 0 else 0
    avg_r2 = np.mean(r2_list) if r2_list else 0
    avg_rmse = np.mean(rmse_list) if rmse_list else 0

    # Within / Between variance ratio
    total_n = sum(region_sizes)
    within_var = 0
    for i, r in enumerate(unique_labels):
        mask = labels == r
        within_var += region_sizes[i] * np.var(Y[mask])
    within_var /= max(total_n, 1)
    between_var = np.var(y_means) if len(y_means) > 1 else 0
    var_ratio = between_var / (within_var + between_var) if (within_var + between_var) > 0 else 0

    return {
        'ssr_reduction': ssr_reduction,
        'avg_r2': avg_r2,
        'avg_rmse': avg_rmse,
        'var_ratio': var_ratio,
        'n_regions': n_regions,
    }


# ============================================================
# Visualization
# ============================================================
def plot_p_sensitivity_single(ds_name, p_values, results_by_algo, output_dir):
    """
    Line plot for one dataset: metrics vs p for each algorithm.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    metrics = [
        ('ssr_reduction', 'SSR Reduction', axes[0, 0]),
        ('avg_r2', 'Average R² per Region', axes[0, 1]),
        ('avg_rmse', 'Average RMSE per Region', axes[0, 2]),
        ('var_ratio', 'Between/Total Variance Ratio', axes[1, 0]),
        ('n_regions', 'Actual # Regions', axes[1, 1]),
        ('runtime', 'Runtime (seconds)', axes[1, 2]),
    ]

    for metric_key, metric_label, ax in metrics:
        for algo in results_by_algo:
            ps = [r['p'] for r in results_by_algo[algo]]
            vals = [r[metric_key] for r in results_by_algo[algo]]
            ax.plot(ps, vals, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                    label=algo, linewidth=2, markersize=7, alpha=0.85)
        ax.set_xlabel('Number of Regions (p)', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(p_values)

    short_name = ds_name.replace('US_', '').replace('_', ' ')
    fig.suptitle(f'P-Sensitivity Analysis: {short_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, f'{ds_name}_p_sensitivity')


def plot_p_sensitivity_all_datasets(all_p_results, p_values, output_dir):
    """
    Multi-panel line plot: one row per metric, one line per algorithm,
    averaged across all datasets with error bands.
    """
    algos = list(ALGO_FILES.keys())
    metric_keys = ['ssr_reduction', 'avg_r2', 'avg_rmse', 'var_ratio']
    metric_labels = ['SSR Reduction', 'Average R²', 'Average RMSE', 'Between/Total Var. Ratio']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flat

    for idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[idx]
        for algo in algos:
            # Collect metric values at each p across datasets
            mean_vals = []
            std_vals = []
            for p in p_values:
                vals_at_p = []
                for ds in all_p_results:
                    if algo in all_p_results[ds]:
                        for r in all_p_results[ds][algo]:
                            if r['p'] == p:
                                vals_at_p.append(r[mkey])
                if vals_at_p:
                    mean_vals.append(np.mean(vals_at_p))
                    std_vals.append(np.std(vals_at_p))
                else:
                    mean_vals.append(np.nan)
                    std_vals.append(0)

            mean_vals = np.array(mean_vals)
            std_vals = np.array(std_vals)
            ax.plot(p_values, mean_vals, marker=ALGO_MARKERS[algo],
                    color=ALGO_COLORS[algo], label=algo, linewidth=2.5, markersize=8)
            ax.fill_between(p_values, mean_vals - std_vals, mean_vals + std_vals,
                            color=ALGO_COLORS[algo], alpha=0.12)

        ax.set_xlabel('Number of Regions (p)', fontsize=11)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(p_values)

    fig.suptitle('P-Sensitivity: Mean ± Std Across All Datasets',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, 'all_datasets_p_sensitivity_mean')


def plot_p_sensitivity_grid(all_p_results, p_values, output_dir):
    """
    Grid figure: each subplot is one dataset, showing SSR reduction vs p for all algos.
    """
    datasets = list(all_p_results.keys())
    algos = list(ALGO_FILES.keys())
    n_ds = len(datasets)
    ncols = 4
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, ds in enumerate(datasets):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        for algo in algos:
            if algo in all_p_results[ds]:
                ps = [r['p'] for r in all_p_results[ds][algo]]
                vals = [r['ssr_reduction'] for r in all_p_results[ds][algo]]
                ax.plot(ps, vals, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                        label=algo, linewidth=1.5, markersize=5)
        short = ds.replace('US_', '').replace('_', ' ')
        ax.set_title(short, fontsize=9, fontweight='bold')
        ax.set_xlabel('p', fontsize=8)
        ax.set_ylabel('SSR Red.', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(p_values)
        ax.tick_params(labelsize=7)

    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('SSR Reduction vs p Across All Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, 'all_datasets_ssr_vs_p_grid')


def plot_p_sensitivity_runtime_grid(all_p_results, p_values, output_dir):
    """Grid: runtime vs p for all datasets."""
    datasets = list(all_p_results.keys())
    algos = list(ALGO_FILES.keys())
    n_ds = len(datasets)
    ncols = 4
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, ds in enumerate(datasets):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        for algo in algos:
            if algo in all_p_results[ds]:
                ps = [r['p'] for r in all_p_results[ds][algo]]
                vals = [r['runtime'] for r in all_p_results[ds][algo]]
                ax.plot(ps, vals, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                        label=algo, linewidth=1.5, markersize=5)
        short = ds.replace('US_', '').replace('_', ' ')
        ax.set_title(short, fontsize=9, fontweight='bold')
        ax.set_xlabel('p', fontsize=8)
        ax.set_ylabel('Runtime (s)', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(p_values)
        ax.tick_params(labelsize=7)

    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Runtime vs p Across All Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, 'all_datasets_runtime_vs_p_grid')


def plot_p_regime_maps(coords, labels_by_p, p_values_used, algo_name, ds_name, us_states, output_dir):
    """
    Map visualization showing how regimes change as p increases (for one algo, one dataset).
    """
    geo = is_geographic(coords)
    n_p = len(p_values_used)
    ncols = min(n_p, 4)
    nrows = (n_p + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, p in enumerate(p_values_used):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        add_basemap(ax, us_states, geo)
        labels = labels_by_p[p]
        n_reg = len(np.unique(labels))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10',
                   s=6, alpha=0.85, edgecolors='white', linewidths=0.1)
        ax.set_title(f'p={p} ({n_reg} regions)', fontsize=11, fontweight='bold')
        if geo:
            ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
            ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
        ax.tick_params(labelsize=7)

    for idx in range(n_p, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    short = ds_name.replace('US_', '').replace('_', ' ')
    fig.suptitle(f'{algo_name} Regimes at Different p: {short}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, f'{ds_name}_{algo_name}_regime_maps_by_p')


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("P-Sensitivity Analysis")
    print("=" * 60)

    us_states = load_us_basemap()
    algos = load_all_algorithms()
    print(f"Loaded {len(algos)} algorithms")
    print(f"Testing p values: {P_VALUES}")

    csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
    all_p_results = {}

    for csv_file in csv_files:
        ds_name = os.path.splitext(csv_file)[0]
        if ds_name in SKIP_DATASETS:
            print(f"\nSkipping {ds_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {ds_name}")
        print(f"{'=' * 60}")

        try:
            csv_path = os.path.join(DATASET_DIR, csv_file)
            df, x_cols = load_dataset(csv_path)
            X, Y, coords, df_clean, sampled = prepare_data(df, x_cols)
            if sampled:
                print(f"  Subsampled to {len(Y)} observations")
            print(f"  Shape: X={X.shape}, Y={Y.shape}")

            print("  Building spatial weights...")
            w = build_spatial_weights(coords)

            results_by_algo = {}

            for algo_name, algo_module in algos.items():
                results_by_algo[algo_name] = []
                labels_by_p = {}

                for p in P_VALUES:
                    min_size = max(X.shape[1], 10)
                    # Skip p values that are too large relative to data
                    if p * min_size > len(Y) * 0.8:
                        print(f"  {algo_name} p={p}: skipped (too large for data)")
                        continue

                    print(f"  {algo_name} p={p}...", end=' ', flush=True)
                    try:
                        start = time.time()
                        labels = algo_module.run_two_stage_kmeans(
                            X, Y, p=p, w=w, min_size=min_size,
                            max_iter=300, init_stoc_step=True, verbose=False)
                        elapsed = time.time() - start
                        metrics = compute_metrics(X, Y, labels)
                        metrics['p'] = p
                        metrics['runtime'] = elapsed
                        results_by_algo[algo_name].append(metrics)
                        labels_by_p[p] = labels
                        print(f"Done in {elapsed:.1f}s, {metrics['n_regions']} regions, "
                              f"SSR_red={metrics['ssr_reduction']:.4f}")
                    except Exception as e:
                        print(f"FAILED: {e}")

                # Plot regime maps across p for GeoEvolve
                if algo_name == 'GeoEvolve' and labels_by_p:
                    plot_p_regime_maps(coords, labels_by_p,
                                       sorted(labels_by_p.keys()),
                                       algo_name, ds_name, us_states, OUTPUT_DIR)

            all_p_results[ds_name] = results_by_algo

            # Per-dataset line plots
            if any(results_by_algo[a] for a in results_by_algo):
                valid_ps = sorted(set(
                    r['p'] for a in results_by_algo for r in results_by_algo[a]))
                plot_p_sensitivity_single(ds_name, valid_ps, results_by_algo, OUTPUT_DIR)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'p_sensitivity_results.pkl'), 'wb') as f:
        pickle.dump(all_p_results, f)

    # Cross-dataset visualizations
    print("\nGenerating cross-dataset p-sensitivity visualizations...")
    valid_ps = sorted(set(
        r['p'] for ds in all_p_results for a in all_p_results[ds]
        for r in all_p_results[ds][a]))
    plot_p_sensitivity_all_datasets(all_p_results, valid_ps, OUTPUT_DIR)
    plot_p_sensitivity_grid(all_p_results, valid_ps, OUTPUT_DIR)
    plot_p_sensitivity_runtime_grid(all_p_results, valid_ps, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 100)
    print("P-SENSITIVITY SUMMARY")
    print("=" * 100)
    print(f"{'Dataset':<30} {'Algorithm':<12} {'Best_p':>6} {'Best_SSR_red':>12} {'p_range':>10}")
    print("-" * 100)
    for ds in all_p_results:
        for algo in all_p_results[ds]:
            results = all_p_results[ds][algo]
            if not results:
                continue
            best = max(results, key=lambda r: r['ssr_reduction'])
            ps = [r['p'] for r in results]
            print(f"{ds:<30} {algo:<12} {best['p']:>6} {best['ssr_reduction']:>12.4f} "
                  f"  {min(ps)}-{max(ps)}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
