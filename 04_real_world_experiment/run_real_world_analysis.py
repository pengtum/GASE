"""
Real-world dataset analysis for Spatial Regime (Two-Stage K-Models).

This script:
1. Loads each CSV dataset
2. Runs the three algorithm versions (initial, openevolve, geoevolve)
3. Computes intra-region and inter-region statistics
4. Generates comprehensive visualizations with US map basemap
5. Compares algorithm performance
"""

import os
import sys
import time
import warnings
import pickle
import importlib.util

import numpy as np
import pandas as pd
import libpysal
import geopandas as gpd
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'all_datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGO_FILES = {
    'Initial': 'initial_program.py',
    'OpenEvolve': 'best_program_openevolve.py',
    'GeoEvolve': 'best_program_geoevolve.py',
}
ALGO_COLORS = {'Initial': '#1f77b4', 'OpenEvolve': '#ff7f0e', 'GeoEvolve': '#2ca02c'}

MAX_SAMPLES = 2000
K_NEIGHBORS = 6
SKIP_DATASETS = ['US_Forest_FIA']  # Skip: limited geographic coverage

# US States basemap (download once)
US_STATES_PATH = os.path.join(BASE_DIR, 'us_states_conus.gpkg')


# ============================================================
# Utility
# ============================================================
def save_fig(output_dir, name, dpi=150):
    """Save current figure as both PNG and PDF."""
    plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.close()


def load_us_basemap():
    """Load US contiguous states boundary for map overlay."""
    if os.path.exists(US_STATES_PATH):
        return gpd.read_file(US_STATES_PATH)
    try:
        us_states = gpd.read_file(
            'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip')
        exclude = ['AK', 'HI', 'AS', 'GU', 'MP', 'PR', 'VI']
        conus = us_states[~us_states['STUSPS'].isin(exclude)]
        conus.to_file(US_STATES_PATH, driver='GPKG')
        return conus
    except Exception as e:
        print(f"  Warning: Could not load US basemap: {e}")
        return None


def add_basemap(ax, us_states, is_latlon=True):
    """Add US state boundary basemap to axis."""
    if us_states is not None and is_latlon:
        us_states.boundary.plot(ax=ax, color='#666666', linewidth=0.4, alpha=0.5)
        us_states.plot(ax=ax, color='#f0f0f0', edgecolor='#999999', linewidth=0.3, alpha=0.2)


def is_geographic(coords):
    """Check if coordinates appear to be lat/lon (CONUS range)."""
    return (coords[:, 0].min() > -130 and coords[:, 0].max() < -60 and
            coords[:, 1].min() > 23 and coords[:, 1].max() < 52)


def get_distinct_colors(n):
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    elif n <= 20:
        return [plt.cm.tab20(i) for i in range(n)]
    else:
        return [plt.cm.gist_ncar(i / max(n - 1, 1)) for i in range(n)]


# ============================================================
# Load algorithm modules
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
# Data loading and preprocessing
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

    # Filter out Alaska and Hawaii if lat/lon coordinates are present
    if 'coord_x' in df_clean.columns and 'coord_y' in df_clean.columns:
        cx = df_clean['coord_x']
        cy = df_clean['coord_y']
        # Only filter if coordinates look like geographic lat/lon (not projected)
        looks_like_latlon = (cx.min() > -200 and cx.max() < 0 and
                             cy.min() > 0 and cy.max() < 80)
        if looks_like_latlon and (cx.min() < -130 or cy.max() > 55):
            conus_mask = (cx > -130) & (cx < -60) & (cy > 23) & (cy < 52)
            n_before = len(df_clean)
            df_clean = df_clean[conus_mask].reset_index(drop=True)
            n_removed = n_before - len(df_clean)
            if n_removed > 0:
                print(f"  Filtered {n_removed} non-CONUS points (Alaska/Hawaii)")

    sampled = False
    if len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, random_state=42).reset_index(drop=True)
        sampled = True

    Y = df_clean['Y'].values
    coords = df_clean[['coord_x', 'coord_y']].values
    regions_orig = df_clean['region'].values if 'region' in df_clean.columns else None

    for col in x_cols:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'str':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    X_raw = df_clean[x_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    return X, Y, coords, regions_orig, df_clean, sampled


def build_spatial_weights(coords, k=K_NEIGHBORS):
    from libpysal.weights import KNN
    w = KNN.from_array(coords, k=k)
    w.transform = 'r'
    return w


def determine_p(regions_orig):
    if regions_orig is None:
        return 5
    n_orig = len(np.unique(regions_orig))
    if n_orig <= 15:
        return n_orig
    elif n_orig <= 30:
        return min(n_orig, 10)
    else:
        return min(n_orig, 8)


# ============================================================
# Run algorithms
# ============================================================
def run_single_algorithm(algo_module, X, Y, p, w, min_size=None, max_iter=300):
    if min_size is None:
        min_size = max(X.shape[1], 10)
    start = time.time()
    labels = algo_module.run_two_stage_kmeans(
        X, Y, p=p, w=w, min_size=min_size, max_iter=max_iter,
        init_stoc_step=True, verbose=False)
    elapsed = time.time() - start
    return labels, elapsed


# ============================================================
# Metrics computation
# ============================================================
def compute_region_regression(X, Y, labels):
    unique_labels = np.unique(labels)
    results = {}
    for r in unique_labels:
        mask = labels == r
        Xr, Yr = X[mask], Y[mask]
        n_r = len(Yr)
        if n_r < X.shape[1]:
            results[r] = {'n': n_r, 'beta': None, 'residuals': None,
                          'ssr': np.nan, 'r2': np.nan, 'rmse': np.nan,
                          'y_mean': np.mean(Yr) if n_r > 0 else np.nan,
                          'y_std': np.std(Yr) if n_r > 0 else np.nan}
            continue
        beta = np.linalg.pinv(Xr).dot(Yr)
        resid = Yr - Xr.dot(beta)
        ssr = np.sum(resid ** 2)
        sst = np.sum((Yr - np.mean(Yr)) ** 2)
        r2 = 1 - ssr / sst if sst > 0 else np.nan
        results[r] = {'n': n_r, 'beta': beta, 'residuals': resid,
                      'ssr': ssr, 'r2': r2, 'rmse': np.sqrt(ssr / n_r),
                      'y_mean': np.mean(Yr), 'y_std': np.std(Yr)}
    return results


def compute_global_regression(X, Y):
    beta = np.linalg.pinv(X).dot(Y)
    resid = Y - X.dot(beta)
    ssr = np.sum(resid ** 2)
    sst = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else np.nan
    return {'beta': beta, 'ssr': ssr, 'r2': r2, 'rmse': np.sqrt(ssr / len(Y)), 'residuals': resid}


def compute_intra_inter_metrics(X, Y, labels):
    region_stats = compute_region_regression(X, Y, labels)
    unique_labels = sorted(region_stats.keys())

    intra = {}
    for r in unique_labels:
        rs = region_stats[r]
        intra[r] = {k: rs[k] for k in ['n', 'y_mean', 'y_std', 'ssr', 'r2', 'rmse', 'beta']}

    betas, valid_labels = [], []
    for r in unique_labels:
        if region_stats[r]['beta'] is not None:
            betas.append(region_stats[r]['beta'])
            valid_labels.append(r)
    betas = np.array(betas) if betas else np.array([])

    coeff_std = np.std(betas, axis=0) if len(betas) > 1 else np.zeros(X.shape[1])
    coeff_range = np.ptp(betas, axis=0) if len(betas) > 1 else np.zeros(X.shape[1])

    inter = {'coeff_std_per_feature': coeff_std, 'coeff_range_per_feature': coeff_range,
             'betas': betas, 'valid_labels': valid_labels}

    total_ssr = sum(rs['ssr'] for rs in region_stats.values() if not np.isnan(rs['ssr']))
    global_reg = compute_global_regression(X, Y)
    summary = {
        'total_ssr': total_ssr, 'global_ssr': global_reg['ssr'],
        'ssr_reduction': 1 - total_ssr / global_reg['ssr'] if global_reg['ssr'] > 0 else np.nan,
        'global_r2': global_reg['r2'], 'n_regions': len(unique_labels),
    }
    return intra, inter, summary, global_reg


# ============================================================
# Visualization: Per-dataset map-based plots
# ============================================================
def plot_regime_maps(coords, labels_dict, Y, dataset_name, us_states, output_dir):
    """
    Main geographic visualization: algorithm regime maps + Y distribution.
    No "True Regions" - real-world data has no ground truth.
    """
    geo = is_geographic(coords)
    n_algos = len(labels_dict)

    fig, axes = plt.subplots(2, max(n_algos, 2), figsize=(7 * max(n_algos, 2), 12))
    if n_algos < 2:
        axes = axes.reshape(2, -1)

    # Row 1: Algorithm regime maps
    for i, (algo_name, labels) in enumerate(labels_dict.items()):
        ax = axes[0, i]
        add_basemap(ax, us_states, geo)
        n_reg = len(np.unique(labels))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10',
                        s=8, alpha=0.85, edgecolors='white', linewidths=0.1)
        ax.set_title(f'{algo_name} ({n_reg} regions)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude' if geo else 'X')
        ax.set_ylabel('Latitude' if geo else 'Y')
        if geo:
            ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
            ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    # Hide unused subplot in row 1 if n_algos < ncols
    for j in range(n_algos, axes.shape[1]):
        axes[0, j].set_visible(False)

    # Row 2: Y distribution, Region-mean Y map, Y std map
    # (a) Y distribution on map
    ax = axes[1, 0]
    add_basemap(ax, us_states, geo)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=Y, cmap='viridis',
                    s=8, alpha=0.85, edgecolors='none')
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Y value')
    ax.set_title('Target Variable (Y)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude' if geo else 'X')
    ax.set_ylabel('Latitude' if geo else 'Y')
    if geo:
        ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
        ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    # (b) GeoEvolve region-mean Y map
    best_algo = list(labels_dict.keys())[-1]
    labels = labels_dict[best_algo]
    region_y_mean = {}
    for r in np.unique(labels):
        region_y_mean[r] = np.mean(Y[labels == r])
    y_mean_map = np.array([region_y_mean[labels[i]] for i in range(len(labels))])

    ax = axes[1, 1]
    add_basemap(ax, us_states, geo)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y_mean_map, cmap='viridis',
                    s=8, alpha=0.85, edgecolors='none')
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Region Mean Y')
    ax.set_title(f'{best_algo}: Region-Mean Y', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude' if geo else 'X')
    ax.set_ylabel('Latitude' if geo else 'Y')
    if geo:
        ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
        ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    for j in range(2, axes.shape[1]):
        axes[1, j].set_visible(False)

    fig.suptitle(f'Spatial Regime Maps: {dataset_name}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_regime_maps')


def plot_detail_maps(coords, labels, Y, X, algo_name, dataset_name, x_cols, us_states, output_dir):
    """
    Detailed map for one algorithm:
    1. Regimes with convex hull boundaries
    2. Spatial residuals
    3. Local R² by region
    4. Dominant covariate
    """
    geo = is_geographic(coords)
    unique_labels = np.unique(labels)
    n_regions = len(unique_labels)
    colors = get_distinct_colors(n_regions)
    region_stats = compute_region_regression(X, Y, labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # 1. Regime map with convex hulls
    ax = axes[0, 0]
    add_basemap(ax, us_states, geo)
    for i, r in enumerate(unique_labels):
        mask = labels == r
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], s=6, alpha=0.7,
                   label=f'R{r} (n={mask.sum()})')
        if mask.sum() > 3:
            try:
                hull = ConvexHull(coords[mask])
                pts = coords[mask][hull.vertices]
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], color=colors[i], linewidth=1.5, alpha=0.6)
            except Exception:
                pass
    ax.set_title('Regimes with Convex Hull Boundaries', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2)

    # 2. Spatial residuals
    ax = axes[0, 1]
    add_basemap(ax, us_states, geo)
    full_resid = np.zeros(len(Y))
    for r in unique_labels:
        rs = region_stats[r]
        if rs['residuals'] is not None:
            full_resid[labels == r] = rs['residuals']
    vmax = np.percentile(np.abs(full_resid), 95)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=full_resid, cmap='RdBu_r',
                    s=6, alpha=0.85, edgecolors='none', vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Residual')
    ax.set_title('Spatial Residuals', fontsize=11, fontweight='bold')

    # 3. Local R² map
    ax = axes[1, 0]
    add_basemap(ax, us_states, geo)
    r2_map = np.zeros(len(Y))
    for r in unique_labels:
        rs = region_stats[r]
        r2_map[labels == r] = rs['r2'] if not np.isnan(rs['r2']) else 0
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=r2_map, cmap='RdYlGn',
                    s=6, alpha=0.85, edgecolors='none', vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='R²')
    ax.set_title('Local R² by Region', fontsize=11, fontweight='bold')

    # 4. Dominant covariate
    ax = axes[1, 1]
    add_basemap(ax, us_states, geo)
    dom_feat = np.zeros(len(Y))
    for r in unique_labels:
        rs = region_stats[r]
        if rs['beta'] is not None and len(rs['beta']) > 1:
            dom_feat[labels == r] = np.argmax(np.abs(rs['beta'][1:]))
    n_f = X.shape[1] - 1
    feat_names = x_cols[:n_f] if len(x_cols) >= n_f else [f'X{i+1}' for i in range(n_f)]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=dom_feat, cmap='Set1',
                    s=6, alpha=0.85, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Dominant Feature Index')
    ax.set_title('Dominant Covariate by Region', fontsize=11, fontweight='bold')

    for ax in axes.flat:
        ax.set_xlabel('Longitude' if geo else 'X')
        ax.set_ylabel('Latitude' if geo else 'Y')
        if geo:
            ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
            ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    fig.suptitle(f'Detailed Map: {dataset_name} ({algo_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{algo_name}_detail_maps')


# ============================================================
# Visualization: Intra/Inter-region analysis
# ============================================================
def plot_intra_region(intra, algo_name, dataset_name, output_dir):
    regions = sorted(intra.keys())
    n = len(regions)
    colors = get_distinct_colors(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sizes = [intra[r]['n'] for r in regions]
    axes[0, 0].bar(range(n), sizes, color=colors)
    axes[0, 0].set_title('Region Size')
    axes[0, 0].set_ylabel('Sample Count')

    y_means = [intra[r]['y_mean'] for r in regions]
    y_stds = [intra[r]['y_std'] for r in regions]
    axes[0, 1].bar(range(n), y_means, yerr=y_stds, color=colors, capsize=3)
    axes[0, 1].set_title('Y Distribution by Region')
    axes[0, 1].set_ylabel('Y (mean ± std)')

    r2s = [intra[r]['r2'] if not np.isnan(intra[r]['r2']) else 0 for r in regions]
    bar_c = ['#2ca02c' if v > 0.5 else '#ff7f0e' if v > 0.2 else '#d62728' for v in r2s]
    axes[1, 0].bar(range(n), r2s, color=bar_c)
    axes[1, 0].set_title('R² by Region')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--')

    rmses = [intra[r]['rmse'] if not np.isnan(intra[r]['rmse']) else 0 for r in regions]
    axes[1, 1].bar(range(n), rmses, color=colors)
    axes[1, 1].set_title('RMSE by Region')

    for ax in axes.flat:
        ax.set_xlabel('Region')
        ax.set_xticks(range(n))

    fig.suptitle(f'Intra-Region Analysis: {dataset_name} ({algo_name})', fontsize=14)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{algo_name}_intra_region')


def plot_inter_region_coeff(inter, algo_name, dataset_name, x_cols, output_dir):
    betas = inter['betas']
    if len(betas) == 0:
        return
    n_f = betas.shape[1]
    feat_names = ['Intercept'] + x_cols[:n_f - 1]
    if len(feat_names) < n_f:
        feat_names += [f'X{i}' for i in range(len(feat_names), n_f)]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(inter['valid_labels']) * 0.5 + 2)))

    im = axes[0].imshow(betas, aspect='auto', cmap='RdBu_r')
    axes[0].set_yticks(range(len(inter['valid_labels'])))
    axes[0].set_yticklabels([f'R{r}' for r in inter['valid_labels']])
    axes[0].set_xticks(range(n_f))
    axes[0].set_xticklabels(feat_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_title('Coefficients by Region')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    axes[1].barh(range(n_f), inter['coeff_std_per_feature'], color='steelblue')
    axes[1].set_yticks(range(n_f))
    axes[1].set_yticklabels(feat_names, fontsize=8)
    axes[1].set_xlabel('Std of β Across Regions')
    axes[1].set_title('Coefficient Heterogeneity')

    fig.suptitle(f'Inter-Region Coefficients: {dataset_name} ({algo_name})', fontsize=14)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{algo_name}_inter_region_coeff')


def plot_residuals(X, Y, labels, algo_name, dataset_name, coords, us_states, output_dir):
    geo = is_geographic(coords)
    region_stats = compute_region_regression(X, Y, labels)
    unique_labels = sorted(region_stats.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Residual boxplot
    resid_data, resid_labels = [], []
    for r in unique_labels:
        if region_stats[r]['residuals'] is not None:
            resid_data.append(region_stats[r]['residuals'])
            resid_labels.append(r)
    if resid_data:
        axes[0].boxplot(resid_data, labels=[f'R{r}' for r in resid_labels])
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Residual Distribution')
    axes[0].set_xlabel('Region')

    # Spatial |residuals|
    full_resid = np.zeros(len(Y))
    for r in unique_labels:
        rs = region_stats[r]
        if rs['residuals'] is not None:
            full_resid[labels == r] = rs['residuals']
    add_basemap(axes[1], us_states, geo)
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=np.abs(full_resid),
                          cmap='YlOrRd', s=4, alpha=0.8)
    plt.colorbar(sc, ax=axes[1], shrink=0.8)
    axes[1].set_title('|Residuals| on Map')

    # Global vs Regime residual histogram
    global_reg = compute_global_regression(X, Y)
    axes[2].hist(global_reg['residuals'], bins=50, alpha=0.5, label='Global OLS', density=True)
    axes[2].hist(full_resid, bins=50, alpha=0.5, label=f'{algo_name} Regime', density=True)
    axes[2].legend()
    axes[2].set_title('Residual: Global vs Regime')
    axes[2].set_xlabel('Residual')

    fig.suptitle(f'Residual Analysis: {dataset_name} ({algo_name})', fontsize=14)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{algo_name}_residuals')


# ============================================================
# Visualization: Per-dataset 3-algorithm comparison
# ============================================================
def plot_three_algo_comparison(coords, Y, X, results_dict, dataset_name, x_cols, us_states, output_dir):
    """
    Single figure comparing all three algorithms for one dataset:
    Row 1: regime maps side by side
    Row 2: R² per region, residual histogram, coefficient heatmap comparison
    Row 3: SSR reduction & runtime bar, variance ratio, per-region RMSE
    """
    algos = list(results_dict.keys())
    n_algo = len(algos)
    if n_algo == 0:
        return

    geo = is_geographic(coords)

    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(3, n_algo, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Regime maps
    for j, algo in enumerate(algos):
        ax = fig.add_subplot(gs[0, j])
        add_basemap(ax, us_states, geo)
        labels = results_dict[algo]['labels']
        n_reg = len(np.unique(labels))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10',
                   s=8, alpha=0.85, edgecolors='white', linewidths=0.1)
        ssr_r = results_dict[algo]['summary']['ssr_reduction']
        ax.set_title(f'{algo}\n({n_reg} regions, SSR_red={ssr_r:.3f})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude' if geo else 'X')
        ax.set_ylabel('Latitude' if geo else 'Y')
        if geo:
            ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
            ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    # Row 2: R² per region bar chart (all algos overlaid)
    ax_r2 = fig.add_subplot(gs[1, 0])
    for j, algo in enumerate(algos):
        intra = results_dict[algo]['intra']
        regions = sorted(intra.keys())
        r2s = [intra[r]['r2'] if not np.isnan(intra[r]['r2']) else 0 for r in regions]
        x_pos = np.arange(len(regions))
        ax_r2.bar(x_pos + j * 0.25, r2s, 0.25, label=algo, color=ALGO_COLORS[algo], alpha=0.8)
    ax_r2.set_xlabel('Region')
    ax_r2.set_ylabel('R²')
    ax_r2.set_title('R² by Region (all algorithms)', fontweight='bold')
    ax_r2.legend(fontsize=8)

    # Row 2: Residual histogram comparison
    ax_resid = fig.add_subplot(gs[1, 1])
    global_reg = compute_global_regression(X, Y)
    ax_resid.hist(global_reg['residuals'], bins=50, alpha=0.3, label='Global OLS',
                  density=True, color='gray')
    for algo in algos:
        labels = results_dict[algo]['labels']
        region_stats = compute_region_regression(X, Y, labels)
        full_resid = np.zeros(len(Y))
        for r in np.unique(labels):
            rs = region_stats[r]
            if rs['residuals'] is not None:
                full_resid[labels == r] = rs['residuals']
        ax_resid.hist(full_resid, bins=50, alpha=0.4, label=algo, density=True,
                      color=ALGO_COLORS[algo])
    ax_resid.set_xlabel('Residual')
    ax_resid.set_ylabel('Density')
    ax_resid.set_title('Residual Distributions', fontweight='bold')
    ax_resid.legend(fontsize=8)

    # Row 2: Performance metrics bar chart
    if n_algo > 2:
        ax_perf = fig.add_subplot(gs[1, 2])
    else:
        ax_perf = fig.add_subplot(gs[1, min(1, n_algo - 1)])

    metrics = ['SSR\nReduction', 'Avg R²', 'Var. Ratio']
    x_m = np.arange(len(metrics))
    for j, algo in enumerate(algos):
        s = results_dict[algo]['summary']
        intra = results_dict[algo]['intra']
        r2_vals = [intra[r]['r2'] for r in intra if not np.isnan(intra[r]['r2'])]
        avg_r2 = np.mean(r2_vals) if r2_vals else 0
        total_n = sum(intra[r]['n'] for r in intra)
        within = sum(intra[r]['n'] * intra[r]['y_std']**2 for r in intra
                     if not np.isnan(intra[r]['y_std'])) / max(total_n, 1)
        means = [intra[r]['y_mean'] for r in intra if not np.isnan(intra[r]['y_mean'])]
        between = np.var(means) if len(means) > 1 else 0
        vr = between / (within + between) if (within + between) > 0 else 0
        vals = [s['ssr_reduction'], avg_r2, vr]
        ax_perf.bar(x_m + j * 0.25, vals, 0.25, label=algo, color=ALGO_COLORS[algo])
    ax_perf.set_xticks(x_m + 0.25)
    ax_perf.set_xticklabels(metrics)
    ax_perf.set_title('Performance Metrics', fontweight='bold')
    ax_perf.legend(fontsize=8)
    ax_perf.set_ylim(0, 1)

    # Row 3: RMSE per region comparison
    ax_rmse = fig.add_subplot(gs[2, 0])
    for j, algo in enumerate(algos):
        intra = results_dict[algo]['intra']
        regions = sorted(intra.keys())
        rmses = [intra[r]['rmse'] if not np.isnan(intra[r]['rmse']) else 0 for r in regions]
        x_pos = np.arange(len(regions))
        ax_rmse.bar(x_pos + j * 0.25, rmses, 0.25, label=algo, color=ALGO_COLORS[algo], alpha=0.8)
    ax_rmse.set_xlabel('Region')
    ax_rmse.set_ylabel('RMSE')
    ax_rmse.set_title('RMSE by Region', fontweight='bold')
    ax_rmse.legend(fontsize=8)

    # Row 3: Runtime comparison
    ax_rt = fig.add_subplot(gs[2, 1])
    runtimes = [results_dict[algo]['runtime'] for algo in algos]
    bars = ax_rt.bar(algos, runtimes, color=[ALGO_COLORS[a] for a in algos])
    for bar, rt in zip(bars, runtimes):
        ax_rt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{rt:.1f}s', ha='center', va='bottom', fontsize=10)
    ax_rt.set_ylabel('Runtime (seconds)')
    ax_rt.set_title('Runtime Comparison', fontweight='bold')

    # Row 3: Y-mean per region comparison
    if n_algo > 2:
        ax_ymean = fig.add_subplot(gs[2, 2])
    else:
        ax_ymean = fig.add_subplot(gs[2, min(1, n_algo - 1)])
    for j, algo in enumerate(algos):
        intra = results_dict[algo]['intra']
        regions = sorted(intra.keys())
        y_means = [intra[r]['y_mean'] for r in regions]
        y_stds = [intra[r]['y_std'] for r in regions]
        x_pos = np.arange(len(regions))
        ax_ymean.errorbar(x_pos + j * 0.1, y_means, yerr=y_stds, fmt='o-',
                          label=algo, color=ALGO_COLORS[algo], capsize=2, markersize=4)
    ax_ymean.set_xlabel('Region')
    ax_ymean.set_ylabel('Y mean ± std')
    ax_ymean.set_title('Y Distribution by Region', fontweight='bold')
    ax_ymean.legend(fontsize=8)

    fig.suptitle(f'Three-Algorithm Comparison: {dataset_name}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_3algo_comparison')


# ============================================================
# Visualization: Cross-dataset summaries
# ============================================================
def plot_algorithm_performance_comparison(all_results, output_dir):
    """
    Comprehensive algorithm performance comparison across all datasets.
    """
    datasets = list(all_results.keys())
    algos = list(ALGO_FILES.keys())
    n_ds, n_algo = len(datasets), len(algos)

    # Collect metrics
    ssr_red = np.full((n_ds, n_algo), np.nan)
    runtime = np.full((n_ds, n_algo), np.nan)
    avg_r2 = np.full((n_ds, n_algo), np.nan)
    variance_ratio = np.full((n_ds, n_algo), np.nan)

    for i, ds in enumerate(datasets):
        for j, algo in enumerate(algos):
            if algo not in all_results[ds]:
                continue
            r = all_results[ds][algo]
            ssr_red[i, j] = r['summary']['ssr_reduction']
            runtime[i, j] = r['runtime']
            intra = r['intra']
            # Average R2 across regions
            r2_vals = [intra[reg]['r2'] for reg in intra if not np.isnan(intra[reg]['r2'])]
            avg_r2[i, j] = np.mean(r2_vals) if r2_vals else np.nan
            # Between/Within variance ratio
            total_n = sum(intra[reg]['n'] for reg in intra)
            within = sum(intra[reg]['n'] * intra[reg]['y_std']**2 for reg in intra
                         if not np.isnan(intra[reg]['y_std'])) / max(total_n, 1)
            means = [intra[reg]['y_mean'] for reg in intra if not np.isnan(intra[reg]['y_mean'])]
            between = np.var(means) if len(means) > 1 else 0
            variance_ratio[i, j] = between / (within + between) if (within + between) > 0 else 0

    short_names = [ds.replace('US_', '').replace('_', '\n') for ds in datasets]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    x = np.arange(n_ds)
    w = 0.25

    # 1. SSR Reduction
    for j, algo in enumerate(algos):
        axes[0, 0].bar(x + j * w, ssr_red[:, j], w, label=algo, color=ALGO_COLORS[algo])
    axes[0, 0].set_xticks(x + w)
    axes[0, 0].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[0, 0].set_ylabel('SSR Reduction Ratio')
    axes[0, 0].set_title('SSR Reduction (1 - SSR_regime / SSR_global)', fontweight='bold')
    axes[0, 0].legend()

    # 2. Runtime
    for j, algo in enumerate(algos):
        axes[0, 1].bar(x + j * w, runtime[:, j], w, label=algo, color=ALGO_COLORS[algo])
    axes[0, 1].set_xticks(x + w)
    axes[0, 1].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Runtime (seconds)')
    axes[0, 1].set_title('Runtime Comparison', fontweight='bold')
    axes[0, 1].legend()

    # 3. Average R² across regions
    for j, algo in enumerate(algos):
        axes[1, 0].bar(x + j * w, avg_r2[:, j], w, label=algo, color=ALGO_COLORS[algo])
    axes[1, 0].set_xticks(x + w)
    axes[1, 0].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Avg R² per Region')
    axes[1, 0].set_title('Average Within-Region R²', fontweight='bold')
    axes[1, 0].legend()

    # 4. Variance ratio
    for j, algo in enumerate(algos):
        axes[1, 1].bar(x + j * w, variance_ratio[:, j], w, label=algo, color=ALGO_COLORS[algo])
    axes[1, 1].set_xticks(x + w)
    axes[1, 1].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Between / Total Variance')
    axes[1, 1].set_title('Region Separation Quality', fontweight='bold')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    fig.suptitle('Algorithm Performance Comparison Across All Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, 'algorithm_performance_comparison')


def plot_summary_heatmap(all_results, output_dir):
    datasets = list(all_results.keys())
    algos = list(ALGO_FILES.keys())
    n_ds, n_algo = len(datasets), len(algos)

    ssr_red = np.full((n_ds, n_algo), np.nan)
    runtime = np.full((n_ds, n_algo), np.nan)
    n_reg = np.full((n_ds, n_algo), np.nan)

    for i, ds in enumerate(datasets):
        for j, algo in enumerate(algos):
            if algo in all_results[ds]:
                ssr_red[i, j] = all_results[ds][algo]['summary']['ssr_reduction']
                runtime[i, j] = all_results[ds][algo]['runtime']
                n_reg[i, j] = all_results[ds][algo]['summary']['n_regions']

    short_names = [ds.replace('US_', '') for ds in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(22, max(8, n_ds * 0.5 + 2)))

    im1 = axes[0].imshow(ssr_red, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_yticks(range(n_ds)); axes[0].set_yticklabels(short_names, fontsize=8)
    axes[0].set_xticks(range(n_algo)); axes[0].set_xticklabels(algos, fontsize=10)
    axes[0].set_title('SSR Reduction', fontsize=12, fontweight='bold')
    for i in range(n_ds):
        for j in range(n_algo):
            if not np.isnan(ssr_red[i, j]):
                axes[0].text(j, i, f'{ssr_red[i, j]:.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(n_reg, aspect='auto', cmap='Blues')
    axes[1].set_yticks(range(n_ds)); axes[1].set_yticklabels(short_names, fontsize=8)
    axes[1].set_xticks(range(n_algo)); axes[1].set_xticklabels(algos, fontsize=10)
    axes[1].set_title('# Regions', fontsize=12, fontweight='bold')
    for i in range(n_ds):
        for j in range(n_algo):
            if not np.isnan(n_reg[i, j]):
                axes[1].text(j, i, f'{int(n_reg[i, j])}', ha='center', va='center', fontsize=8)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    rt_log = np.log10(runtime + 0.01)
    im3 = axes[2].imshow(rt_log, aspect='auto', cmap='YlOrRd')
    axes[2].set_yticks(range(n_ds)); axes[2].set_yticklabels(short_names, fontsize=8)
    axes[2].set_xticks(range(n_algo)); axes[2].set_xticklabels(algos, fontsize=10)
    axes[2].set_title('Runtime (log₁₀ s)', fontsize=12, fontweight='bold')
    for i in range(n_ds):
        for j in range(n_algo):
            if not np.isnan(runtime[i, j]):
                axes[2].text(j, i, f'{runtime[i, j]:.1f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    fig.suptitle('Summary Heatmap Across All Datasets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(output_dir, 'summary_heatmap')


def plot_all_datasets_regime_maps(all_coords, all_labels, all_datasets, us_states, output_dir):
    """Multi-panel figure showing GeoEvolve regimes for all datasets with basemap."""
    n_ds = len(all_datasets)
    if n_ds == 0:
        return
    ncols = 4
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols > 1:
        axes = axes.reshape(1, -1)
    elif nrows == 1 and ncols == 1:
        axes = np.array([[axes]])

    for idx, ds_name in enumerate(all_datasets):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        coords = all_coords[ds_name]
        labels = all_labels[ds_name]
        geo = is_geographic(coords)
        add_basemap(ax, us_states, geo)
        n_reg = len(np.unique(labels))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=3, alpha=0.8)
        short = ds_name.replace('US_', '').replace('_', ' ')
        ax.set_title(f'{short}\n({n_reg} regions)', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=6)
        if geo:
            ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
            ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('GeoEvolve Spatial Regimes Across All Datasets', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(output_dir, 'all_datasets_regime_maps')


def plot_coefficient_heterogeneity(all_results, x_cols_dict, output_dir):
    target_algo = 'GeoEvolve'
    datasets = [ds for ds in all_results if target_algo in all_results[ds]]
    if not datasets:
        return
    n_ds = min(len(datasets), 7)
    fig, axes = plt.subplots(n_ds, 1, figsize=(12, 3 * n_ds), squeeze=False)
    for idx, ds in enumerate(datasets[:n_ds]):
        inter = all_results[ds][target_algo]['inter']
        coeff_std = inter['coeff_std_per_feature']
        x_cols = x_cols_dict.get(ds, [])
        n_f = len(coeff_std)
        feat_names = ['Intercept'] + x_cols[:n_f - 1]
        if len(feat_names) < n_f:
            feat_names += [f'X{i}' for i in range(len(feat_names), n_f)]
        axes[idx, 0].barh(range(n_f), coeff_std, color='steelblue')
        axes[idx, 0].set_yticks(range(n_f))
        axes[idx, 0].set_yticklabels(feat_names, fontsize=7)
        axes[idx, 0].set_title(ds.replace('US_', ''), fontsize=10)
        axes[idx, 0].set_xlabel('Std of β')
    fig.suptitle(f'Coefficient Heterogeneity ({target_algo})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, 'coefficient_heterogeneity')


# ============================================================
# Main pipeline
# ============================================================
def process_single_dataset(csv_path, algos, us_states, output_dir):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n{'=' * 60}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 60}")

    df, x_cols = load_dataset(csv_path)
    X, Y, coords, regions_orig, df_clean, sampled = prepare_data(df, x_cols)

    if sampled:
        print(f"  Subsampled to {len(Y)} observations")
    print(f"  Data shape: X={X.shape}, Y={Y.shape}")

    print("  Building spatial weights...")
    w = build_spatial_weights(coords)

    p = determine_p(regions_orig)
    print(f"  Target p={p}")

    results = {}
    labels_dict = {}

    for algo_name, algo_module in algos.items():
        print(f"  Running {algo_name}...", end=' ', flush=True)
        try:
            labels, elapsed = run_single_algorithm(algo_module, X, Y, p, w)
            print(f"Done in {elapsed:.1f}s, {len(np.unique(labels))} regions")

            intra, inter, summary, global_reg = compute_intra_inter_metrics(X, Y, labels)
            results[algo_name] = {
                'labels': labels, 'runtime': elapsed,
                'intra': intra, 'inter': inter,
                'summary': summary, 'global_reg': global_reg,
            }
            labels_dict[algo_name] = labels

            plot_intra_region(intra, algo_name, dataset_name, output_dir)
            plot_inter_region_coeff(inter, algo_name, dataset_name, x_cols, output_dir)
            plot_residuals(X, Y, labels, algo_name, dataset_name, coords, us_states, output_dir)

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    if labels_dict:
        plot_regime_maps(coords, labels_dict, Y, dataset_name, us_states, output_dir)
        for algo_name, labels in labels_dict.items():
            plot_detail_maps(coords, labels, Y, X, algo_name, dataset_name, x_cols, us_states, output_dir)
        # 3-algorithm side-by-side comparison
        if len(results) > 1:
            plot_three_algo_comparison(coords, Y, X, results, dataset_name, x_cols, us_states, output_dir)

    return results, x_cols, coords, Y


def main():
    print("=" * 60)
    print("Real-World Spatial Regime Analysis")
    print("=" * 60)

    us_states = load_us_basemap()
    if us_states is not None:
        print(f"Loaded US basemap with {len(us_states)} states")
    else:
        print("WARNING: No basemap loaded")

    algos = load_all_algorithms()
    print(f"Loaded {len(algos)} algorithms: {list(algos.keys())}")

    all_results = {}
    x_cols_dict = {}
    all_coords = {}
    all_best_labels = {}
    all_Y = {}

    csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} datasets")

    for csv_file in csv_files:
        ds_name = os.path.splitext(csv_file)[0]
        if ds_name in SKIP_DATASETS:
            print(f"\nSkipping {ds_name} (in skip list)")
            continue
        csv_path = os.path.join(DATASET_DIR, csv_file)
        try:
            results, x_cols, coords, Y_data = process_single_dataset(csv_path, algos, us_states, OUTPUT_DIR)
            all_results[ds_name] = results
            x_cols_dict[ds_name] = x_cols
            all_coords[ds_name] = coords
            all_Y[ds_name] = Y_data
            if 'GeoEvolve' in results:
                all_best_labels[ds_name] = results['GeoEvolve']['labels']
            elif results:
                all_best_labels[ds_name] = list(results.values())[-1]['labels']
        except Exception as e:
            print(f"\nERROR processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save raw results
    to_save = {}
    for ds, dr in all_results.items():
        to_save[ds] = {}
        for algo, ar in dr.items():
            to_save[ds][algo] = {'runtime': ar['runtime'], 'summary': ar['summary']}
    with open(os.path.join(OUTPUT_DIR, 'results_summary.pkl'), 'wb') as f:
        pickle.dump(to_save, f)

    # Cross-dataset visualizations
    print("\nGenerating cross-dataset visualizations...")
    plot_algorithm_performance_comparison(all_results, OUTPUT_DIR)
    plot_summary_heatmap(all_results, OUTPUT_DIR)
    plot_coefficient_heterogeneity(all_results, x_cols_dict, OUTPUT_DIR)

    if all_best_labels:
        plot_all_datasets_regime_maps(
            {ds: all_coords[ds] for ds in all_best_labels},
            all_best_labels,
            list(all_best_labels.keys()),
            us_states, OUTPUT_DIR)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Dataset':<35} {'Algorithm':<12} {'SSR_red':>8} {'#Reg':>5} {'Avg_R2':>7} {'Time(s)':>8}")
    print("-" * 90)
    for ds in all_results:
        for algo in all_results[ds]:
            r = all_results[ds][algo]
            s = r['summary']
            intra = r['intra']
            r2_vals = [intra[reg]['r2'] for reg in intra if not np.isnan(intra[reg]['r2'])]
            avg_r2 = np.mean(r2_vals) if r2_vals else 0
            print(f"{ds:<35} {algo:<12} {s['ssr_reduction']:>8.4f} {s['n_regions']:>5} {avg_r2:>7.4f} {r['runtime']:>8.1f}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
