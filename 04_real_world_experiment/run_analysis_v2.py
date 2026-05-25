"""
Real-world Spatial Regime Analysis v2 - 4 algorithms comparison.

Changes from v1:
- 3 algorithms: Initial, OpenEvolve, GeoEvolve
- EPSG:5070 basemap support for projected data (Politics)
- Actual SSR values alongside SSR reduction ratios
- Coefficient magnitude maps
- P-sensitivity with actual SSR
- Comparison CSV output
- Categorical variable detection and removal
- Polygon-based Voronoi visualization for point data
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
from scipy import stats
from scipy.spatial import ConvexHull, Voronoi
from sklearn.preprocessing import StandardScaler, LabelEncoder
import libpysal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon, MultiPoint

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'all_datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGO_FILES = {
    'Initial': 'initial_program.py',
    'OpenEvolve': 'best_program_openevolve.py',
    'GeoEvolve': 'best_program_geoevolve.py',
}
ALGO_COLORS = {
    'Initial': '#1f77b4',
    'OpenEvolve': '#ff7f0e',
    'GeoEvolve': '#2ca02c',
}
ALGO_MARKERS = {
    'Initial': 'o',
    'OpenEvolve': 's',
    'GeoEvolve': '^',
}

MAX_SAMPLES = 2000
K_NEIGHBORS = 6
SKIP_DATASETS = ['US_Forest_FIA', 'US_Climate_ERA5_CENSUS', 'US_Climate_ERA5_STATE']

# P-sensitivity values
P_VALUES = [3, 4, 5, 6, 7, 8, 10, 12]

# Basemap paths
US_STATES_PATH = os.path.join(BASE_DIR, 'us_states_conus.gpkg')
US_STATES_5070_PATH = os.path.join(BASE_DIR, 'us_states_conus_5070.gpkg')
US_COUNTIES_PATH = os.path.join(BASE_DIR, 'us_counties_conus.gpkg')
US_COUNTIES_5070_PATH = os.path.join(BASE_DIR, 'us_counties_conus_5070.gpkg')


# ============================================================
# Utility
# ============================================================
def arr_ptp(a):
    """Replacement for deprecated np.ptp / ndarray.ptp."""
    return float(np.max(a) - np.min(a))


def set_coord_lims(ax, coords, margin=0.05):
    """Set axis limits with margin around coordinate extent."""
    dx = arr_ptp(coords[:, 0]) * margin
    dy = arr_ptp(coords[:, 1]) * margin
    ax.set_xlim(coords[:, 0].min() - dx, coords[:, 0].max() + dx)
    ax.set_ylim(coords[:, 1].min() - dy, coords[:, 1].max() + dy)


def save_fig(output_dir, name, dpi=150):
    plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.close()


def load_us_basemap():
    """Load US CONUS states in WGS84 (EPSG:4326)."""
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


def load_us_basemap_5070():
    """Load US CONUS states in EPSG:5070 (Albers Equal Area)."""
    if os.path.exists(US_STATES_5070_PATH):
        return gpd.read_file(US_STATES_5070_PATH)
    states_4326 = load_us_basemap()
    if states_4326 is not None:
        try:
            states_5070 = states_4326.to_crs('EPSG:5070')
            states_5070.to_file(US_STATES_5070_PATH, driver='GPKG')
            return states_5070
        except Exception as e:
            print(f"  Warning: Could not reproject basemap: {e}")
    return None


def load_us_counties():
    """Load US CONUS county boundaries in WGS84."""
    if os.path.exists(US_COUNTIES_PATH):
        return gpd.read_file(US_COUNTIES_PATH)
    try:
        counties = gpd.read_file(
            'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip')
        exclude_states = ['02', '15', '60', '66', '69', '72', '78']
        conus = counties[~counties['STATEFP'].isin(exclude_states)]
        conus.to_file(US_COUNTIES_PATH, driver='GPKG')
        return conus
    except Exception as e:
        print(f"  Warning: Could not load US counties: {e}")
        return None


def load_us_counties_5070():
    """Load US CONUS counties in EPSG:5070."""
    if os.path.exists(US_COUNTIES_5070_PATH):
        return gpd.read_file(US_COUNTIES_5070_PATH)
    counties_4326 = load_us_counties()
    if counties_4326 is not None:
        try:
            counties_5070 = counties_4326.to_crs('EPSG:5070')
            counties_5070.to_file(US_COUNTIES_5070_PATH, driver='GPKG')
            return counties_5070
        except Exception as e:
            print(f"  Warning: Could not reproject counties: {e}")
    return None


def add_basemap(ax, basemap_gdf):
    """Add basemap boundary overlay to axis. Works with any CRS."""
    if basemap_gdf is not None:
        basemap_gdf.boundary.plot(ax=ax, color='#666666', linewidth=0.4, alpha=0.5)
        basemap_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#999999',
                        linewidth=0.3, alpha=0.2)


def is_latlon(coords):
    """Check if coords are geographic lat/lon (CONUS range)."""
    return (coords[:, 0].min() > -130 and coords[:, 0].max() < -60 and
            coords[:, 1].min() > 23 and coords[:, 1].max() < 52)


def is_projected(coords):
    """Check if coords look like EPSG:5070 projected coordinates."""
    return (coords[:, 0].min() > -3e6 and coords[:, 0].max() < 3e6 and
            coords[:, 1].min() > 0 and coords[:, 1].max() < 4e6)


def get_basemap_for_coords(coords, us_states_4326, us_states_5070):
    """Return appropriate basemap GeoDataFrame based on coordinate system."""
    if is_latlon(coords):
        return us_states_4326
    elif is_projected(coords):
        return us_states_5070
    return None


def get_distinct_colors(n):
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    elif n <= 20:
        return [plt.cm.tab20(i) for i in range(n)]
    else:
        return [plt.cm.gist_ncar(i / max(n - 1, 1)) for i in range(n)]


def algo_short(name):
    """Short version of algorithm name for file names."""
    return name.replace('\n', '_').replace('(', '').replace(')', '').replace(' ', '')


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
        if os.path.exists(fpath):
            algos[name] = load_algorithm(f'algo_{algo_short(name)}', fpath)
        else:
            print(f"  WARNING: Algorithm file not found: {fname}")
    return algos


# ============================================================
# Data loading
# ============================================================
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # Detect coordinate columns
    is_proj = False
    if 'proj_x' in df.columns:
        df = df.rename(columns={'proj_x': 'coord_x', 'proj_y': 'coord_y'})
        is_proj = True
    elif 'proj_X' in df.columns:
        df = df.rename(columns={'proj_X': 'coord_x', 'proj_Y': 'coord_y'})
        is_proj = True
    elif 'lat' in df.columns and 'lon' in df.columns:
        df['coord_x'] = df['lon']
        df['coord_y'] = df['lat']
    else:
        raise ValueError(f"Cannot find coordinate columns in {csv_path}")

    x_cols = sorted([c for c in df.columns if c.startswith('X') and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
    return df, x_cols, is_proj


def identify_continuous_cols(df, x_cols, min_unique=15):
    """Identify and keep only continuous X columns.
    Remove string/object columns and low-cardinality numeric columns."""
    continuous = []
    removed = []
    for col in x_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'str':
            removed.append((col, 'string/categorical'))
            continue
        nunique = df[col].nunique()
        if nunique < min_unique:
            removed.append((col, f'low cardinality ({nunique} unique)'))
            continue
        continuous.append(col)
    return continuous, removed


def prepare_data(df, x_cols, is_proj=False, max_samples=MAX_SAMPLES):
    # Identify continuous variables only
    continuous_cols, removed_cols = identify_continuous_cols(df, x_cols)
    if removed_cols:
        print(f"  Removed non-continuous variables: {removed_cols}")
    if not continuous_cols:
        raise ValueError("No continuous X columns found")
    x_cols = continuous_cols

    key_cols = ['Y'] + x_cols + ['coord_x', 'coord_y']
    df_clean = df.dropna(subset=key_cols).copy()

    # Filter Alaska/Hawaii for geographic (non-projected) coordinates
    if not is_proj and 'coord_x' in df_clean.columns:
        cx, cy = df_clean['coord_x'], df_clean['coord_y']
        looks_like_geo = (cx.min() > -200 and cx.max() < 0 and
                          cy.min() > 0 and cy.max() < 80)
        if looks_like_geo and (cx.min() < -130 or cy.max() > 55):
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
    regions_orig = df_clean['region'].values if 'region' in df_clean.columns else None

    X_raw = df_clean[x_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    return X, Y, coords, regions_orig, df_clean, sampled, x_cols


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
# Metrics
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
    coeff_range = (np.max(betas, axis=0) - np.min(betas, axis=0)) if len(betas) > 1 else np.zeros(X.shape[1])

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


def compute_metrics_quick(X, Y, labels):
    """Quick metric computation for p-sensitivity."""
    unique_labels = np.unique(labels)
    n_regions = len(unique_labels)

    beta_g = np.linalg.pinv(X).dot(Y)
    resid_g = Y - X.dot(beta_g)
    ssr_global = np.sum(resid_g ** 2)

    total_ssr = 0
    r2_list, rmse_list, y_means, region_sizes = [], [], [], []

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

    total_n = sum(region_sizes)
    within_var = sum(s * np.var(Y[labels == r]) for s, r in zip(region_sizes, unique_labels)) / max(total_n, 1)
    between_var = np.var(y_means) if len(y_means) > 1 else 0
    var_ratio = between_var / (within_var + between_var) if (within_var + between_var) > 0 else 0

    return {
        'ssr_reduction': ssr_reduction,
        'ssr_regime': total_ssr,
        'ssr_global': ssr_global,
        'avg_r2': avg_r2,
        'avg_rmse': avg_rmse,
        'var_ratio': var_ratio,
        'n_regions': n_regions,
    }


# ============================================================
# Visualization: Maps with proper basemap
# ============================================================
def plot_regime_maps(coords, labels_dict, Y, dataset_name, basemap, output_dir):
    n_algos = len(labels_dict)
    fig, axes = plt.subplots(2, max(n_algos, 2), figsize=(6 * max(n_algos, 2), 11))
    if n_algos < 2:
        axes = axes.reshape(2, -1)

    for i, (aname, labels) in enumerate(labels_dict.items()):
        ax = axes[0, i]
        add_basemap(ax, basemap)
        n_reg = len(np.unique(labels))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10',
                   s=8, alpha=0.85, edgecolors='white', linewidths=0.1)
        ax.set_title(f'{aname}\n({n_reg} regions)', fontsize=10, fontweight='bold')
        set_coord_lims(ax, coords)

    for j in range(n_algos, axes.shape[1]):
        axes[0, j].set_visible(False)

    # Y distribution
    ax = axes[1, 0]
    add_basemap(ax, basemap)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=Y, cmap='viridis', s=8, alpha=0.85)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Y')
    ax.set_title('Target Variable (Y)', fontsize=10, fontweight='bold')
    set_coord_lims(ax, coords)

    # Region-mean Y (last algo)
    best_algo = list(labels_dict.keys())[-1]
    labels = labels_dict[best_algo]
    region_y_mean = {r: np.mean(Y[labels == r]) for r in np.unique(labels)}
    y_mean_map = np.array([region_y_mean[labels[i]] for i in range(len(labels))])

    ax = axes[1, 1]
    add_basemap(ax, basemap)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y_mean_map, cmap='viridis', s=8, alpha=0.85)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Region Mean Y')
    ax.set_title(f'{best_algo}: Region-Mean Y', fontsize=10, fontweight='bold')
    set_coord_lims(ax, coords)

    for j in range(2, axes.shape[1]):
        axes[1, j].set_visible(False)

    fig.suptitle(f'Spatial Regime Maps: {dataset_name}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_regime_maps')


def plot_coefficient_maps(coords, X, Y, labels, algo_name, dataset_name, x_cols, basemap, output_dir):
    """Map showing regression coefficient magnitudes at each point."""
    region_stats = compute_region_regression(X, Y, labels)
    unique_labels = np.unique(labels)
    n_features = X.shape[1] - 1  # exclude intercept
    feat_names = x_cols[:n_features] if len(x_cols) >= n_features else [f'X{i+1}' for i in range(n_features)]

    # Show top 4 most heterogeneous features
    betas_all = []
    for r in unique_labels:
        if region_stats[r]['beta'] is not None:
            betas_all.append(region_stats[r]['beta'][1:])  # skip intercept
    if not betas_all:
        return
    betas_arr = np.array(betas_all)
    coeff_std = np.std(betas_arr, axis=0)
    top_feats = np.argsort(coeff_std)[::-1][:min(4, len(coeff_std))]

    n_plots = len(top_feats)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for idx, fi in enumerate(top_feats):
        ax = axes[idx]
        add_basemap(ax, basemap)
        # Map coefficient value to each point
        coeff_map = np.zeros(len(Y))
        for r in unique_labels:
            rs = region_stats[r]
            if rs['beta'] is not None:
                coeff_map[labels == r] = rs['beta'][fi + 1]  # +1 for intercept
        vmax = np.percentile(np.abs(coeff_map), 95)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=coeff_map, cmap='RdBu_r',
                       s=8, alpha=0.85, vmin=-vmax, vmax=vmax)
        plt.colorbar(sc, ax=ax, shrink=0.7, label=f'beta({feat_names[fi]})')
        ax.set_title(f'{feat_names[fi]}\n(std={coeff_std[fi]:.3f})', fontsize=10, fontweight='bold')
        set_coord_lims(ax, coords)

    short_algo = algo_short(algo_name)
    fig.suptitle(f'Coefficient Maps: {dataset_name} ({algo_name})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{short_algo}_coeff_maps')


def plot_voronoi_regions(coords, labels, algo_name, dataset_name, basemap, output_dir):
    """Voronoi polygon-based visualization of spatial regimes."""
    from scipy.spatial import Voronoi
    unique_labels = np.unique(labels)
    n_reg = len(unique_labels)
    colors = get_distinct_colors(n_reg)

    try:
        # Add bounding points far away to close Voronoi cells
        margin = max(arr_ptp(coords[:, 0]), arr_ptp(coords[:, 1])) * 2
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        far_pts = np.array([
            [cx - margin, cy - margin],
            [cx + margin, cy - margin],
            [cx - margin, cy + margin],
            [cx + margin, cy + margin],
        ])
        all_pts = np.vstack([coords, far_pts])
        vor = Voronoi(all_pts)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        add_basemap(ax, basemap)

        for idx in range(len(coords)):
            region_idx = vor.point_region[idx]
            if region_idx == -1:
                continue
            verts_idx = vor.regions[region_idx]
            if -1 in verts_idx or len(verts_idx) == 0:
                continue
            polygon = vor.vertices[verts_idx]
            label = labels[idx]
            li = list(unique_labels).index(label)
            ax.fill(polygon[:, 0], polygon[:, 1], color=colors[li], alpha=0.5,
                    edgecolor='white', linewidth=0.2)

        ax.scatter(coords[:, 0], coords[:, 1], c='black', s=1, alpha=0.3)
        set_coord_lims(ax, coords)

        patches = [Patch(color=colors[i], alpha=0.5, label=f'Region {r}')
                   for i, r in enumerate(unique_labels)]
        ax.legend(handles=patches, fontsize=7, loc='lower right', ncol=2)
        ax.set_title(f'Voronoi Regions: {dataset_name} ({algo_name}, {n_reg} regions)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        short_algo = algo_short(algo_name)
        save_fig(output_dir, f'{dataset_name}_{short_algo}_voronoi')
    except Exception as e:
        print(f"    Voronoi plot failed: {e}")


def plot_detail_maps(coords, labels, Y, X, algo_name, dataset_name, x_cols, basemap, output_dir):
    unique_labels = np.unique(labels)
    n_regions = len(unique_labels)
    colors = get_distinct_colors(n_regions)
    region_stats = compute_region_regression(X, Y, labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # 1. Regime map with convex hulls
    ax = axes[0, 0]
    add_basemap(ax, basemap)
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
    ax.set_title('Regimes with Convex Hull', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2)

    # 2. Spatial residuals
    ax = axes[0, 1]
    add_basemap(ax, basemap)
    full_resid = np.zeros(len(Y))
    for r in unique_labels:
        rs = region_stats[r]
        if rs['residuals'] is not None:
            full_resid[labels == r] = rs['residuals']
    vmax = np.percentile(np.abs(full_resid), 95)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=full_resid, cmap='RdBu_r',
                    s=6, alpha=0.85, vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Residual')
    ax.set_title('Spatial Residuals', fontsize=11, fontweight='bold')

    # 3. Local R^2
    ax = axes[1, 0]
    add_basemap(ax, basemap)
    r2_map = np.zeros(len(Y))
    for r in unique_labels:
        rs = region_stats[r]
        r2_map[labels == r] = rs['r2'] if not np.isnan(rs['r2']) else 0
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=r2_map, cmap='RdYlGn',
                    s=6, alpha=0.85, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='R^2')
    ax.set_title('Local R^2 by Region', fontsize=11, fontweight='bold')

    # 4. Dominant covariate
    ax = axes[1, 1]
    add_basemap(ax, basemap)
    dom_feat = np.zeros(len(Y))
    for r in unique_labels:
        rs = region_stats[r]
        if rs['beta'] is not None and len(rs['beta']) > 1:
            dom_feat[labels == r] = np.argmax(np.abs(rs['beta'][1:]))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=dom_feat, cmap='Set1', s=6, alpha=0.85)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Dominant Feature Idx')
    ax.set_title('Dominant Covariate by Region', fontsize=11, fontweight='bold')

    for ax in axes.flat:
        set_coord_lims(ax, coords)

    short_algo = algo_short(algo_name)
    fig.suptitle(f'Detail Maps: {dataset_name} ({algo_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{short_algo}_detail_maps')


# ============================================================
# Intra/Inter-region plots
# ============================================================
def plot_intra_region(intra, algo_name, dataset_name, output_dir):
    regions = sorted(intra.keys())
    n = len(regions)
    colors = get_distinct_colors(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sizes = [intra[r]['n'] for r in regions]
    axes[0, 0].bar(range(n), sizes, color=colors)
    axes[0, 0].set_title('Region Size')
    axes[0, 0].set_ylabel('Count')

    y_means = [intra[r]['y_mean'] for r in regions]
    y_stds = [intra[r]['y_std'] for r in regions]
    axes[0, 1].bar(range(n), y_means, yerr=y_stds, color=colors, capsize=3)
    axes[0, 1].set_title('Y by Region')
    axes[0, 1].set_ylabel('Y (mean +- std)')

    r2s = [intra[r]['r2'] if not np.isnan(intra[r]['r2']) else 0 for r in regions]
    bar_c = ['#2ca02c' if v > 0.5 else '#ff7f0e' if v > 0.2 else '#d62728' for v in r2s]
    axes[1, 0].bar(range(n), r2s, color=bar_c)
    axes[1, 0].set_title('R^2 by Region')

    rmses = [intra[r]['rmse'] if not np.isnan(intra[r]['rmse']) else 0 for r in regions]
    axes[1, 1].bar(range(n), rmses, color=colors)
    axes[1, 1].set_title('RMSE by Region')

    for ax in axes.flat:
        ax.set_xlabel('Region')
        ax.set_xticks(range(n))

    short_algo = algo_short(algo_name)
    fig.suptitle(f'Intra-Region: {dataset_name} ({algo_name})', fontsize=14)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{short_algo}_intra_region')


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
    axes[1].set_xlabel('Std of beta')
    axes[1].set_title('Coefficient Heterogeneity')

    short_algo = algo_short(algo_name)
    fig.suptitle(f'Inter-Region Coefficients: {dataset_name} ({algo_name})', fontsize=14)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_{short_algo}_inter_region_coeff')


# ============================================================
# Multi-algorithm comparison (per dataset)
# ============================================================
def plot_algo_comparison(coords, Y, X, results_dict, dataset_name, x_cols, basemap, output_dir):
    algos = list(results_dict.keys())
    n_algo = len(algos)
    if n_algo == 0:
        return

    fig = plt.figure(figsize=(6 * n_algo, 18))
    gs = GridSpec(3, n_algo, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Regime maps
    for j, algo in enumerate(algos):
        ax = fig.add_subplot(gs[0, j])
        add_basemap(ax, basemap)
        labels = results_dict[algo]['labels']
        n_reg = len(np.unique(labels))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10',
                   s=8, alpha=0.85, edgecolors='white', linewidths=0.1)
        ssr_r = results_dict[algo]['summary']['ssr_reduction']
        ax.set_title(f'{algo}\n({n_reg} reg, SSR_red={ssr_r:.3f})', fontsize=9, fontweight='bold')
        set_coord_lims(ax, coords)

    # Row 2: Residual histogram + Performance bar
    ax_resid = fig.add_subplot(gs[1, :n_algo // 2 + 1])
    global_reg = compute_global_regression(X, Y)
    ax_resid.hist(global_reg['residuals'], bins=50, alpha=0.3, label='Global OLS',
                  density=True, color='gray')
    for algo in algos:
        labels = results_dict[algo]['labels']
        rs = compute_region_regression(X, Y, labels)
        full_resid = np.zeros(len(Y))
        for r in np.unique(labels):
            if rs[r]['residuals'] is not None:
                full_resid[labels == r] = rs[r]['residuals']
        ax_resid.hist(full_resid, bins=50, alpha=0.3, label=algo, density=True,
                      color=ALGO_COLORS[algo])
    ax_resid.set_xlabel('Residual')
    ax_resid.set_title('Residual Distributions', fontweight='bold')
    ax_resid.legend(fontsize=7)

    ax_perf = fig.add_subplot(gs[1, n_algo // 2 + 1:])
    metrics = ['SSR\nReduction', 'Avg R^2', 'Var.\nRatio']
    x_m = np.arange(len(metrics))
    bw = 0.8 / n_algo
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
        ax_perf.bar(x_m + j * bw, vals, bw, label=algo, color=ALGO_COLORS[algo])
    ax_perf.set_xticks(x_m + bw * n_algo / 2)
    ax_perf.set_xticklabels(metrics)
    ax_perf.set_title('Performance Metrics', fontweight='bold')
    ax_perf.legend(fontsize=7)
    ax_perf.set_ylim(0, 1)

    # Row 3: Runtime + actual SSR
    ax_rt = fig.add_subplot(gs[2, :n_algo // 2 + 1])
    runtimes = [results_dict[algo]['runtime'] for algo in algos]
    short_algos = [a.replace('\n', ' ') for a in algos]
    bars = ax_rt.bar(short_algos, runtimes, color=[ALGO_COLORS[a] for a in algos])
    for bar, rt in zip(bars, runtimes):
        ax_rt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{rt:.1f}s', ha='center', va='bottom', fontsize=9)
    ax_rt.set_ylabel('Runtime (s)')
    ax_rt.set_title('Runtime', fontweight='bold')
    ax_rt.tick_params(axis='x', rotation=20, labelsize=8)

    ax_ssr = fig.add_subplot(gs[2, n_algo // 2 + 1:])
    global_ssr = results_dict[algos[0]]['summary']['global_ssr']
    ssrs = [results_dict[algo]['summary']['total_ssr'] for algo in algos]
    bars_g = ax_ssr.bar(['Global OLS'], [global_ssr], color='gray', alpha=0.5)
    bars_r = ax_ssr.bar(short_algos, ssrs, color=[ALGO_COLORS[a] for a in algos])
    ax_ssr.set_ylabel('SSR')
    ax_ssr.set_title('Actual SSR (lower is better)', fontweight='bold')
    ax_ssr.tick_params(axis='x', rotation=20, labelsize=8)

    fig.suptitle(f'Algorithm Comparison: {dataset_name}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(output_dir, f'{dataset_name}_algo_comparison')


# ============================================================
# Cross-dataset summaries
# ============================================================
def plot_performance_comparison(all_results, output_dir):
    datasets = list(all_results.keys())
    algos = list(ALGO_FILES.keys())
    # Only include algos that have results
    algos = [a for a in algos if any(a in all_results[ds] for ds in datasets)]
    n_ds, n_algo = len(datasets), len(algos)

    ssr_red = np.full((n_ds, n_algo), np.nan)
    ssr_actual = np.full((n_ds, n_algo), np.nan)
    runtime = np.full((n_ds, n_algo), np.nan)
    avg_r2 = np.full((n_ds, n_algo), np.nan)
    var_ratio = np.full((n_ds, n_algo), np.nan)

    for i, ds in enumerate(datasets):
        for j, algo in enumerate(algos):
            if algo not in all_results[ds]:
                continue
            r = all_results[ds][algo]
            ssr_red[i, j] = r['summary']['ssr_reduction']
            ssr_actual[i, j] = r['summary']['total_ssr']
            runtime[i, j] = r['runtime']
            intra = r['intra']
            r2_vals = [intra[reg]['r2'] for reg in intra if not np.isnan(intra[reg]['r2'])]
            avg_r2[i, j] = np.mean(r2_vals) if r2_vals else np.nan
            total_n = sum(intra[reg]['n'] for reg in intra)
            within = sum(intra[reg]['n'] * intra[reg]['y_std']**2 for reg in intra
                         if not np.isnan(intra[reg]['y_std'])) / max(total_n, 1)
            means = [intra[reg]['y_mean'] for reg in intra if not np.isnan(intra[reg]['y_mean'])]
            between = np.var(means) if len(means) > 1 else 0
            var_ratio[i, j] = between / (within + between) if (within + between) > 0 else 0

    short_names = [ds.replace('US_', '').replace('_', '\n') for ds in datasets]
    x = np.arange(n_ds)
    bw = 0.8 / n_algo

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))

    # SSR Reduction
    for j, algo in enumerate(algos):
        axes[0, 0].bar(x + j * bw, ssr_red[:, j], bw, label=algo.replace('\n', ' '),
                       color=ALGO_COLORS[algo])
    axes[0, 0].set_xticks(x + bw * n_algo / 2)
    axes[0, 0].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[0, 0].set_ylabel('SSR Reduction')
    axes[0, 0].set_title('SSR Reduction (higher is better)', fontweight='bold')
    axes[0, 0].legend(fontsize=7)

    # Actual SSR
    for j, algo in enumerate(algos):
        axes[0, 1].bar(x + j * bw, ssr_actual[:, j], bw, label=algo.replace('\n', ' '),
                       color=ALGO_COLORS[algo])
    axes[0, 1].set_xticks(x + bw * n_algo / 2)
    axes[0, 1].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[0, 1].set_ylabel('SSR')
    axes[0, 1].set_title('Actual SSR (lower is better)', fontweight='bold')
    axes[0, 1].legend(fontsize=7)

    # Runtime
    for j, algo in enumerate(algos):
        axes[1, 0].bar(x + j * bw, runtime[:, j], bw, label=algo.replace('\n', ' '),
                       color=ALGO_COLORS[algo])
    axes[1, 0].set_xticks(x + bw * n_algo / 2)
    axes[1, 0].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Runtime (s)')
    axes[1, 0].set_title('Runtime', fontweight='bold')
    axes[1, 0].legend(fontsize=7)

    # Avg R^2
    for j, algo in enumerate(algos):
        axes[1, 1].bar(x + j * bw, avg_r2[:, j], bw, label=algo.replace('\n', ' '),
                       color=ALGO_COLORS[algo])
    axes[1, 1].set_xticks(x + bw * n_algo / 2)
    axes[1, 1].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Avg R^2')
    axes[1, 1].set_title('Average Within-Region R^2', fontweight='bold')
    axes[1, 1].legend(fontsize=7)

    # Variance ratio
    for j, algo in enumerate(algos):
        axes[2, 0].bar(x + j * bw, var_ratio[:, j], bw, label=algo.replace('\n', ' '),
                       color=ALGO_COLORS[algo])
    axes[2, 0].set_xticks(x + bw * n_algo / 2)
    axes[2, 0].set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    axes[2, 0].set_ylabel('Between / Total Var')
    axes[2, 0].set_title('Region Separation Quality', fontweight='bold')
    axes[2, 0].legend(fontsize=7)
    axes[2, 0].set_ylim(0, 1)

    # Mean metrics radar-like bar
    ax = axes[2, 1]
    metric_names = ['SSR Red.', 'Avg R^2', 'Var Ratio']
    x_m = np.arange(len(metric_names))
    for j, algo in enumerate(algos):
        vals = [np.nanmean(ssr_red[:, j]), np.nanmean(avg_r2[:, j]), np.nanmean(var_ratio[:, j])]
        ax.bar(x_m + j * bw, vals, bw, label=algo.replace('\n', ' '), color=ALGO_COLORS[algo])
    ax.set_xticks(x_m + bw * n_algo / 2)
    ax.set_xticklabels(metric_names)
    ax.set_title('Mean Performance Across All Datasets', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)

    fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, 'algorithm_performance_comparison')


def plot_summary_heatmap(all_results, output_dir):
    datasets = list(all_results.keys())
    algos = [a for a in ALGO_FILES.keys() if any(a in all_results[ds] for ds in datasets)]
    n_ds, n_algo = len(datasets), len(algos)

    ssr_red = np.full((n_ds, n_algo), np.nan)
    ssr_actual = np.full((n_ds, n_algo), np.nan)
    runtime = np.full((n_ds, n_algo), np.nan)

    for i, ds in enumerate(datasets):
        for j, algo in enumerate(algos):
            if algo in all_results[ds]:
                ssr_red[i, j] = all_results[ds][algo]['summary']['ssr_reduction']
                ssr_actual[i, j] = all_results[ds][algo]['summary']['total_ssr']
                runtime[i, j] = all_results[ds][algo]['runtime']

    short_names = [ds.replace('US_', '') for ds in datasets]
    algo_labels = [a.replace('\n', ' ') for a in algos]

    fig, axes = plt.subplots(1, 3, figsize=(24, max(8, n_ds * 0.5 + 2)))

    im1 = axes[0].imshow(ssr_red, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_yticks(range(n_ds)); axes[0].set_yticklabels(short_names, fontsize=8)
    axes[0].set_xticks(range(n_algo)); axes[0].set_xticklabels(algo_labels, fontsize=9)
    axes[0].set_title('SSR Reduction', fontsize=12, fontweight='bold')
    for i in range(n_ds):
        for j in range(n_algo):
            if not np.isnan(ssr_red[i, j]):
                axes[0].text(j, i, f'{ssr_red[i, j]:.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(ssr_actual, aspect='auto', cmap='YlOrRd')
    axes[1].set_yticks(range(n_ds)); axes[1].set_yticklabels(short_names, fontsize=8)
    axes[1].set_xticks(range(n_algo)); axes[1].set_xticklabels(algo_labels, fontsize=9)
    axes[1].set_title('Actual SSR', fontsize=12, fontweight='bold')
    for i in range(n_ds):
        for j in range(n_algo):
            if not np.isnan(ssr_actual[i, j]):
                axes[1].text(j, i, f'{ssr_actual[i, j]:.0f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    rt_log = np.log10(runtime + 0.01)
    im3 = axes[2].imshow(rt_log, aspect='auto', cmap='YlOrRd')
    axes[2].set_yticks(range(n_ds)); axes[2].set_yticklabels(short_names, fontsize=8)
    axes[2].set_xticks(range(n_algo)); axes[2].set_xticklabels(algo_labels, fontsize=9)
    axes[2].set_title('Runtime (log10 s)', fontsize=12, fontweight='bold')
    for i in range(n_ds):
        for j in range(n_algo):
            if not np.isnan(runtime[i, j]):
                axes[2].text(j, i, f'{runtime[i, j]:.1f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    fig.suptitle('Summary Heatmap', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(output_dir, 'summary_heatmap')


def plot_all_datasets_regime_maps(all_coords, all_labels, all_datasets, basemaps, output_dir):
    n_ds = len(all_datasets)
    if n_ds == 0:
        return
    ncols = 4
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, ds_name in enumerate(all_datasets):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        coords = all_coords[ds_name]
        labels = all_labels[ds_name]
        bm = basemaps.get(ds_name)
        add_basemap(ax, bm)
        n_reg = len(np.unique(labels))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=3, alpha=0.8)
        short = ds_name.replace('US_', '').replace('_', ' ')
        ax.set_title(f'{short}\n({n_reg} regions)', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=6)
        set_coord_lims(ax, coords)

    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Best Algorithm Regimes Across All Datasets', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(output_dir, 'all_datasets_regime_maps')


# ============================================================
# P-sensitivity plots
# ============================================================
def plot_p_sensitivity_single(ds_name, p_values, results_by_algo, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(28, 11))
    metrics = [
        ('ssr_reduction', 'SSR Reduction', axes[0, 0]),
        ('ssr_regime', 'SSR (actual)', axes[0, 1]),
        ('avg_r2', 'Average R^2', axes[0, 2]),
        ('avg_rmse', 'Average RMSE', axes[0, 3]),
        ('var_ratio', 'Between/Total Var Ratio', axes[1, 0]),
        ('n_regions', 'Actual # Regions', axes[1, 1]),
        ('runtime', 'Runtime (s)', axes[1, 2]),
    ]
    axes[1, 3].set_visible(False)

    for mkey, mlabel, ax in metrics:
        for algo in results_by_algo:
            ps = [r['p'] for r in results_by_algo[algo]]
            vals = [r[mkey] for r in results_by_algo[algo]]
            ax.plot(ps, vals, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                    label=algo.replace('\n', ' '), linewidth=2, markersize=7, alpha=0.85)
        ax.set_xlabel('p', fontsize=11)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(p_values)

    short_name = ds_name.replace('US_', '').replace('_', ' ')
    fig.suptitle(f'P-Sensitivity: {short_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, f'{ds_name}_p_sensitivity')


def plot_p_sensitivity_all_datasets(all_p_results, p_values, output_dir):
    algos = [a for a in ALGO_FILES.keys()
             if any(a in all_p_results[ds] for ds in all_p_results)]
    metric_keys = ['ssr_reduction', 'ssr_regime', 'avg_r2', 'avg_rmse', 'var_ratio']
    metric_labels = ['SSR Reduction', 'Actual SSR', 'Avg R^2', 'Avg RMSE', 'Var Ratio']

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    axes_flat = axes.flat

    for idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes_flat[idx]
        for algo in algos:
            mean_vals, std_vals = [], []
            for p in p_values:
                vals_at_p = []
                for ds in all_p_results:
                    if algo in all_p_results[ds]:
                        for r in all_p_results[ds][algo]:
                            if r['p'] == p:
                                vals_at_p.append(r[mkey])
                mean_vals.append(np.mean(vals_at_p) if vals_at_p else np.nan)
                std_vals.append(np.std(vals_at_p) if vals_at_p else 0)

            mean_vals = np.array(mean_vals)
            std_vals = np.array(std_vals)
            ax.plot(p_values, mean_vals, marker=ALGO_MARKERS[algo],
                    color=ALGO_COLORS[algo], label=algo.replace('\n', ' '),
                    linewidth=2.5, markersize=8)
            ax.fill_between(p_values, mean_vals - std_vals, mean_vals + std_vals,
                            color=ALGO_COLORS[algo], alpha=0.12)

        ax.set_xlabel('p', fontsize=11)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(p_values)

    axes_flat[5].set_visible(False)
    fig.suptitle('P-Sensitivity: Mean +- Std Across All Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, 'all_datasets_p_sensitivity_mean')


def plot_p_sensitivity_grid(all_p_results, p_values, output_dir, metric_key='ssr_reduction', ylabel='SSR Red.'):
    datasets = list(all_p_results.keys())
    algos = [a for a in ALGO_FILES.keys()
             if any(a in all_p_results[ds] for ds in datasets)]
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
                vals = [r[metric_key] for r in all_p_results[ds][algo]]
                ax.plot(ps, vals, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                        label=algo.replace('\n', ' '), linewidth=1.5, markersize=5)
        short = ds.replace('US_', '').replace('_', ' ')
        ax.set_title(short, fontsize=9, fontweight='bold')
        ax.set_xlabel('p', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(p_values)
        ax.tick_params(labelsize=7)

    for idx in range(n_ds, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    title_map = {'ssr_reduction': 'SSR Reduction', 'ssr_regime': 'Actual SSR',
                 'runtime': 'Runtime', 'avg_r2': 'Avg R^2'}
    fig.suptitle(f'{title_map.get(metric_key, metric_key)} vs p', fontsize=16, fontweight='bold')
    plt.tight_layout()
    suffix = metric_key.replace('ssr_reduction', 'ssr').replace('ssr_regime', 'ssr_actual')
    save_fig(output_dir, f'all_datasets_{suffix}_vs_p_grid')


def plot_p_regime_maps(coords, labels_by_p, p_values_used, algo_name, ds_name, basemap, output_dir):
    n_p = len(p_values_used)
    ncols = min(n_p, 4)
    nrows = (n_p + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    for idx, p in enumerate(p_values_used):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        add_basemap(ax, basemap)
        labels = labels_by_p[p]
        n_reg = len(np.unique(labels))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10',
                   s=6, alpha=0.85, edgecolors='white', linewidths=0.1)
        ax.set_title(f'p={p} ({n_reg} regions)', fontsize=11, fontweight='bold')
        set_coord_lims(ax, coords)
        ax.tick_params(labelsize=7)

    for idx in range(n_p, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    short = ds_name.replace('US_', '').replace('_', ' ')
    short_algo = algo_short(algo_name)
    fig.suptitle(f'{algo_name} Regimes at Different p: {short}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(output_dir, f'{ds_name}_{short_algo}_regime_maps_by_p')


# ============================================================
# CSV export
# ============================================================
def export_comparison_csv(all_results, output_dir):
    rows = []
    for ds in all_results:
        for algo in all_results[ds]:
            r = all_results[ds][algo]
            s = r['summary']
            intra = r['intra']
            r2_vals = [intra[reg]['r2'] for reg in intra if not np.isnan(intra[reg]['r2'])]
            avg_r2 = np.mean(r2_vals) if r2_vals else np.nan
            rmse_vals = [intra[reg]['rmse'] for reg in intra if not np.isnan(intra[reg]['rmse'])]
            avg_rmse = np.mean(rmse_vals) if rmse_vals else np.nan
            total_n = sum(intra[reg]['n'] for reg in intra)
            within = sum(intra[reg]['n'] * intra[reg]['y_std']**2 for reg in intra
                         if not np.isnan(intra[reg]['y_std'])) / max(total_n, 1)
            means = [intra[reg]['y_mean'] for reg in intra if not np.isnan(intra[reg]['y_mean'])]
            between = np.var(means) if len(means) > 1 else 0
            vr = between / (within + between) if (within + between) > 0 else 0
            rows.append({
                'Dataset': ds,
                'Algorithm': algo.replace('\n', ' '),
                'N_Regions': s['n_regions'],
                'SSR_Global': s['global_ssr'],
                'SSR_Regime': s['total_ssr'],
                'SSR_Reduction': s['ssr_reduction'],
                'Avg_R2': avg_r2,
                'Avg_RMSE': avg_rmse,
                'Variance_Ratio': vr,
                'Global_R2': s['global_r2'],
                'Runtime_s': r['runtime'],
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'algorithm_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nComparison CSV saved to: {csv_path}")
    return df


def export_p_sensitivity_csv(all_p_results, output_dir):
    rows = []
    for ds in all_p_results:
        for algo in all_p_results[ds]:
            for r in all_p_results[ds][algo]:
                rows.append({
                    'Dataset': ds,
                    'Algorithm': algo.replace('\n', ' '),
                    'p': r['p'],
                    'N_Regions': r['n_regions'],
                    'SSR_Global': r['ssr_global'],
                    'SSR_Regime': r['ssr_regime'],
                    'SSR_Reduction': r['ssr_reduction'],
                    'Avg_R2': r['avg_r2'],
                    'Avg_RMSE': r['avg_rmse'],
                    'Variance_Ratio': r['var_ratio'],
                    'Runtime_s': r['runtime'],
                })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'p_sensitivity_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"P-sensitivity CSV saved to: {csv_path}")


# ============================================================
# Main pipeline
# ============================================================
def process_single_dataset(csv_path, algos, basemap, output_dir):
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n{'=' * 60}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 60}")

    df, x_cols_orig, is_proj = load_dataset(csv_path)
    X, Y, coords, regions_orig, df_clean, sampled, x_cols = prepare_data(df, x_cols_orig, is_proj)

    if sampled:
        print(f"  Subsampled to {len(Y)} observations")
    print(f"  Shape: X={X.shape}, Y={Y.shape}, features={x_cols}")

    print("  Building spatial weights...")
    w = build_spatial_weights(coords)
    p = determine_p(regions_orig)
    print(f"  Target p={p}")

    results = {}
    labels_dict = {}

    for algo_name, algo_module in algos.items():
        print(f"  Running {algo_name.replace(chr(10), ' ')}...", end=' ', flush=True)
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

            short_algo = algo_short(algo_name)
            plot_intra_region(intra, algo_name, dataset_name, output_dir)
            plot_inter_region_coeff(inter, algo_name, dataset_name, x_cols, output_dir)
            plot_coefficient_maps(coords, X, Y, labels, algo_name, dataset_name, x_cols, basemap, output_dir)

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    if labels_dict:
        plot_regime_maps(coords, labels_dict, Y, dataset_name, basemap, output_dir)
        for algo_name, labels in labels_dict.items():
            plot_detail_maps(coords, labels, Y, X, algo_name, dataset_name, x_cols, basemap, output_dir)
            # Voronoi visualization for all datasets
            plot_voronoi_regions(coords, labels, algo_name, dataset_name, basemap, output_dir)
        if len(results) > 1:
            plot_algo_comparison(coords, Y, X, results, dataset_name, x_cols, basemap, output_dir)

    return results, x_cols, coords, Y, is_proj


def run_p_sensitivity(algos, us_states_4326, us_states_5070, output_dir):
    """Run p-sensitivity analysis for all datasets."""
    p_output = os.path.join(output_dir, 'p_sensitivity')
    os.makedirs(p_output, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
    all_p_results = {}

    for csv_file in csv_files:
        ds_name = os.path.splitext(csv_file)[0]
        if ds_name in SKIP_DATASETS:
            print(f"\nSkipping {ds_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"P-Sensitivity: {ds_name}")
        print(f"{'=' * 60}")

        try:
            csv_path = os.path.join(DATASET_DIR, csv_file)
            df, x_cols_orig, is_proj = load_dataset(csv_path)
            X, Y, coords, _, df_clean, sampled, x_cols = prepare_data(df, x_cols_orig, is_proj)
            if sampled:
                print(f"  Subsampled to {len(Y)}")
            print(f"  Shape: X={X.shape}, Y={Y.shape}")

            basemap = get_basemap_for_coords(coords, us_states_4326, us_states_5070)
            w = build_spatial_weights(coords)

            results_by_algo = {}

            for algo_name, algo_module in algos.items():
                results_by_algo[algo_name] = []
                labels_by_p = {}

                for p in P_VALUES:
                    min_size = max(X.shape[1], 10)
                    if p * min_size > len(Y) * 0.8:
                        continue

                    aname = algo_name.replace('\n', ' ')
                    print(f"  {aname} p={p}...", end=' ', flush=True)
                    try:
                        start = time.time()
                        labels = algo_module.run_two_stage_kmeans(
                            X, Y, p=p, w=w, min_size=min_size,
                            max_iter=300, init_stoc_step=True, verbose=False)
                        elapsed = time.time() - start
                        metrics = compute_metrics_quick(X, Y, labels)
                        metrics['p'] = p
                        metrics['runtime'] = elapsed
                        results_by_algo[algo_name].append(metrics)
                        labels_by_p[p] = labels
                        print(f"Done {elapsed:.1f}s, {metrics['n_regions']} reg, "
                              f"SSR_red={metrics['ssr_reduction']:.4f}")
                    except Exception as e:
                        print(f"FAILED: {e}")

                # Regime maps across p for last algo (GeoEvolve)
                if 'geoevolve' in algo_name.lower() and labels_by_p:
                    plot_p_regime_maps(coords, labels_by_p, sorted(labels_by_p.keys()),
                                      algo_name, ds_name, basemap, p_output)

            all_p_results[ds_name] = results_by_algo

            if any(results_by_algo[a] for a in results_by_algo):
                valid_ps = sorted(set(r['p'] for a in results_by_algo for r in results_by_algo[a]))
                plot_p_sensitivity_single(ds_name, valid_ps, results_by_algo, p_output)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Cross-dataset p-sensitivity plots
    if all_p_results:
        valid_ps = sorted(set(r['p'] for ds in all_p_results for a in all_p_results[ds]
                              for r in all_p_results[ds][a]))
        plot_p_sensitivity_all_datasets(all_p_results, valid_ps, p_output)
        plot_p_sensitivity_grid(all_p_results, valid_ps, p_output,
                               metric_key='ssr_reduction', ylabel='SSR Red.')
        plot_p_sensitivity_grid(all_p_results, valid_ps, p_output,
                               metric_key='ssr_regime', ylabel='SSR')
        plot_p_sensitivity_grid(all_p_results, valid_ps, p_output,
                               metric_key='runtime', ylabel='Runtime (s)')
        plot_p_sensitivity_grid(all_p_results, valid_ps, p_output,
                               metric_key='avg_r2', ylabel='Avg R^2')
        export_p_sensitivity_csv(all_p_results, p_output)

    return all_p_results


def main():
    print("=" * 60)
    print("Real-World Spatial Regime Analysis v2 (4 algorithms)")
    print("=" * 60)

    # Load basemaps
    us_states_4326 = load_us_basemap()
    us_states_5070 = load_us_basemap_5070()
    if us_states_4326 is not None:
        print(f"Loaded basemap (WGS84): {len(us_states_4326)} states")
    if us_states_5070 is not None:
        print(f"Loaded basemap (EPSG:5070): {len(us_states_5070)} states")

    algos = load_all_algorithms()
    print(f"Loaded {len(algos)} algorithms: {[a.replace(chr(10), ' ') for a in algos.keys()]}")

    all_results = {}
    x_cols_dict = {}
    all_coords = {}
    all_best_labels = {}
    all_basemaps = {}

    csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} datasets")

    # Phase 1: Run algorithms on each dataset
    print("\n" + "=" * 60)
    print("PHASE 1: Algorithm Execution")
    print("=" * 60)

    for csv_file in csv_files:
        ds_name = os.path.splitext(csv_file)[0]
        if ds_name in SKIP_DATASETS:
            print(f"\nSkipping {ds_name}")
            continue
        csv_path = os.path.join(DATASET_DIR, csv_file)
        try:
            # Quick load to determine CRS for basemap selection
            df_tmp, _, is_proj_tmp = load_dataset(csv_path)
            if is_proj_tmp:
                basemap = us_states_5070
            else:
                basemap = us_states_4326

            results, x_cols_out, coords_out, Y_out, is_proj_out = process_single_dataset(
                csv_path, algos, basemap, OUTPUT_DIR)

            all_results[ds_name] = results
            x_cols_dict[ds_name] = x_cols_out
            all_coords[ds_name] = coords_out
            all_basemaps[ds_name] = basemap

            # Pick best algo labels for summary maps
            best_algo = 'GeoEvolve' if 'GeoEvolve' in results else \
                        list(results.keys())[-1] if results else None
            if best_algo and best_algo in results:
                all_best_labels[ds_name] = results[best_algo]['labels']

        except Exception as e:
            print(f"\nERROR processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save raw results
    to_save = {}
    for ds, dr in all_results.items():
        to_save[ds] = {}
        for algo, ar in dr.items():
            to_save[ds][algo.replace('\n', ' ')] = {
                'runtime': ar['runtime'], 'summary': ar['summary']
            }
    with open(os.path.join(OUTPUT_DIR, 'results_summary.pkl'), 'wb') as f:
        pickle.dump(to_save, f)

    # Cross-dataset visualizations
    print("\n" + "=" * 60)
    print("PHASE 2: Cross-Dataset Visualization")
    print("=" * 60)
    plot_performance_comparison(all_results, OUTPUT_DIR)
    plot_summary_heatmap(all_results, OUTPUT_DIR)
    if all_best_labels:
        plot_all_datasets_regime_maps(
            {ds: all_coords[ds] for ds in all_best_labels},
            all_best_labels, list(all_best_labels.keys()),
            all_basemaps, OUTPUT_DIR)

    # Export CSV
    export_comparison_csv(all_results, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Dataset':<30} {'Algorithm':<22} {'SSR_red':>8} {'SSR':>12} {'#Reg':>5} {'R2':>6} {'Time':>7}")
    print("-" * 100)
    for ds in all_results:
        for algo in all_results[ds]:
            r = all_results[ds][algo]
            s = r['summary']
            intra = r['intra']
            r2_vals = [intra[reg]['r2'] for reg in intra if not np.isnan(intra[reg]['r2'])]
            avg_r2 = np.mean(r2_vals) if r2_vals else 0
            algo_label = algo.replace('\n', ' ')
            print(f"{ds:<30} {algo_label:<22} {s['ssr_reduction']:>8.4f} "
                  f"{s['total_ssr']:>12.1f} {s['n_regions']:>5} "
                  f"{avg_r2:>6.4f} {r['runtime']:>7.1f}")

    # Phase 3: P-sensitivity analysis
    print("\n" + "=" * 60)
    print("PHASE 3: P-Sensitivity Analysis")
    print("=" * 60)
    all_p_results = run_p_sensitivity(algos, us_states_4326, us_states_5070, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"All results saved to: {OUTPUT_DIR}")
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png') or f.endswith('.pdf')])
    p_dir = os.path.join(OUTPUT_DIR, 'p_sensitivity')
    if os.path.exists(p_dir):
        n_files += len([f for f in os.listdir(p_dir) if f.endswith('.png') or f.endswith('.pdf')])
    print(f"Total output files: {n_files}")


if __name__ == '__main__':
    main()
