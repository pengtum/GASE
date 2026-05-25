#!/usr/bin/env python3
"""
Generate two 4×3 map matrices:
  1. Spatial Residuals   (continuous RdBu colormap)
  2. Dominant Covariate  (discrete legend with real variable names from Metadata)

Rows  = domains (Climate, Health, Hydrology, Politics)
Cols  = algorithms (Initial, OpenEvolve, GeoEvolve)
All maps in unified EPSG:5070; no axis borders.
"""

import os, sys, time, importlib.util, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
import libpysal
from libpysal.weights import KNN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

# ── paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, 'all_datasets')
OUT_DIR     = os.path.join(BASE, 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── representative datasets ──
REPRESENTATIVE = [
    ('Climate',   'US_Climate_ERA5_CLIMATE'),
    ('Health',    'US_Health_CANCER'),
    ('Hydrology', 'US_Hydro_CAMELS'),
    ('Politics',  'US_Politics_Voting'),
]

ALGOS = ['Initial', 'OpenEvolve', 'GeoEvolve']
ALGO_FILES = {
    'Initial':    'initial_program.py',
    'OpenEvolve': 'best_program_openevolve.py',
    'GeoEvolve':  'best_program_geoevolve.py',
}

MAX_SAMPLES  = 2000
K_NEIGHBORS  = 6

# Basemap
US_STATES_5070 = gpd.read_file(os.path.join(BASE, 'us_states_conus_5070.gpkg'))
CONUS_XLIM = (-2.35e6, 2.25e6)
CONUS_YLIM = (0.27e6, 3.15e6)
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

# ── Real variable name mapping (from Metadata.xlsx) ──
# Key = dataset_name, Value = dict mapping X-column-index (0-based) -> short label
# Index 0 = X1, index 1 = X2, ...
VAR_NAMES = {
    'US_Climate_ERA5_CLIMATE': {
        0: 'Precip.', 1: 'Surf. Press.', 2: 'Wind-U', 3: 'Wind-V',
        4: 'Pot. Evap.', 5: 'Therm. Rad.', 6: 'Solar Rad.',
        7: 'LAI (high)', 8: 'LAI (low)',
    },
    'US_Health_CANCER': {
        0: 'No Insur.', 1: 'Checkup', 2: 'Dental', 3: 'Binge',
        4: 'Smoking', 5: 'No Exercise', 6: 'Short Sleep',
        7: 'No Emot. Spt.',
    },
    'US_Hydro_CAMELS': {
        0: 'Precip.', 1: 'PET', 2: 'Aridity', 3: 'Elevation',
        4: 'Slope', 5: 'Forest Frac.', 6: 'LAI max',
        7: 'Soil Depth', 8: 'Sand Frac.', 9: 'Porosity',
    },
    'US_Politics_Voting': {
        0: 'Sex Ratio', 1: '% Black', 2: '% Hispanic',
        3: '% Bachelor', 4: 'Med. Income', 5: '% Age 65+',
        6: '% Age 18-29', 7: 'Gini', 8: '% Manuf.',
        9: 'ln(Pop. Den.)', 10: '% 3rd Party', 11: 'Turnout',
        12: '% Foreign', 13: '% Uninsured',
    },
}

# Distinct colors for dominant covariates (max 14 needed for Politics)
QUAL_COLORS = (
    list(plt.cm.Set2.colors) +      # 8 colors
    list(plt.cm.Dark2.colors) +      # 8 colors
    list(plt.cm.tab10.colors)        # 10 colors
)


# ============================================================
# Data helpers  (same as generate_map_matrix.py)
# ============================================================
def load_algorithm(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def identify_continuous_cols(df, x_cols, min_unique=15):
    return [c for c in x_cols
            if df[c].dtype not in ('object',) and df[c].nunique() >= min_unique]


def load_and_prepare(dataset_name):
    csv_path = os.path.join(DATASET_DIR, f'{dataset_name}.csv')
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    is_proj = False
    if 'proj_x' in df.columns:
        df = df.rename(columns={'proj_x': 'coord_x', 'proj_y': 'coord_y'})
        is_proj = True
    elif 'proj_X' in df.columns:
        df = df.rename(columns={'proj_X': 'coord_x', 'proj_Y': 'coord_y'})
        is_proj = True
    elif 'lat' in df.columns and 'lon' in df.columns:
        df['coord_x'] = df['lon']; df['coord_y'] = df['lat']

    x_cols = sorted([c for c in df.columns if c.startswith('X') and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
    x_cols = identify_continuous_cols(df, x_cols)

    key_cols = ['Y'] + x_cols + ['coord_x', 'coord_y']
    df_clean = df.dropna(subset=key_cols).copy()

    if not is_proj:
        cx, cy = df_clean['coord_x'], df_clean['coord_y']
        df_clean = df_clean[(cx > -130) & (cx < -60) & (cy > 23) & (cy < 52)].reset_index(drop=True)

    if len(df_clean) > MAX_SAMPLES:
        df_clean = df_clean.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    Y = df_clean['Y'].values
    coords_raw = df_clean[['coord_x', 'coord_y']].values

    if not is_proj:
        px, py = TRANSFORMER.transform(coords_raw[:, 0], coords_raw[:, 1])
        coords_5070 = np.column_stack([px, py])
    else:
        coords_5070 = coords_raw.copy()

    X_raw = df_clean[x_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    regions_orig = df_clean['region'].values if 'region' in df_clean.columns else None
    if regions_orig is not None:
        n_orig = len(np.unique(regions_orig))
        p = n_orig if n_orig <= 15 else (min(n_orig, 10) if n_orig <= 30 else 8)
    else:
        p = 5

    w = KNN.from_array(coords_raw, k=K_NEIGHBORS)
    w.transform = 'r'

    return X, Y, coords_raw, coords_5070, p, w, len(x_cols)


def run_algo(algo_module, X, Y, p, w):
    min_size = max(X.shape[1], 10)
    return algo_module.run_two_stage_kmeans(
        X, Y, p=p, w=w, min_size=min_size, max_iter=300,
        init_stoc_step=True, verbose=False)


def compute_region_stats(X, Y, labels):
    """Per-region OLS -> residuals + beta."""
    unique = np.unique(labels)
    stats = {}
    for r in unique:
        mask = labels == r
        Xr, Yr = X[mask], Y[mask]
        n_r = len(Yr)
        if n_r < X.shape[1]:
            stats[r] = {'beta': None, 'residuals': np.full(n_r, np.nan)}
            continue
        beta = np.linalg.pinv(Xr).dot(Yr)
        resid = Yr - Xr.dot(beta)
        stats[r] = {'beta': beta, 'residuals': resid}
    return stats


# ============================================================
# Main
# ============================================================
def main():
    # Load algorithms
    algo_modules = {}
    for name, fname in ALGO_FILES.items():
        fpath = os.path.join(BASE, fname)
        algo_modules[name] = load_algorithm(f'algo_{name}', fpath)
    print("Algorithms loaded.\n")

    # Run all combinations and compute metrics
    all_labels   = {}
    all_coords   = {}
    all_residuals = {}   # (domain, algo) -> array of residuals per point
    all_dom_feat  = {}   # (domain, algo) -> array of dominant covariate index per point
    all_n_x      = {}    # domain -> number of X variables (excluding intercept)
    ds_names     = {}    # domain -> dataset_name

    for domain, ds_name in REPRESENTATIVE:
        print(f"── {domain}: {ds_name} ──")
        X, Y, coords_raw, coords_5070, p, w, n_x = load_and_prepare(ds_name)
        all_coords[domain] = coords_5070
        all_n_x[domain] = n_x
        ds_names[domain] = ds_name
        print(f"   N={len(Y)}, p={p}, n_x={n_x}")

        for algo_name in ALGOS:
            print(f"   Running {algo_name}...", end=' ', flush=True)
            t0 = time.time()
            labels = run_algo(algo_modules[algo_name], X, Y, p, w)
            elapsed = time.time() - t0
            n_reg = len(np.unique(labels))
            print(f"{n_reg} regions, {elapsed:.1f}s")
            all_labels[(domain, algo_name)] = labels

            # Compute per-region OLS
            rstats = compute_region_stats(X, Y, labels)

            # Full residual array
            full_resid = np.zeros(len(Y))
            for r in np.unique(labels):
                mask = labels == r
                if rstats[r]['residuals'] is not None:
                    full_resid[mask] = rstats[r]['residuals']
            all_residuals[(domain, algo_name)] = full_resid

            # Dominant covariate index (0-based, among X1..Xn)
            dom_feat = np.full(len(Y), -1, dtype=int)
            for r in np.unique(labels):
                mask = labels == r
                beta = rstats[r]['beta']
                if beta is not None and len(beta) > 1:
                    dom_feat[mask] = int(np.argmax(np.abs(beta[1:])))  # skip intercept
            all_dom_feat[(domain, algo_name)] = dom_feat

    # ================================================================
    # FIGURE 1: Spatial Residuals (4×3)
    # ================================================================
    print("\n── Generating Spatial Residual map matrix ──")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7})

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.0))
    sub_labels = [chr(ord('a') + i) for i in range(12)]
    DS_DISPLAY = {
        'Climate': 'Climate (ERA5)', 'Health': 'Health (Cancer)',
        'Hydrology': 'Hydrology (CAMELS)', 'Politics': 'Politics (Voting)',
    }

    # Compute global vmax per domain (shared across 3 algos for fair comparison)
    domain_vmax = {}
    for domain, _ in REPRESENTATIVE:
        all_resid_dom = np.concatenate([
            all_residuals[(domain, a)] for a in ALGOS])
        domain_vmax[domain] = np.percentile(np.abs(all_resid_dom), 95)

    panel_idx = 0
    for ri, (domain, _) in enumerate(REPRESENTATIVE):
        coords = all_coords[domain]
        vmax = domain_vmax[domain]

        for ci, algo_name in enumerate(ALGOS):
            ax = axes[ri, ci]
            resid = all_residuals[(domain, algo_name)]

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

            US_STATES_5070.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.25)

            sc = ax.scatter(coords[:, 0], coords[:, 1], c=resid,
                            cmap='RdBu_r', s=1.2, alpha=0.85,
                            vmin=-vmax, vmax=vmax,
                            edgecolors='none', rasterized=True)

            ax.set_xlim(CONUS_XLIM); ax.set_ylim(CONUS_YLIM)
            ax.set_aspect('equal')

            if ri == 0:
                ax.set_title(algo_name, fontsize=9, fontweight='bold', pad=5)
            if ci == 0:
                ax.annotate(domain, xy=(-0.05, 0.5), xycoords='axes fraction',
                            fontsize=9, fontweight='bold', rotation=90,
                            ha='right', va='center')

            cap = f'({sub_labels[panel_idx]}) {DS_DISPLAY[domain]} – {algo_name}'
            ax.text(0.5, -0.02, cap, transform=ax.transAxes,
                    fontsize=6, ha='center', va='top')

            # Colorbar for rightmost column
            if ci == 2:
                cax = fig.add_axes([
                    ax.get_position().x1 + 0.005,
                    ax.get_position().y0 + ax.get_position().height * 0.15,
                    0.008,
                    ax.get_position().height * 0.7
                ])
                cb = plt.colorbar(sc, cax=cax)
                cb.ax.tick_params(labelsize=5)
                cb.set_label('Residual', fontsize=5.5, labelpad=2)

            panel_idx += 1

    plt.subplots_adjust(left=0.06, right=0.92, top=0.95, bottom=0.03,
                        hspace=0.12, wspace=0.02)

    for ext in ['pdf', 'png']:
        out = os.path.join(OUT_DIR, f'residual_map_matrix.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out}")
    plt.close()

    # ================================================================
    # FIGURE 2: Dominant Covariate (4×3) with discrete legend
    # ================================================================
    print("\n── Generating Dominant Covariate map matrix ──")

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.5))
    panel_idx = 0

    for ri, (domain, ds_name) in enumerate(REPRESENTATIVE):
        coords = all_coords[domain]
        n_x = all_n_x[domain]
        var_map = VAR_NAMES.get(ds_name, {i: f'X{i+1}' for i in range(n_x)})

        # Collect all appearing covariate indices across 3 algos
        all_feat_indices = set()
        for algo_name in ALGOS:
            df = all_dom_feat[(domain, algo_name)]
            all_feat_indices.update(df[df >= 0].tolist())
        feat_indices = sorted(all_feat_indices)

        # Build color mapping: covariate index -> color
        idx_to_color = {}
        for ci_feat, fidx in enumerate(feat_indices):
            idx_to_color[fidx] = QUAL_COLORS[ci_feat % len(QUAL_COLORS)]

        for ci, algo_name in enumerate(ALGOS):
            ax = axes[ri, ci]
            dom_feat = all_dom_feat[(domain, algo_name)]

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

            US_STATES_5070.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.25)

            # Plot each dominant covariate group
            for fidx in feat_indices:
                mask = dom_feat == fidx
                if mask.sum() == 0:
                    continue
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[idx_to_color[fidx]], s=1.2, alpha=0.85,
                           edgecolors='none', rasterized=True)

            ax.set_xlim(CONUS_XLIM); ax.set_ylim(CONUS_YLIM)
            ax.set_aspect('equal')

            if ri == 0:
                ax.set_title(algo_name, fontsize=9, fontweight='bold', pad=5)
            if ci == 0:
                ax.annotate(domain, xy=(-0.05, 0.5), xycoords='axes fraction',
                            fontsize=9, fontweight='bold', rotation=90,
                            ha='right', va='center')

            cap = f'({sub_labels[panel_idx]}) {DS_DISPLAY[domain]} – {algo_name}'
            ax.text(0.5, -0.02, cap, transform=ax.transAxes,
                    fontsize=6, ha='center', va='top')
            panel_idx += 1

        # Add shared discrete legend for this row (right side)
        # Collect legend handles
        handles = []
        for fidx in feat_indices:
            label = var_map.get(fidx, f'X{fidx+1}')
            handles.append(Patch(facecolor=idx_to_color[fidx],
                                 edgecolor='none', label=label))

        # Place legend to the right of the last column
        ax_last = axes[ri, 2]
        ax_last.legend(handles=handles, loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       fontsize=5, frameon=False,
                       handlelength=0.8, handletextpad=0.3,
                       labelspacing=0.25, borderpad=0.2)

    plt.subplots_adjust(left=0.06, right=0.84, top=0.95, bottom=0.03,
                        hspace=0.12, wspace=0.02)

    for ext in ['pdf', 'png']:
        out = os.path.join(OUT_DIR, f'dominant_covariate_map_matrix.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out}")
    plt.close()

    print("\nAll done.")


if __name__ == '__main__':
    main()
