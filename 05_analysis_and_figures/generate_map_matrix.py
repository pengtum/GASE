#!/usr/bin/env python3
"""
Generate a 4-row × 3-column map matrix showing spatial regime results.
Rows  = domains (Climate, Health, Hydrology, Politics)
Cols  = algorithms (Initial, OpenEvolve, GeoEvolve)

ALL maps use unified EPSG:5070 projection (Albers Equal-Area).
Lat/lon data is reprojected on-the-fly.
No axis borders – clean academic style with subcaption labels.
Output saved to paper_figures/.
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

warnings.filterwarnings('ignore')

# ── paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, 'all_datasets')
OUT_DIR     = os.path.join(BASE, 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── representative datasets (one per domain) ──
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

# Basemap (EPSG:5070 only)
US_STATES_5070 = gpd.read_file(os.path.join(BASE, 'us_states_conus_5070.gpkg'))

# Unified CONUS extent in EPSG:5070
CONUS_XLIM = (-2.35e6, 2.25e6)
CONUS_YLIM = (0.27e6, 3.15e6)

# Transformer: WGS84 -> EPSG:5070
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

# Regime colors
TAB10 = plt.cm.tab10


# ============================================================
# Data loading
# ============================================================
def load_algorithm(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def identify_continuous_cols(df, x_cols, min_unique=15):
    continuous = []
    for col in x_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'str':
            continue
        if df[col].nunique() < min_unique:
            continue
        continuous.append(col)
    return continuous


def load_and_prepare(dataset_name):
    """Load dataset, prepare X, Y, coords (always in EPSG:5070), spatial weights."""
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
        df['coord_x'] = df['lon']
        df['coord_y'] = df['lat']
    else:
        raise ValueError(f"Cannot find coordinate columns in {csv_path}")

    x_cols = sorted([c for c in df.columns if c.startswith('X') and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
    x_cols = identify_continuous_cols(df, x_cols)

    key_cols = ['Y'] + x_cols + ['coord_x', 'coord_y']
    df_clean = df.dropna(subset=key_cols).copy()

    # Filter non-CONUS for geographic data
    if not is_proj:
        cx, cy = df_clean['coord_x'], df_clean['coord_y']
        conus_mask = (cx > -130) & (cx < -60) & (cy > 23) & (cy < 52)
        df_clean = df_clean[conus_mask].reset_index(drop=True)

    # Subsample
    if len(df_clean) > MAX_SAMPLES:
        df_clean = df_clean.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    Y = df_clean['Y'].values
    coords_raw = df_clean[['coord_x', 'coord_y']].values
    regions_orig = df_clean['region'].values if 'region' in df_clean.columns else None

    # ── Reproject lat/lon to EPSG:5070 ──
    if not is_proj:
        px, py = TRANSFORMER.transform(coords_raw[:, 0], coords_raw[:, 1])
        coords_5070 = np.column_stack([px, py])
    else:
        coords_5070 = coords_raw.copy()

    # Use original coords for spatial weights (algorithm needs original CRS)
    X_raw = df_clean[x_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    # p
    if regions_orig is not None:
        n_orig = len(np.unique(regions_orig))
        p = min(n_orig, 10) if n_orig <= 30 else 8
        p = n_orig if n_orig <= 15 else p
    else:
        p = 5

    # Spatial weights on original coordinates
    w = KNN.from_array(coords_raw, k=K_NEIGHBORS)
    w.transform = 'r'

    return X, Y, coords_raw, coords_5070, p, w


# ============================================================
# Run single algorithm
# ============================================================
def run_algo(algo_module, X, Y, p, w):
    min_size = max(X.shape[1], 10)
    labels = algo_module.run_two_stage_kmeans(
        X, Y, p=p, w=w, min_size=min_size, max_iter=300,
        init_stoc_step=True, verbose=False)
    return labels


# ============================================================
# Main
# ============================================================
def main():
    # Load algorithms
    algo_modules = {}
    for name, fname in ALGO_FILES.items():
        fpath = os.path.join(BASE, fname)
        algo_modules[name] = load_algorithm(f'algo_{name}', fpath)
    print("Algorithms loaded.")

    # Run all combinations
    all_labels  = {}   # (domain, algo) -> labels
    all_coords  = {}   # domain -> coords_5070
    all_p       = {}

    for domain, ds_name in REPRESENTATIVE:
        print(f"\n── {domain}: {ds_name} ──")
        X, Y, coords_raw, coords_5070, p, w = load_and_prepare(ds_name)
        all_coords[domain] = coords_5070
        all_p[domain] = p
        print(f"   N={len(Y)}, p={p}")

        for algo_name in ALGOS:
            print(f"   Running {algo_name}...", end=' ', flush=True)
            t0 = time.time()
            labels = run_algo(algo_modules[algo_name], X, Y, p, w)
            elapsed = time.time() - t0
            n_reg = len(np.unique(labels))
            print(f"{n_reg} regions, {elapsed:.1f}s")
            all_labels[(domain, algo_name)] = labels

    # ── Plot 4×3 matrix (no borders, EPSG:5070 unified) ──
    print("\nGenerating map matrix...")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 7,
    })

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.0))

    # Subcaption labels: (a) through (l)
    sub_labels = [chr(ord('a') + i) for i in range(12)]
    panel_idx = 0

    # Domain display names for subcaptions
    DS_DISPLAY = {
        'Climate':   'Climate (ERA5)',
        'Health':    'Health (Cancer)',
        'Hydrology': 'Hydrology (CAMELS)',
        'Politics':  'Politics (Voting)',
    }

    for ri, (domain, ds_name) in enumerate(REPRESENTATIVE):
        coords = all_coords[domain]

        for ci, algo_name in enumerate(ALGOS):
            ax = axes[ri, ci]
            labels = all_labels[(domain, algo_name)]
            n_reg = len(np.unique(labels))

            # ── Remove all borders/spines ──
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # ── Basemap (light gray fill + thin boundary) ──
            US_STATES_5070.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc',
                                linewidth=0.25)

            # ── Scatter regime points ──
            unique_labels = np.unique(labels)
            for li, lab in enumerate(unique_labels):
                mask = labels == lab
                color = TAB10(li % 10)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[color], s=1.2, alpha=0.85,
                           edgecolors='none', rasterized=True)

            # ── Consistent EPSG:5070 extent ──
            ax.set_xlim(CONUS_XLIM)
            ax.set_ylim(CONUS_YLIM)
            ax.set_aspect('equal')

            # ── Column header (algorithm) – top row only ──
            if ri == 0:
                ax.set_title(algo_name, fontsize=9, fontweight='bold', pad=5)

            # ── Row label (domain) – left column only ──
            if ci == 0:
                ax.annotate(
                    domain, xy=(-0.05, 0.5), xycoords='axes fraction',
                    fontsize=9, fontweight='bold', rotation=90,
                    ha='right', va='center')

            # ── Subcaption below each panel: (a) Dataset – Algorithm (k=N) ──
            cap = f'({sub_labels[panel_idx]}) {DS_DISPLAY[domain]} – {algo_name} (k={n_reg})'
            ax.text(0.5, -0.02, cap, transform=ax.transAxes,
                    fontsize=6, ha='center', va='top')

            panel_idx += 1

    plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.03,
                        hspace=0.12, wspace=0.02)

    # Save
    for ext in ['pdf', 'png']:
        out = os.path.join(OUT_DIR, f'regime_map_matrix.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out}")
    plt.close()
    print("Done.")


if __name__ == '__main__':
    main()
