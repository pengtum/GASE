#!/usr/bin/env python3
"""
Generate 4×3 map matrix of Local R² by Region.
Rows  = domains (Climate, Health, Hydrology, Politics)
Cols  = algorithms (Initial, OpenEvolve, GeoEvolve)

Each point is coloured by the R² of its assigned region's OLS model.
RdYlGn colormap, range [0, 1], one shared colorbar per row.
Unified EPSG:5070, no axis borders.
"""

import os, time, importlib.util, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
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

# ── datasets ──
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
MAX_SAMPLES = 2000
K_NEIGHBORS = 6

# basemap & projection
US_STATES_5070 = gpd.read_file(os.path.join(BASE, 'us_states_conus_5070.gpkg'))
CONUS_XLIM = (-2.35e6, 2.25e6)
CONUS_YLIM = (0.27e6, 3.15e6)
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


# ============================================================
# helpers (same as other map-matrix scripts)
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

    w = KNN.from_array(coords_raw, k=K_NEIGHBORS); w.transform = 'r'
    return X, Y, coords_raw, coords_5070, p, w


def run_algo(mod, X, Y, p, w):
    min_size = max(X.shape[1], 10)
    return mod.run_two_stage_kmeans(
        X, Y, p=p, w=w, min_size=min_size, max_iter=300,
        init_stoc_step=True, verbose=False)


def compute_local_r2(X, Y, labels):
    """Return per-point R² (each point gets the R² of its region)."""
    r2_map = np.zeros(len(Y))
    for r in np.unique(labels):
        mask = labels == r
        Xr, Yr = X[mask], Y[mask]
        if len(Yr) < X.shape[1]:
            r2_map[mask] = np.nan
            continue
        beta = np.linalg.pinv(Xr).dot(Yr)
        resid = Yr - Xr.dot(beta)
        ssr = np.sum(resid ** 2)
        sst = np.sum((Yr - np.mean(Yr)) ** 2)
        r2 = 1 - ssr / sst if sst > 0 else np.nan
        r2_map[mask] = r2
    return r2_map


# ============================================================
def main():
    algo_modules = {}
    for name, fname in ALGO_FILES.items():
        algo_modules[name] = load_algorithm(f'algo_{name}', os.path.join(BASE, fname))
    print("Algorithms loaded.\n")

    all_coords = {}
    all_r2     = {}   # (domain, algo) -> r2 array

    for domain, ds_name in REPRESENTATIVE:
        print(f"── {domain}: {ds_name} ──")
        X, Y, coords_raw, coords_5070, p, w = load_and_prepare(ds_name)
        all_coords[domain] = coords_5070
        print(f"   N={len(Y)}, p={p}")

        for algo_name in ALGOS:
            print(f"   Running {algo_name}...", end=' ', flush=True)
            t0 = time.time()
            labels = run_algo(algo_modules[algo_name], X, Y, p, w)
            elapsed = time.time() - t0
            n_reg = len(np.unique(labels))
            r2_map = compute_local_r2(X, Y, labels)
            print(f"{n_reg} regions, {elapsed:.1f}s, "
                  f"R² range [{np.nanmin(r2_map):.3f}, {np.nanmax(r2_map):.3f}]")
            all_r2[(domain, algo_name)] = r2_map

    # ── Plot ──
    print("\nGenerating R² map matrix...")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7})

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.0))
    sub_labels = [chr(ord('a') + i) for i in range(12)]
    DS_DISPLAY = {
        'Climate': 'Climate (ERA5)', 'Health': 'Health (Cancer)',
        'Hydrology': 'Hydrology (CAMELS)', 'Politics': 'Politics (Voting)',
    }

    panel_idx = 0
    for ri, (domain, _) in enumerate(REPRESENTATIVE):
        coords = all_coords[domain]
        for ci, algo_name in enumerate(ALGOS):
            ax = axes[ri, ci]
            r2 = all_r2[(domain, algo_name)]

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])

            US_STATES_5070.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.25)

            sc = ax.scatter(coords[:, 0], coords[:, 1], c=r2,
                            cmap='RdYlGn', s=1.2, alpha=0.85,
                            vmin=0, vmax=1,
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

            # Shared colorbar on rightmost column
            if ci == 2:
                cax = fig.add_axes([
                    ax.get_position().x1 + 0.005,
                    ax.get_position().y0 + ax.get_position().height * 0.15,
                    0.008,
                    ax.get_position().height * 0.7
                ])
                cb = plt.colorbar(sc, cax=cax)
                cb.ax.tick_params(labelsize=5)
                cb.set_label('R²', fontsize=6, labelpad=2)

            panel_idx += 1

    plt.subplots_adjust(left=0.06, right=0.92, top=0.95, bottom=0.03,
                        hspace=0.12, wspace=0.02)

    for ext in ['pdf', 'png']:
        out = os.path.join(OUT_DIR, f'r2_map_matrix.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out}")
    plt.close()
    print("Done.")


if __name__ == '__main__':
    main()
