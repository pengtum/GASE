"""4-row × 6-column R² map matrix: domains × algorithms.
For each unit, color encodes the within-region OLS R² of the unit's region.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(BASE, "results_baselines")
DATASET_DIR = os.path.join(BASE, "all_datasets")
OUT_DIR = os.path.join(BASE, "paper_figures_baselines")
os.makedirs(OUT_DIR, exist_ok=True)

REPRESENTATIVE = [
    ("Climate", "US_Climate_ERA5_CLIMATE"),
    ("Health", "US_Health_CANCER"),
    ("Hydrology", "US_Hydro_CAMELS"),
    ("Politics", "US_Politics_Voting"),
]

ALGOS = ["AZP", "RegionKModels", "GWR+SKATER", "SKATER-reg", "SHAP-based", "GeoEvolve"]
ALGO_DISPLAY = {
    "AZP": "AZP", "RegionKModels": "RKM", "GWR+SKATER": "GWR+SKATER",
    "SKATER-reg": "SKATER-reg", "SHAP-based": "SHAP-based",
    "GeoEvolve": "GeoEvolve (ours)",
}

US_STATES_5070 = gpd.read_file(os.path.join(BASE, "us_states_conus_5070.gpkg"))
CONUS_XLIM = (-2.35e6, 2.25e6)
CONUS_YLIM = (0.27e6, 3.15e6)
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
MAX_SAMPLES = 2000


def is_latlon(coords):
    return (coords[:, 0].min() > -130 and coords[:, 0].max() < -60 and
            coords[:, 1].min() > 23 and coords[:, 1].max() < 52)


def to_5070(coords):
    if is_latlon(coords):
        px, py = TRANSFORMER.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([px, py])
    return coords.copy()


def load_dataset(ds_name):
    """Recompute X, Y matched to the row order used during baseline runs."""
    csv_path = os.path.join(DATASET_DIR, f"{ds_name}.csv")
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    is_proj = False
    if "proj_x" in df.columns:
        df = df.rename(columns={"proj_x": "coord_x", "proj_y": "coord_y"}); is_proj = True
    elif "proj_X" in df.columns:
        df = df.rename(columns={"proj_X": "coord_x", "proj_Y": "coord_y"}); is_proj = True
    elif "lat" in df.columns and "lon" in df.columns:
        df["coord_x"] = df["lon"]; df["coord_y"] = df["lat"]
    x_cols = sorted([c for c in df.columns if c.startswith("X") and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
    x_cols = [c for c in x_cols if df[c].dtype != "object" and df[c].nunique() >= 15]
    key = ["Y"] + x_cols + ["coord_x", "coord_y"]
    d = df.dropna(subset=key).copy()
    if not is_proj:
        cx, cy = d["coord_x"], d["coord_y"]
        if (cx.min() < -130 or cy.max() > 55) and (cx.min() > -200 and cx.max() < 0):
            d = d[(cx > -130) & (cx < -60) & (cy > 23) & (cy < 52)].reset_index(drop=True)
    if len(d) > MAX_SAMPLES:
        d = d.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)
    Y = d["Y"].values
    Xraw = d[x_cols].values.astype(float)
    Xs = StandardScaler().fit_transform(Xraw)
    X = np.column_stack([np.ones(len(Xs)), Xs])
    return X, Y


def per_unit_r2(X, Y, labels):
    out = np.zeros(len(labels))
    for r in np.unique(labels):
        mask = labels == r
        Xr, Yr = X[mask], Y[mask]
        if len(Yr) < X.shape[1]:
            continue
        beta = np.linalg.pinv(Xr).dot(Yr)
        resid = Yr - Xr.dot(beta)
        ssr = float(np.sum(resid ** 2))
        sst = float(np.sum((Yr - np.mean(Yr)) ** 2))
        r2 = 1 - ssr / sst if sst > 0 else 0
        out[mask] = max(0.0, min(1.0, r2))
    return out


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7})
    fig, axes = plt.subplots(len(REPRESENTATIVE), len(ALGOS),
                             figsize=(2.4 * len(ALGOS), 2.0 * len(REPRESENTATIVE)))

    DS_DISPLAY = {
        "Climate": "Climate (ERA5)",
        "Health": "Health (Cancer)",
        "Hydrology": "Hydrology (CAMELS)",
        "Politics": "Politics (Voting)",
    }
    sub = [chr(ord("a") + i) for i in range(len(REPRESENTATIVE) * len(ALGOS))]
    panel = 0
    cmap = "RdYlGn"

    last_sc = None
    for ri, (domain, ds_name) in enumerate(REPRESENTATIVE):
        path = os.path.join(LABEL_DIR, f"{ds_name}_labels.npz")
        if not os.path.exists(path):
            for ci in range(len(ALGOS)):
                axes[ri, ci].axis("off")
            continue
        npz = np.load(path, allow_pickle=False)
        coords = npz["coords"]
        coords_5070 = to_5070(coords)
        X, Y = load_dataset(ds_name)

        for ci, algo in enumerate(ALGOS):
            ax = axes[ri, ci]
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            US_STATES_5070.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.25)

            if algo not in npz.files:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#999")
            else:
                labels = npz[algo]
                if len(labels) != len(X):
                    ax.text(0.5, 0.5, "shape\nmismatch", transform=ax.transAxes,
                            ha="center", va="center", fontsize=6, color="#c33")
                else:
                    r2 = per_unit_r2(X, Y, labels)
                    sc = ax.scatter(coords_5070[:, 0], coords_5070[:, 1], c=r2,
                                    cmap=cmap, vmin=0, vmax=1, s=1.0, alpha=0.9,
                                    edgecolors="none", rasterized=True)
                    last_sc = sc

            ax.set_xlim(CONUS_XLIM); ax.set_ylim(CONUS_YLIM); ax.set_aspect("equal")
            if ri == 0:
                ax.set_title(ALGO_DISPLAY[algo], fontsize=8, fontweight="bold", pad=5)
            if ci == 0:
                ax.annotate(domain, xy=(-0.05, 0.5), xycoords="axes fraction",
                            fontsize=8, fontweight="bold", rotation=90,
                            ha="right", va="center")
            cap = f"({sub[panel]}) {DS_DISPLAY[domain]} – {ALGO_DISPLAY[algo]}"
            ax.text(0.5, -0.02, cap, transform=ax.transAxes,
                    fontsize=5.5, ha="center", va="top")
            panel += 1

    plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.04,
                        hspace=0.18, wspace=0.02)
    if last_sc is not None:
        cax = fig.add_axes([0.94, 0.30, 0.012, 0.4])
        cb = fig.colorbar(last_sc, cax=cax)
        cb.set_label("Within-region R²", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"baseline_r2_map_matrix.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
