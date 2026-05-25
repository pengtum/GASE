"""4-row × 7-column map matrix for the 7 evolved variants on representative
real-world datasets. Reads results_evolved/<ds>_labels.npz.
"""
import os
import warnings

import numpy as np
import geopandas as gpd
from pyproj import Transformer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(BASE, "results_evolved")
OUT_DIR = os.path.join(BASE, "paper_figures_combined")
os.makedirs(OUT_DIR, exist_ok=True)

REPRESENTATIVE = [
    ("Climate", "US_Climate_ERA5_CLIMATE"),
    ("Health", "US_Health_CANCER"),
    ("Hydrology", "US_Hydro_CAMELS"),
    ("Politics", "US_Politics_Voting"),
]

ALGOS = [
    "2kmodels (initial)",
    "OpenEvolve_NoGeoKnow",
    "OpenEvolve_SimpleGeoKnow",
    "OpenEvolve_SpecificGeoKnow",
    "GeoEvolve_NoRAG",
    "GeoEvolve_StaticRAG",
]
ALGO_DISPLAY = {
    "2kmodels (initial)": "2kmodels",
    "OpenEvolve_NoGeoKnow": "OE-NoGeo",
    "OpenEvolve_SimpleGeoKnow": "OE-SimpGeo",
    "OpenEvolve_SpecificGeoKnow": "OE-SpecGeo",
    "GeoEvolve_NoRAG": "GE-NoRAG",
    "GeoEvolve_StaticRAG": "GeoEvolve",
}

US_STATES_5070 = gpd.read_file(os.path.join(BASE, "us_states_conus_5070.gpkg"))
CONUS_XLIM = (-2.35e6, 2.25e6)
CONUS_YLIM = (0.27e6, 3.15e6)
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
TAB10 = plt.cm.tab10


def is_latlon(coords):
    return (coords[:, 0].min() > -130 and coords[:, 0].max() < -60 and
            coords[:, 1].min() > 23 and coords[:, 1].max() < 52)


def to_5070(coords):
    if is_latlon(coords):
        px, py = TRANSFORMER.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([px, py])
    return coords.copy()


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7})
    fig, axes = plt.subplots(len(REPRESENTATIVE), len(ALGOS),
                             figsize=(2.0 * len(ALGOS), 2.0 * len(REPRESENTATIVE)))
    sub = [chr(ord("a") + i) for i in range(len(REPRESENTATIVE) * len(ALGOS))]
    panel = 0
    DS_DISPLAY = {
        "Climate": "Climate (ERA5)",
        "Health": "Health (Cancer)",
        "Hydrology": "Hydrology (CAMELS)",
        "Politics": "Politics (Voting)",
    }

    for ri, (domain, ds_name) in enumerate(REPRESENTATIVE):
        path = os.path.join(LABEL_DIR, f"{ds_name}_labels.npz")
        if not os.path.exists(path):
            for ci in range(len(ALGOS)):
                axes[ri, ci].axis("off")
            continue
        npz = np.load(path, allow_pickle=False)
        coords = npz["coords"]
        coords_5070 = to_5070(coords)

        for ci, algo in enumerate(ALGOS):
            ax = axes[ri, ci]
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            US_STATES_5070.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.25)
            if algo not in npz.files:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="#999")
            else:
                labels = npz[algo]
                uniq = np.unique(labels)
                for li, lab in enumerate(uniq):
                    m = labels == lab
                    color = TAB10(li % 10)
                    ax.scatter(coords_5070[m, 0], coords_5070[m, 1],
                               c=[color], s=1.0, alpha=0.85, edgecolors="none",
                               rasterized=True)
                ax.text(0.5, -0.02, f"({sub[panel]}) {DS_DISPLAY[domain]} – {ALGO_DISPLAY[algo]} (k={len(uniq)})",
                        transform=ax.transAxes, fontsize=5.5, ha="center", va="top")
            ax.set_xlim(CONUS_XLIM); ax.set_ylim(CONUS_YLIM); ax.set_aspect("equal")
            if ri == 0:
                ax.set_title(ALGO_DISPLAY[algo], fontsize=8, fontweight="bold", pad=5)
            if ci == 0:
                ax.annotate(domain, xy=(-0.05, 0.5), xycoords="axes fraction",
                            fontsize=8, fontweight="bold", rotation=90,
                            ha="right", va="center")
            panel += 1

    plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.04,
                        hspace=0.20, wspace=0.02)
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"evolved_regime_map_matrix.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
