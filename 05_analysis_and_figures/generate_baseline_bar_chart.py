"""Grouped bar chart of all baseline algorithms + GeoEvolve, grouped by domain.
Mirrors generate_bar_chart.py but reads results_baselines/algorithm_comparison.csv
and writes to paper_figures_baselines/.
"""
import csv
import os
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "results_baselines", "algorithm_comparison.csv")
OUT_DIR = os.path.join(BASE, "paper_figures_baselines")
os.makedirs(OUT_DIR, exist_ok=True)

ALGOS = ["AZP", "RegionKModels", "GWR+SKATER", "SKATER-reg", "SHAP-based", "GeoEvolve"]
ALGO_LABELS = {
    "AZP": "AZP", "RegionKModels": "RKM",
    "GWR+SKATER": "GWR+SKATER", "SKATER-reg": "SKATER-reg",
    "SHAP-based": "SHAP-based", "GeoEvolve": "GeoEvolve (ours)",
}
ALGO_COLORS = {
    "AZP":          "#003f5c",
    "RegionKModels": "#374c80",
    "GWR+SKATER":    "#7a5195",
    "SKATER-reg":    "#bc5090",
    "SHAP-based":    "#ef5675",
    "GeoEvolve":     "#a8d5ba",
}

DOMAINS_ORDER = ["Climate", "Health", "Hydro", "Politics"]
DOMAIN_LABELS = {"Climate": "Climate", "Health": "Health",
                 "Hydro": "Hydro.", "Politics": "Politics"}

DS_PRETTY = {
    "US_Climate_ERA5_CLIMATE": "ERA5",
    "US_Health_ARTHRITIS": "Arthr.", "US_Health_BPHIGH": "BPHigh",
    "US_Health_CANCER": "Cancer", "US_Health_CASTHMA": "Asthma",
    "US_Health_DEPRESSION": "Depr.", "US_Health_DIABETES": "Diab.",
    "US_Health_OBESITY": "Obesity", "US_Health_STROKE": "Stroke",
    "US_Hydro_CAMELS": "CAMELS", "US_Politics_Voting": "Voting",
}

METRICS = [
    ("Avg_R2", "Avg R²", True),
    ("Avg_RMSE", "Avg RMSE", False),
    ("SSR_Reduction", "SSR Reduction", True),
    ("Variance_Ratio", "Variance Ratio", True),
]


def get_domain(ds):
    if "Climate" in ds: return "Climate"
    if "Health" in ds: return "Health"
    if "Hydro" in ds: return "Hydro"
    if "Politics" in ds: return "Politics"
    return "Other"


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    data = OrderedDict((d, OrderedDict()) for d in DOMAINS_ORDER)
    for r in rows:
        ds = r["Dataset"]
        dom = get_domain(ds)
        if dom not in data:
            data[dom] = OrderedDict()
        if ds not in data[dom]:
            data[dom][ds] = {}
        data[dom][ds][r["Algorithm"]] = r

    ds_list, ds_labels, domain_boundaries = [], [], []
    idx = 0
    for dom in DOMAINS_ORDER:
        if dom not in data or not data[dom]:
            continue
        start = idx
        for ds in data[dom]:
            ds_list.append(ds)
            ds_labels.append(DS_PRETTY.get(ds, ds))
            idx += 1
        domain_boundaries.append((start, idx - 1, DOMAIN_LABELS[dom]))

    n_ds = len(ds_list)
    n_algo = len(ALGOS)
    bar_width = 0.13
    gap = 0.6
    x_positions, current_x, prev = [], 0, -1
    for dom in DOMAINS_ORDER:
        if dom not in data or not data[dom]:
            continue
        for i, ds in enumerate(data[dom]):
            if prev >= 0 and i == 0:
                current_x += gap
            x_positions.append(current_x)
            current_x += 1
        prev = current_x
    x_positions = np.array(x_positions)

    metric_all = {}
    for mk, _, _ in METRICS:
        v = []
        for ds in ds_list:
            dom = get_domain(ds)
            for a in ALGOS:
                if a in data[dom][ds]:
                    try:
                        v.append(float(data[dom][ds][a][mk]))
                    except (ValueError, TypeError):
                        pass
        metric_all[mk] = v

    def get_ylim(vals, higher_better):
        flat = [v for v in vals if v > 0]
        if not flat:
            return 0, 1
        vmin, vmax = min(flat), max(flat)
        span = vmax - vmin
        margin = span * 0.15
        if higher_better:
            lo = max(0, vmin - margin - span * 0.3)
            hi = vmax + margin
            lo = np.floor(lo * 20) / 20
            hi = min(1.0, np.ceil(hi * 20) / 20)
        else:
            lo = max(0, vmin - margin)
            hi = vmax + margin + span * 0.3
            lo = np.floor(lo * 20) / 20
            hi = np.ceil(hi * 20) / 20
        return lo, hi

    best_per_ds = {}
    for mk, _, hb in METRICS:
        for di, ds in enumerate(ds_list):
            dom = get_domain(ds)
            best_ai, best_val = 0, None
            for ai, a in enumerate(ALGOS):
                if a in data[dom][ds]:
                    try:
                        v = float(data[dom][ds][a][mk])
                    except (ValueError, TypeError):
                        continue
                    if best_val is None or (hb and v > best_val) or (not hb and v < best_val):
                        best_val, best_ai = v, ai
            best_per_ds[(mk, di)] = best_ai

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.2))
    axes = axes.flatten()

    for mi, (mk, mlabel, hb) in enumerate(METRICS):
        ax = axes[mi]
        use_log = mk == "Avg_RMSE"
        bar_containers = {}
        for ai, a in enumerate(ALGOS):
            vals = []
            for ds in ds_list:
                dom = get_domain(ds)
                try:
                    vals.append(float(data[dom][ds][a][mk]) if a in data[dom][ds] else 0)
                except (ValueError, TypeError):
                    vals.append(0)
            vals = np.array(vals)
            offset = (ai - (n_algo - 1) / 2) * bar_width
            bars = ax.bar(x_positions + offset, vals, bar_width,
                          label=ALGO_LABELS[a] if mi == 0 else "",
                          color=ALGO_COLORS[a], edgecolor="white", linewidth=0.4)
            bar_containers[ai] = bars

        for di in range(n_ds):
            best_ai = best_per_ds[(mk, di)]
            br = bar_containers[best_ai][di]
            cx = br.get_x() + br.get_width() / 2
            cy = br.get_height()
            star_y = cy * 1.1 if use_log else cy + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
            ax.plot(cx, star_y, marker="*", color="#e63946",
                    markersize=5, markeredgewidth=0)

        for (s, e, dlabel) in domain_boundaries:
            if s > 0:
                sep = (x_positions[s] + x_positions[s - 1]) / 2
                ax.axvline(sep, color="gray", lw=0.7, ls="--", alpha=0.5)
            mid = (x_positions[s] + x_positions[e]) / 2
            ax.text(mid, 1.02, dlabel, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7,
                    fontweight="bold", fontstyle="italic", color="#333")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ds_labels, rotation=35, ha="right")
        ax.set_ylabel(mlabel, fontweight="bold")
        # Domain headers are drawn at y=1.02 of the axes; place title above them
        ax.set_title(mlabel, fontweight="bold", pad=20)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if use_log:
            ax.set_yscale("log")
            ax.set_ylabel(mlabel + " (log)", fontweight="bold")
        else:
            lo, hi = get_ylim(metric_all[mk], hb)
            ax.set_ylim(lo, hi)

    handles = [mpatches.Patch(facecolor=ALGO_COLORS[a], edgecolor="white",
                              label=ALGO_LABELS[a]) for a in ALGOS]
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                              markerfacecolor="#e63946", markersize=7, label="Best"))
    fig.legend(handles=handles, loc="upper center", ncol=len(ALGOS) + 1,
               frameon=True, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.0, w_pad=1.5)
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"baseline_comparison_bars.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
