#!/usr/bin/env python3
"""
Generate grouped bar charts comparing algorithm performance across datasets,
grouped by domain (Climate, Health, Hydrology, Politics).
- Truncated y-axes to amplify differences
- Star marker on the best algorithm per dataset
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

# ── paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE, "results_v2", "algorithm_comparison.csv")
OUT_DIR   = os.path.join(BASE, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── config ──
ALGOS = ['Initial', 'OpenEvolve', 'GeoEvolve']
ALGO_COLORS = {
    'Initial':    '#003f5c',   # dark navy
    'OpenEvolve': '#58a4b0',   # sky / teal blue
    'GeoEvolve':  '#a8d5ba',   # mint green
}
ALGO_HATCHES = {
    'Initial':    '',
    'OpenEvolve': '',
    'GeoEvolve':  '',
}

DOMAINS_ORDER = ['Climate', 'Health', 'Hydro', 'Politics']
DOMAIN_LABELS = {
    'Climate': 'Climate', 'Health': 'Health',
    'Hydro': 'Hydro.', 'Politics': 'Politics',
}

DS_PRETTY = {
    'US_Climate_ERA5_CLIMATE': 'ERA5',
    'US_Health_ARTHRITIS':     'Arthr.',
    'US_Health_BPHIGH':        'BPHigh',
    'US_Health_CANCER':        'Cancer',
    'US_Health_CASTHMA':       'Asthma',
    'US_Health_DEPRESSION':    'Depr.',
    'US_Health_DIABETES':      'Diab.',
    'US_Health_OBESITY':       'Obesity',
    'US_Health_STROKE':        'Stroke',
    'US_Hydro_CAMELS':         'CAMELS',
    'US_Politics_Voting':      'Voting',
}

METRICS = [
    ('Avg_R2',         'Avg R²',           True),
    ('Avg_RMSE',       'Avg RMSE',         False),
    ('SSR_Reduction',  'SSR Reduction',    True),
    ('Variance_Ratio', 'Variance Ratio',   True),
]

def get_domain(ds):
    if 'Climate'  in ds: return 'Climate'
    if 'Health'   in ds: return 'Health'
    if 'Hydro'    in ds: return 'Hydro'
    if 'Politics' in ds: return 'Politics'
    return 'Other'

# ── read data ──
rows = []
with open(CSV_PATH, 'r') as f:
    for r in csv.DictReader(f):
        rows.append(r)

# group: domain -> dataset -> algo -> row
data = OrderedDict()
for dom in DOMAINS_ORDER:
    data[dom] = OrderedDict()
for r in rows:
    ds  = r['Dataset']
    dom = get_domain(ds)
    if dom not in data:
        data[dom] = OrderedDict()
    if ds not in data[dom]:
        data[dom][ds] = {}
    data[dom][ds][r['Algorithm']] = r

# build ordered dataset list with domain boundaries
ds_list = []
ds_labels = []
domain_boundaries = []
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

n_ds   = len(ds_list)
n_algo = len(ALGOS)
bar_width = 0.24
gap_between_groups = 0.65

# compute x positions with domain gaps
x_positions = []
current_x = 0
prev_domain_end = -1
for dom in DOMAINS_ORDER:
    if dom not in data or not data[dom]:
        continue
    for i, ds in enumerate(data[dom]):
        if prev_domain_end >= 0 and i == 0:
            current_x += gap_between_groups
        x_positions.append(current_x)
        current_x += 1
    prev_domain_end = current_x

x_positions = np.array(x_positions)

# ── helper: compute smart y-axis limits (truncated) ──
def get_ylim_truncated(all_vals, higher_better):
    """Truncate y-axis to zoom into the range where differences are visible."""
    flat = [v for v in all_vals if v > 0]
    vmin, vmax = min(flat), max(flat)
    span = vmax - vmin
    margin = span * 0.15
    if higher_better:
        # show from slightly below min to slightly above max
        lo = max(0, vmin - margin - span * 0.3)
        hi = vmax + margin
        # round lo down for cleaner axis
        lo = np.floor(lo * 20) / 20  # snap to 0.05
        hi = min(1.0, np.ceil(hi * 20) / 20)
    else:
        lo = max(0, vmin - margin)
        hi = vmax + margin + span * 0.3
        lo = np.floor(lo * 20) / 20
        hi = np.ceil(hi * 20) / 20
    return lo, hi

# ── collect all values per metric for axis scaling ──
metric_all_vals = {}
for metric_key, _, _ in METRICS:
    vals = []
    for ds in ds_list:
        dom = get_domain(ds)
        for algo in ALGOS:
            if algo in data[dom][ds]:
                vals.append(float(data[dom][ds][algo][metric_key]))
    metric_all_vals[metric_key] = vals

# ── find best algo per dataset per metric ──
best_per_ds = {}   # (metric_key, ds_idx) -> algo_idx
for metric_key, _, higher_better in METRICS:
    for di, ds in enumerate(ds_list):
        dom = get_domain(ds)
        best_ai = 0
        best_val = None
        for ai, algo in enumerate(ALGOS):
            if algo in data[dom][ds]:
                v = float(data[dom][ds][algo][metric_key])
                if best_val is None:
                    best_val = v
                    best_ai = ai
                elif (higher_better and v > best_val) or (not higher_better and v < best_val):
                    best_val = v
                    best_ai = ai
        best_per_ds[(metric_key, di)] = best_ai

# ── plot  (compact for LaTeX embedding, ~\textwidth) ──
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7.5,
    'axes.titlepad': 10,
    'axes.labelpad': 3,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
})
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))  # ~\textwidth for two-column
axes = axes.flatten()

for mi, (metric_key, metric_label, higher_better) in enumerate(METRICS):
    ax = axes[mi]

    # special handling for RMSE (log scale, no truncation)
    use_log = (metric_key == 'Avg_RMSE')

    bar_containers = {}  # algo -> list of bar rects
    for ai, algo in enumerate(ALGOS):
        vals = []
        for ds in ds_list:
            dom = get_domain(ds)
            if algo in data[dom][ds]:
                vals.append(float(data[dom][ds][algo][metric_key]))
            else:
                vals.append(0)
        vals = np.array(vals)

        offset = (ai - (n_algo - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset, vals, bar_width,
            label=algo if mi == 0 else '',
            color=ALGO_COLORS[algo],
            hatch=ALGO_HATCHES[algo],
            edgecolor='white',
            linewidth=0.5,
            zorder=3,
        )
        bar_containers[ai] = bars

    # add star on best bar per dataset
    for di in range(n_ds):
        best_ai = best_per_ds[(metric_key, di)]
        bar_rect = bar_containers[best_ai][di]
        cx = bar_rect.get_x() + bar_rect.get_width() / 2
        cy = bar_rect.get_height()
        if use_log:
            star_y = cy * 1.15
        else:
            star_y = cy + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
        ax.plot(cx, star_y, marker='*', color='#e63946', markersize=5,
                zorder=5, markeredgewidth=0)

    # domain separator lines & labels
    for (start, end, dom_label) in domain_boundaries:
        if start > 0:
            sep_x = (x_positions[start] + x_positions[start - 1]) / 2
            ax.axvline(sep_x, color='gray', linewidth=0.8, linestyle='--',
                       alpha=0.5, zorder=1)
        mid_x = (x_positions[start] + x_positions[end]) / 2
        ax.text(mid_x, 1.02, dom_label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                fontstyle='italic', color='#333333')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ds_labels, rotation=35, ha='right')
    ax.set_ylabel(metric_label, fontweight='bold')
    ax.set_title(metric_label, fontweight='bold')

    # grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # y-axis: truncated for ratio metrics, log for RMSE
    if use_log:
        ax.set_yscale('log')
        ax.set_ylabel(metric_label + ' (log)', fontweight='bold')
        ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.3, zorder=0)
    else:
        lo, hi = get_ylim_truncated(metric_all_vals[metric_key], higher_better)
        ax.set_ylim(lo, hi)
        # broken-axis indicator (slanted lines at bottom)
        if lo > 0:
            d = 0.01
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (-d + 0.012, +d + 0.012), **kwargs)

# shared legend + star explanation
handles = [mpatches.Patch(facecolor=ALGO_COLORS[a], edgecolor='white', label=a)
           for a in ALGOS]
star_handle = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#e63946',
                         markersize=7, label='Best', markeredgewidth=0)
handles.append(star_handle)

fig.legend(handles=handles, loc='upper center', ncol=4,
           frameon=True, fancybox=False, shadow=False,
           bbox_to_anchor=(0.5, 1.005), edgecolor='#cccccc',
           handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=1.8, w_pad=1.2)

# save
for ext in ['pdf', 'png']:
    out_path = os.path.join(OUT_DIR, f'algorithm_comparison_bars.{ext}')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")

plt.close()
print("Done.")
