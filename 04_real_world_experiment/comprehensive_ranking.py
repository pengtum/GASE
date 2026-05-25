#!/usr/bin/env python3
"""
Comprehensive Algorithm Performance Ranking
============================================
Multi-dimensional ranking across 6 metrics, 13 datasets, 4 algorithms.
Generates:
  1. Per-metric ranking tables
  2. Composite score with configurable weights
  3. Win-count analysis (how often each algo is #1 per dataset)
  4. Pareto frontier (quality vs speed)
  5. Robustness analysis (std of rankings across datasets)
  6. Domain-level breakdown (Climate, Health, Hydro, Politics)
  7. Publication-ready summary figure (PDF + PNG)
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

# ── paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, 'results_v2', 'algorithm_comparison.csv')
OUT_DIR  = os.path.join(BASE, 'results_v2', 'comprehensive_ranking')
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

ALGOS = df['Algorithm'].unique().tolist()
DATASETS = df['Dataset'].unique().tolist()
N_ALGO = len(ALGOS)
N_DS   = len(DATASETS)
print(f"Algorithms: {ALGOS}")
print(f"Datasets ({N_DS}): {DATASETS}")

# ── domain mapping ─────────────────────────────────────────────────────
def get_domain(ds):
    if 'Climate' in ds: return 'Climate'
    if 'Health'  in ds: return 'Health'
    if 'Hydro'   in ds: return 'Hydro'
    if 'Politics' in ds: return 'Politics'
    return 'Other'

df['Domain'] = df['Dataset'].apply(get_domain)

# ══════════════════════════════════════════════════════════════════════
#  1. PER-METRIC AVERAGES & RANKINGS
# ══════════════════════════════════════════════════════════════════════
# Metrics: higher is better → SSR_Reduction, Avg_R2, Variance_Ratio
# Metrics: lower  is better → Avg_RMSE, Runtime_s, SSR_Regime (actual)

METRICS = {
    'SSR_Reduction':  {'higher_better': True,  'label': 'SSR Reduction',     'fmt': '.4f', 'weight': 0.30},
    'Avg_R2':         {'higher_better': True,  'label': 'Avg R²',           'fmt': '.4f', 'weight': 0.20},
    'Variance_Ratio': {'higher_better': True,  'label': 'Variance Ratio',   'fmt': '.4f', 'weight': 0.15},
    'Avg_RMSE':       {'higher_better': False, 'label': 'Avg RMSE',         'fmt': '.4f', 'weight': 0.10},
    'Runtime_s':      {'higher_better': False, 'label': 'Runtime (s)',       'fmt': '.1f', 'weight': 0.10},
    'SSR_Regime':     {'higher_better': False, 'label': 'Regime SSR (abs)', 'fmt': '.1f', 'weight': 0.15},
}

print("\n" + "=" * 90)
print("  1. PER-METRIC AVERAGES (across all datasets)")
print("=" * 90)

avg_table = df.groupby('Algorithm')[list(METRICS.keys())].mean()
# Rank for each metric
rank_table = pd.DataFrame(index=avg_table.index)
for m, info in METRICS.items():
    rank_table[m + '_rank'] = avg_table[m].rank(ascending=not info['higher_better'])

print(f"\n{'Algorithm':<22}", end='')
for m, info in METRICS.items():
    print(f"  {info['label']:>16}", end='')
print()
print("-" * 120)
for algo in ALGOS:
    print(f"{algo:<22}", end='')
    for m, info in METRICS.items():
        val = avg_table.loc[algo, m]
        rnk = int(rank_table.loc[algo, m + '_rank'])
        fmt = info['fmt']
        print(f"  {val:>12{fmt}} (#{rnk})", end='')
    print()

# ══════════════════════════════════════════════════════════════════════
#  2. COMPOSITE SCORE (weighted normalised rank)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  2. COMPOSITE SCORE (Weighted Rank Aggregation)")
print("=" * 90)

# For each dataset × metric, rank the 4 algorithms (1=best, 4=worst)
per_ds_ranks = []
for ds in DATASETS:
    sub = df[df['Dataset'] == ds].set_index('Algorithm')
    row = {'Dataset': ds}
    for m, info in METRICS.items():
        ranks = sub[m].rank(ascending=not info['higher_better'])
        for algo in ALGOS:
            row[f'{algo}_{m}_rank'] = ranks[algo] if algo in ranks.index else np.nan
    per_ds_ranks.append(row)
ranks_df = pd.DataFrame(per_ds_ranks)

# Compute weighted composite score per algorithm per dataset
# Lower composite = better
composite_per_ds = pd.DataFrame({'Dataset': DATASETS})
for algo in ALGOS:
    scores = []
    for _, row in ranks_df.iterrows():
        s = 0.0
        for m, info in METRICS.items():
            s += row[f'{algo}_{m}_rank'] * info['weight']
        scores.append(s)
    composite_per_ds[algo] = scores

# Average composite across datasets
composite_avg = composite_per_ds[ALGOS].mean()
composite_std = composite_per_ds[ALGOS].std()
composite_rank = composite_avg.rank()

print(f"\nWeights: ", end='')
for m, info in METRICS.items():
    print(f"{info['label']}={info['weight']:.0%}  ", end='')
print(f"\n\n{'Algorithm':<22} {'Avg Composite':>14} {'Std':>8} {'Rank':>6}")
print("-" * 55)
for algo in composite_avg.sort_values().index:
    print(f"{algo:<22} {composite_avg[algo]:>14.3f} {composite_std[algo]:>8.3f} {int(composite_rank[algo]):>6}")

# ══════════════════════════════════════════════════════════════════════
#  3. WIN-COUNT ANALYSIS (per metric)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  3. WIN-COUNT ANALYSIS (# datasets where algorithm is #1)")
print("=" * 90)

win_matrix = pd.DataFrame(0, index=ALGOS, columns=list(METRICS.keys()) + ['Overall_Best'])
for ds in DATASETS:
    sub = df[df['Dataset'] == ds].set_index('Algorithm')
    for m, info in METRICS.items():
        if info['higher_better']:
            winner = sub[m].idxmax()
        else:
            winner = sub[m].idxmin()
        win_matrix.loc[winner, m] += 1
    # Overall best = lowest composite
    row = composite_per_ds[composite_per_ds['Dataset'] == ds]
    best_algo = row[ALGOS].values[0].argmin()
    win_matrix.loc[ALGOS[best_algo], 'Overall_Best'] += 1

print(f"\n{'Algorithm':<22}", end='')
for m, info in METRICS.items():
    print(f"  {info['label']:>16}", end='')
print(f"  {'Overall Best':>14}")
print("-" * 135)
for algo in ALGOS:
    print(f"{algo:<22}", end='')
    for m in list(METRICS.keys()) + ['Overall_Best']:
        print(f"  {win_matrix.loc[algo, m]:>16}", end='')
    print()

# ══════════════════════════════════════════════════════════════════════
#  4. DOMAIN-LEVEL BREAKDOWN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  4. DOMAIN-LEVEL BREAKDOWN")
print("=" * 90)

for domain in ['Climate', 'Health', 'Hydro', 'Politics']:
    sub = df[df['Domain'] == domain]
    if sub.empty:
        continue
    n_ds = sub['Dataset'].nunique()
    print(f"\n  ── {domain} ({n_ds} datasets) ──")
    domain_avg = sub.groupby('Algorithm')[['SSR_Reduction', 'Avg_R2', 'Variance_Ratio', 'Runtime_s']].mean()
    domain_avg['SSR_Red_Rank'] = domain_avg['SSR_Reduction'].rank(ascending=False)
    domain_avg = domain_avg.sort_values('SSR_Red_Rank')
    print(f"  {'Algorithm':<22} {'SSR_Red':>9} {'Avg_R2':>9} {'VarRatio':>9} {'Runtime':>9} {'Rank':>6}")
    print(f"  {'-'*70}")
    for algo, row in domain_avg.iterrows():
        print(f"  {algo:<22} {row['SSR_Reduction']:>9.4f} {row['Avg_R2']:>9.4f} "
              f"{row['Variance_Ratio']:>9.4f} {row['Runtime_s']:>9.1f} #{int(row['SSR_Red_Rank'])}")

# ══════════════════════════════════════════════════════════════════════
#  5. ROBUSTNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  5. ROBUSTNESS ANALYSIS (Consistency across datasets)")
print("=" * 90)

# For each algo, compute std of SSR_Reduction across datasets
# + coefficient of variation, min, max
robust = df.groupby('Algorithm')['SSR_Reduction'].agg(['mean', 'std', 'min', 'max'])
robust['CV'] = robust['std'] / robust['mean']  # coefficient of variation
robust['Range'] = robust['max'] - robust['min']
robust = robust.sort_values('CV')  # lower CV = more consistent

print(f"\n{'Algorithm':<22} {'Mean':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
print("-" * 75)
for algo, row in robust.iterrows():
    print(f"{algo:<22} {row['mean']:>8.4f} {row['std']:>8.4f} {row['CV']:>8.4f} "
          f"{row['min']:>8.4f} {row['max']:>8.4f} {row['Range']:>8.4f}")

# ══════════════════════════════════════════════════════════════════════
#  6. PARETO FRONTIER (Quality vs Speed)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  6. PARETO ANALYSIS (SSR_Reduction vs Runtime)")
print("=" * 90)

pareto_df = df.groupby('Algorithm')[['SSR_Reduction', 'Runtime_s']].mean()
pareto_df['Efficiency'] = pareto_df['SSR_Reduction'] / np.log1p(pareto_df['Runtime_s'])
pareto_df = pareto_df.sort_values('Efficiency', ascending=False)

print(f"\n{'Algorithm':<22} {'Avg SSR_Red':>12} {'Avg Runtime':>12} {'Efficiency':>12}")
print(f"{'':22} {'':>12} {'(seconds)':>12} {'(SSR/log(t))':>12}")
print("-" * 62)
for algo, row in pareto_df.iterrows():
    print(f"{algo:<22} {row['SSR_Reduction']:>12.4f} {row['Runtime_s']:>12.1f} {row['Efficiency']:>12.4f}")

# ══════════════════════════════════════════════════════════════════════
#  7. NORMALISED SCORE TABLE (0-100 scale)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  7. NORMALISED SCORE (0-100, per metric, across all datasets)")
print("=" * 90)

# Min-max normalise each metric so best=100, worst=0
norm_scores = pd.DataFrame(index=ALGOS)
for m, info in METRICS.items():
    vals = avg_table[m]
    if info['higher_better']:
        norm = (vals - vals.min()) / (vals.max() - vals.min()) * 100
    else:
        norm = (vals.max() - vals) / (vals.max() - vals.min()) * 100
    norm_scores[info['label']] = norm

# Weighted final
weights = [info['weight'] for info in METRICS.values()]
norm_scores['★ Weighted Total'] = (norm_scores.iloc[:, :len(METRICS)].values * weights).sum(axis=1)
norm_scores = norm_scores.sort_values('★ Weighted Total', ascending=False)

print(f"\n{'Algorithm':<22}", end='')
for col in norm_scores.columns:
    print(f"  {col:>16}", end='')
print()
print("-" * (22 + 18 * len(norm_scores.columns)))
for algo, row in norm_scores.iterrows():
    print(f"{algo:<22}", end='')
    for v in row:
        print(f"  {v:>16.1f}", end='')
    print()

# ══════════════════════════════════════════════════════════════════════
#  8. STATISTICAL SIGNIFICANCE (Paired comparison)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  8. HEAD-TO-HEAD COMPARISON (SSR Reduction per dataset)")
print("=" * 90)

# Build pivot
pivot = df.pivot(index='Dataset', columns='Algorithm', values='SSR_Reduction')
for i, a1 in enumerate(ALGOS):
    for a2 in ALGOS[i+1:]:
        diff = pivot[a1] - pivot[a2]
        a1_wins = (diff > 0).sum()
        a2_wins = (diff < 0).sum()
        ties = (diff == 0).sum()
        avg_diff = diff.mean()
        print(f"  {a1:<22} vs {a2:<22}: {a1_wins}-{a2_wins}-{ties}  (avg diff = {avg_diff:+.4f})")


# ══════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  GENERATING COMPREHENSIVE RANKING FIGURE ...")
print("=" * 90)

ALGO_COLORS = {
    'Initial': '#1f77b4',
    'OpenEvolve': '#ff7f0e',
    'GeoEvolve (static)': '#2ca02c',
    'GeoEvolve (dynamic)': '#d62728',
}

fig = plt.figure(figsize=(24, 20))
gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.30,
                       left=0.06, right=0.97, top=0.94, bottom=0.04)

fig.suptitle('Comprehensive Algorithm Performance Ranking\n(4 Algorithms × 13 Datasets × 6 Metrics)',
             fontsize=16, fontweight='bold', y=0.98)

# ── (A) Radar chart: normalised scores ──
ax_radar = fig.add_subplot(gs[0, 0], polar=True)
metric_labels = [info['label'] for info in METRICS.values()]
N_met = len(metric_labels)
angles = np.linspace(0, 2 * np.pi, N_met, endpoint=False).tolist()
angles += angles[:1]

for algo in ALGOS:
    vals = norm_scores.loc[algo, [info['label'] for info in METRICS.values()]].values.tolist()
    vals += vals[:1]
    color = ALGO_COLORS.get(algo, 'gray')
    ax_radar.plot(angles, vals, 'o-', linewidth=2, label=algo, color=color)
    ax_radar.fill(angles, vals, alpha=0.1, color=color)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(metric_labels, fontsize=8)
ax_radar.set_ylim(0, 110)
ax_radar.set_title('(A) Normalised Score Radar', fontsize=11, fontweight='bold', pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)

# ── (B) Composite score bar chart ──
ax_comp = fig.add_subplot(gs[0, 1])
sorted_algos = composite_avg.sort_values().index.tolist()
bars = ax_comp.barh(range(len(sorted_algos)), [composite_avg[a] for a in sorted_algos],
                     color=[ALGO_COLORS.get(a, 'gray') for a in sorted_algos], edgecolor='black')
ax_comp.set_yticks(range(len(sorted_algos)))
ax_comp.set_yticklabels(sorted_algos, fontsize=10)
ax_comp.set_xlabel('Weighted Composite Rank Score (lower = better)', fontsize=9)
ax_comp.set_title('(B) Composite Score Ranking', fontsize=11, fontweight='bold')
for i, (a, bar) in enumerate(zip(sorted_algos, bars)):
    ax_comp.text(bar.get_width() + 0.02, i, f'{composite_avg[a]:.3f} ± {composite_std[a]:.3f}',
                 va='center', fontsize=9)
ax_comp.set_xlim(0, composite_avg.max() * 1.25)

# ── (C) Win count stacked bar ──
ax_win = fig.add_subplot(gs[0, 2])
metric_names = [info['label'] for info in METRICS.values()]
bottom = np.zeros(N_ALGO)
cmap = plt.cm.Set2
for j, m in enumerate(METRICS.keys()):
    vals = [win_matrix.loc[algo, m] for algo in ALGOS]
    ax_win.bar(range(N_ALGO), vals, bottom=bottom, label=METRICS[m]['label'],
               color=cmap(j / len(METRICS)), edgecolor='white')
    bottom += np.array(vals)
ax_win.set_xticks(range(N_ALGO))
ax_win.set_xticklabels([a.replace(' ', '\n') for a in ALGOS], fontsize=9)
ax_win.set_ylabel('# Datasets Won', fontsize=9)
ax_win.set_title('(C) Win Count by Metric', fontsize=11, fontweight='bold')
ax_win.legend(fontsize=7, ncol=2, loc='upper right')

# ── (D) SSR Reduction boxplot per algorithm ──
ax_box = fig.add_subplot(gs[1, 0])
data_for_box = [df[df['Algorithm'] == algo]['SSR_Reduction'].values for algo in ALGOS]
bp = ax_box.boxplot(data_for_box, patch_artist=True, labels=[a.replace(' ', '\n') for a in ALGOS])
for patch, algo in zip(bp['boxes'], ALGOS):
    patch.set_facecolor(ALGO_COLORS.get(algo, 'gray'))
    patch.set_alpha(0.7)
ax_box.set_ylabel('SSR Reduction', fontsize=10)
ax_box.set_title('(D) SSR Reduction Distribution', fontsize=11, fontweight='bold')
ax_box.grid(axis='y', alpha=0.3)

# ── (E) Pareto frontier: SSR_Reduction vs Runtime ──
ax_pareto = fig.add_subplot(gs[1, 1])
for algo in ALGOS:
    sub = df[df['Algorithm'] == algo]
    color = ALGO_COLORS.get(algo, 'gray')
    ax_pareto.scatter(sub['Runtime_s'], sub['SSR_Reduction'], c=color,
                      s=50, alpha=0.5, edgecolors='black', linewidth=0.5)
    # Average point (large)
    ax_pareto.scatter(sub['Runtime_s'].mean(), sub['SSR_Reduction'].mean(),
                      c=color, s=200, marker='*', edgecolors='black', linewidth=1.2,
                      zorder=10, label=algo)
ax_pareto.set_xlabel('Runtime (seconds)', fontsize=10)
ax_pareto.set_ylabel('SSR Reduction', fontsize=10)
ax_pareto.set_title('(E) Pareto: Quality vs Speed', fontsize=11, fontweight='bold')
ax_pareto.legend(fontsize=8)
ax_pareto.grid(alpha=0.3)

# ── (F) Domain-level heatmap ──
ax_domain = fig.add_subplot(gs[1, 2])
domains = ['Climate', 'Health', 'Hydro', 'Politics']
domain_matrix = np.zeros((len(domains), N_ALGO))
for i, dom in enumerate(domains):
    sub = df[df['Domain'] == dom]
    for j, algo in enumerate(ALGOS):
        vals = sub[sub['Algorithm'] == algo]['SSR_Reduction']
        domain_matrix[i, j] = vals.mean() if len(vals) > 0 else 0

im = ax_domain.imshow(domain_matrix, cmap='YlGn', aspect='auto', vmin=0.45, vmax=0.75)
ax_domain.set_xticks(range(N_ALGO))
ax_domain.set_xticklabels([a.replace(' ', '\n') for a in ALGOS], fontsize=9)
ax_domain.set_yticks(range(len(domains)))
ax_domain.set_yticklabels(domains, fontsize=10)
for i in range(len(domains)):
    for j in range(N_ALGO):
        ax_domain.text(j, i, f'{domain_matrix[i, j]:.4f}', ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='white' if domain_matrix[i, j] > 0.65 else 'black')
ax_domain.set_title('(F) SSR Reduction by Domain', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax_domain, shrink=0.8)

# ── (G) Normalised score grouped bar ──
ax_norm = fig.add_subplot(gs[2, 0:2])
x = np.arange(len(metric_labels))
width = 0.2
for i, algo in enumerate(ALGOS):
    vals = norm_scores.loc[algo, [info['label'] for info in METRICS.values()]].values
    color = ALGO_COLORS.get(algo, 'gray')
    bars = ax_norm.bar(x + i * width - width * 1.5, vals, width, label=algo,
                       color=color, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 5:
            ax_norm.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{v:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)
ax_norm.set_xticks(x)
ax_norm.set_xticklabels(metric_labels, fontsize=10)
ax_norm.set_ylabel('Normalised Score (0–100)', fontsize=10)
ax_norm.set_title('(G) Per-Metric Normalised Scores', fontsize=11, fontweight='bold')
ax_norm.legend(fontsize=9)
ax_norm.set_ylim(0, 120)
ax_norm.grid(axis='y', alpha=0.3)

# ── (H) Final ranking summary table ──
ax_table = fig.add_subplot(gs[2, 2])
ax_table.axis('off')

# Build table data
table_data = []
rank_order = norm_scores['★ Weighted Total'].sort_values(ascending=False)
for rank_i, (algo, total) in enumerate(rank_order.items(), 1):
    ssr_red = avg_table.loc[algo, 'SSR_Reduction']
    avg_r2 = avg_table.loc[algo, 'Avg_R2']
    runtime = avg_table.loc[algo, 'Runtime_s']
    wins = win_matrix.loc[algo, 'Overall_Best']
    table_data.append([f'#{rank_i}', algo, f'{total:.1f}', f'{ssr_red:.4f}',
                       f'{avg_r2:.4f}', f'{runtime:.1f}s', str(wins)])

col_labels = ['Rank', 'Algorithm', 'Score', 'SSR_Red', 'Avg R²', 'Runtime', 'Wins']
table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center',
                        cellLoc='center', colColours=['#f0f0f0'] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# Color first column by rank
rank_colors = ['#ffd700', '#c0c0c0', '#cd7f32', '#e0e0e0']  # gold, silver, bronze, grey
for i in range(len(table_data)):
    table[i + 1, 0].set_facecolor(rank_colors[i] if i < 4 else '#ffffff')
    table[i + 1, 0].set_text_props(fontweight='bold', fontsize=12)
    # Also color algorithm cell
    algo_name = table_data[i][1]
    base_color = ALGO_COLORS.get(algo_name, '#808080')
    table[i + 1, 1].set_facecolor(base_color + '40')

ax_table.set_title('(H) Final Ranking Summary', fontsize=11, fontweight='bold', pad=15)

# Save
for ext in ['pdf', 'png']:
    path = os.path.join(OUT_DIR, f'comprehensive_ranking.{ext}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
plt.close()

# ══════════════════════════════════════════════════════════════════════
#  ADDITIONAL FIGURE: Heatmap of ranks per dataset
# ══════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(20, 10))

# Left: SSR Reduction heatmap (value)
pivot_ssr = df.pivot(index='Dataset', columns='Algorithm', values='SSR_Reduction')
pivot_ssr = pivot_ssr[ALGOS]  # consistent order
pivot_ssr.index = [d.replace('US_', '').replace('_', ' ') for d in pivot_ssr.index]

im1 = axes[0].imshow(pivot_ssr.values, cmap='YlGn', aspect='auto')
axes[0].set_xticks(range(N_ALGO))
axes[0].set_xticklabels([a.replace(' ', '\n') for a in ALGOS], fontsize=9)
axes[0].set_yticks(range(len(pivot_ssr)))
axes[0].set_yticklabels(pivot_ssr.index, fontsize=9)
for i in range(len(pivot_ssr)):
    vals_row = pivot_ssr.values[i]
    best_j = np.argmax(vals_row)
    for j in range(N_ALGO):
        weight = 'bold' if j == best_j else 'normal'
        axes[0].text(j, i, f'{vals_row[j]:.4f}', ha='center', va='center',
                     fontsize=8, fontweight=weight,
                     color='white' if vals_row[j] > 0.65 else 'black')
axes[0].set_title('SSR Reduction by Dataset × Algorithm\n(bold = best per dataset)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Right: Composite rank heatmap
# Build per-dataset composite rank
rank_per_ds = pd.DataFrame(index=[d.replace('US_', '').replace('_', ' ') for d in DATASETS])
for algo in ALGOS:
    rank_per_ds[algo] = composite_per_ds[algo].values
# Rank per row (1=best)
rank_disp = rank_per_ds.rank(axis=1)

im2 = axes[1].imshow(rank_disp.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=4)
axes[1].set_xticks(range(N_ALGO))
axes[1].set_xticklabels([a.replace(' ', '\n') for a in ALGOS], fontsize=9)
axes[1].set_yticks(range(len(rank_disp)))
axes[1].set_yticklabels(rank_disp.index, fontsize=9)
for i in range(len(rank_disp)):
    for j in range(N_ALGO):
        v = rank_disp.values[i, j]
        axes[1].text(j, i, f'#{int(v)}', ha='center', va='center',
                     fontsize=10, fontweight='bold' if v == 1 else 'normal',
                     color='white' if v >= 3 else 'black')
axes[1].set_title('Composite Rank by Dataset × Algorithm\n(#1=best, weighted across all metrics)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1], shrink=0.8)

fig2.suptitle('Per-Dataset Performance Heatmaps', fontsize=14, fontweight='bold', y=1.01)
fig2.tight_layout()
for ext in ['pdf', 'png']:
    path = os.path.join(OUT_DIR, f'per_dataset_heatmaps.{ext}')
    fig2.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
#  EXPORT CSV SUMMARY
# ══════════════════════════════════════════════════════════════════════
# Final summary CSV
summary_rows = []
for algo in ALGOS:
    row = {
        'Algorithm': algo,
        'Composite_Score': composite_avg[algo],
        'Composite_Rank': int(composite_rank[algo]),
        'Normalised_Total': norm_scores.loc[algo, '★ Weighted Total'],
        'Wins_Overall': win_matrix.loc[algo, 'Overall_Best'],
    }
    for m, info in METRICS.items():
        row[f'Avg_{m}'] = avg_table.loc[algo, m]
        row[f'Rank_{m}'] = int(rank_table.loc[algo, m + '_rank'])
        row[f'Wins_{m}'] = win_matrix.loc[algo, m]
    # Robustness
    row['SSR_Red_Std'] = robust.loc[algo, 'std']
    row['SSR_Red_CV'] = robust.loc[algo, 'CV']
    # Efficiency
    row['Efficiency'] = pareto_df.loc[algo, 'Efficiency']
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values('Composite_Rank')
csv_path = os.path.join(OUT_DIR, 'comprehensive_ranking_summary.csv')
summary_df.to_csv(csv_path, index=False)
print(f"\n  Summary CSV: {csv_path}")

# Also export per-dataset composite
composite_per_ds.to_csv(os.path.join(OUT_DIR, 'composite_per_dataset.csv'), index=False)

print("\n" + "=" * 90)
print("  ★ FINAL COMPREHENSIVE RANKING ★")
print("=" * 90)
for _, row in summary_df.iterrows():
    print(f"\n  #{int(row['Composite_Rank'])} {row['Algorithm']}")
    print(f"     Weighted Composite Score: {row['Composite_Score']:.3f} (lower=better)")
    print(f"     Normalised Total (0-100): {row['Normalised_Total']:.1f} (higher=better)")
    print(f"     Overall Wins: {int(row['Wins_Overall'])}/{N_DS}")
    print(f"     Avg SSR Reduction: {row['Avg_SSR_Reduction']:.4f} (rank #{int(row['Rank_SSR_Reduction'])})")
    print(f"     Avg R²: {row['Avg_Avg_R2']:.4f} (rank #{int(row['Rank_Avg_R2'])})")
    print(f"     Avg Variance Ratio: {row['Avg_Variance_Ratio']:.4f} (rank #{int(row['Rank_Variance_Ratio'])})")
    print(f"     Avg RMSE: {row['Avg_Avg_RMSE']:.4f} (rank #{int(row['Rank_Avg_RMSE'])})")
    print(f"     Avg Runtime: {row['Avg_Runtime_s']:.1f}s (rank #{int(row['Rank_Runtime_s'])})")
    print(f"     Robustness (CV): {row['SSR_Red_CV']:.4f}")
    print(f"     Efficiency (SSR/log(t)): {row['Efficiency']:.4f}")

print(f"\n  Total output files: {len(os.listdir(OUT_DIR))}")
print("  DONE!")
