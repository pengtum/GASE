#!/usr/bin/env python3
"""Generate LaTeX tables from algorithm_comparison.csv, grouped by domain."""

import csv
from collections import OrderedDict

CSV_PATH = "results_v2/algorithm_comparison.csv"
OUT_PATH = "results_v2/latex_tables.tex"

# Domain classification
def get_domain(ds):
    if 'Climate' in ds:  return 'Climate'
    if 'Health'  in ds:  return 'Health'
    if 'Hydro'   in ds:  return 'Hydro'
    if 'Politics' in ds: return 'Politics'
    return 'Other'

# Pretty dataset names
def pretty_name(ds):
    mapping = {
        'US_Climate_ERA5_CLIMATE': 'ERA5 Climate',
        'US_Health_ARTHRITIS':     'Arthritis',
        'US_Health_BPHIGH':        'High Blood Pressure',
        'US_Health_CANCER':        'Cancer',
        'US_Health_CASTHMA':       'Asthma',
        'US_Health_DEPRESSION':    'Depression',
        'US_Health_DIABETES':      'Diabetes',
        'US_Health_OBESITY':       'Obesity',
        'US_Health_STROKE':        'Stroke',
        'US_Hydro_CAMELS':         'CAMELS Hydrology',
        'US_Politics_Voting':      'Voting',
    }
    return mapping.get(ds, ds)

# Read CSV
rows = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Group by domain -> dataset -> algorithm
domains_order = ['Climate', 'Health', 'Hydro', 'Politics']
data = OrderedDict()
for dom in domains_order:
    data[dom] = OrderedDict()

for r in rows:
    ds = r['Dataset']
    dom = get_domain(ds)
    if dom not in data:
        data[dom] = OrderedDict()
    if ds not in data[dom]:
        data[dom][ds] = {}
    data[dom][ds][r['Algorithm']] = r

algos = ['Initial', 'OpenEvolve', 'GeoEvolve']

# Metrics to display (4 quality metrics only)
metrics = [
    ('Avg_R2',        'Avg $R^2$', '{:.4f}', True),    # higher is better
    ('Avg_RMSE',      'Avg RMSE',  '{:.4f}', False),   # lower is better
    ('SSR_Reduction', 'SSR Red.',   '{:.4f}', True),    # higher is better
    ('Variance_Ratio','Var. Ratio', '{:.4f}', True),    # higher is better
]

def fmt_val(val_str, fmt):
    return fmt.format(float(val_str))

def bold(s):
    return r'\textbf{' + s + '}'

# ---------------------------------------------------------------------------
# Build a single comprehensive table grouped by domain with midrules
# ---------------------------------------------------------------------------
lines = []
lines.append(r'% ============================================================')
lines.append(r'% Comprehensive algorithm comparison table (grouped by domain)')
lines.append(r'% ============================================================')
lines.append(r'\begin{table}[htbp]')
lines.append(r'\centering')
lines.append(r'\caption{Algorithm performance comparison across real-world datasets grouped by domain. '
             r'Best values per dataset are in \textbf{bold}.}')
lines.append(r'\label{tab:algo_comparison}')
lines.append(r'\small')
lines.append(r'\setlength{\tabcolsep}{4pt}')
lines.append(r'\begin{tabular}{ll' + 'r' * len(metrics) + '}')
lines.append(r'\toprule')

# Header
header_cols = ['Dataset', 'Algorithm']
for _, label, _, _ in metrics:
    header_cols.append(label)
lines.append(' & '.join(header_cols) + r' \\')
lines.append(r'\midrule')

domain_labels = {
    'Climate':  r'\multicolumn{' + str(2 + len(metrics)) + r'}{l}{\textit{Climate}} \\',
    'Health':   r'\multicolumn{' + str(2 + len(metrics)) + r'}{l}{\textit{Health}} \\',
    'Hydro':    r'\multicolumn{' + str(2 + len(metrics)) + r'}{l}{\textit{Hydrology}} \\',
    'Politics': r'\multicolumn{' + str(2 + len(metrics)) + r'}{l}{\textit{Politics}} \\',
}

first_domain = True
for dom in domains_order:
    if dom not in data or not data[dom]:
        continue
    if not first_domain:
        lines.append(r'\midrule')
    first_domain = False
    lines.append(domain_labels[dom])
    lines.append(r'\midrule')

    datasets_in_dom = list(data[dom].keys())
    for di, ds in enumerate(datasets_in_dom):
        algo_data = data[dom][ds]

        # Find best value per metric for this dataset
        best_vals = {}
        for key, _, fmt, higher_better in metrics:
            vals = []
            for algo in algos:
                if algo in algo_data:
                    vals.append((algo, float(algo_data[algo][key])))
            if vals:
                if higher_better:
                    best_algo = max(vals, key=lambda x: x[1])[0]
                else:
                    best_algo = min(vals, key=lambda x: x[1])[0]
                best_vals[key] = best_algo

        for ai, algo in enumerate(algos):
            if algo not in algo_data:
                continue
            r_data = algo_data[algo]

            # First column: dataset name only for first algo
            if ai == 0:
                col0 = pretty_name(ds)
            else:
                col0 = ''

            cols = [col0, algo]
            for key, _, fmt, _ in metrics:
                val_str = fmt_val(r_data[key], fmt)
                if best_vals.get(key) == algo:
                    val_str = bold(val_str)
                cols.append(val_str)

            lines.append(' & '.join(cols) + r' \\')

        # Add a small space between datasets within same domain
        if di < len(datasets_in_dom) - 1:
            lines.append(r'\addlinespace[2pt]')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

# ---------------------------------------------------------------------------
# Also build a compact summary table: domain-level averages
# ---------------------------------------------------------------------------
lines.append('')
lines.append(r'% ============================================================')
lines.append(r'% Domain-level average performance summary')
lines.append(r'% ============================================================')
lines.append(r'\begin{table}[htbp]')
lines.append(r'\centering')
lines.append(r'\caption{Domain-level average performance by algorithm. '
             r'Best values per domain are in \textbf{bold}.}')
lines.append(r'\label{tab:domain_avg}')
lines.append(r'\small')
lines.append(r'\begin{tabular}{ll' + 'r' * len(metrics) + '}')
lines.append(r'\toprule')
header_cols = ['Domain', 'Algorithm']
for _, label, _, _ in metrics:
    header_cols.append(label)
lines.append(' & '.join(header_cols) + r' \\')
lines.append(r'\midrule')

domain_pretty = {'Climate': 'Climate', 'Health': 'Health', 'Hydro': 'Hydrology', 'Politics': 'Politics'}

first_dom = True
for dom in domains_order:
    if dom not in data or not data[dom]:
        continue
    if not first_dom:
        lines.append(r'\midrule')
    first_dom = False

    # Compute averages
    avg = {algo: {key: [] for key, _, _, _ in metrics} for algo in algos}
    for ds, algo_data in data[dom].items():
        for algo in algos:
            if algo in algo_data:
                for key, _, _, _ in metrics:
                    avg[algo][key].append(float(algo_data[algo][key]))

    # Find best
    best_vals = {}
    for key, _, fmt, higher_better in metrics:
        algo_avgs = []
        for algo in algos:
            if avg[algo][key]:
                algo_avgs.append((algo, sum(avg[algo][key]) / len(avg[algo][key])))
        if algo_avgs:
            if higher_better:
                best_vals[key] = max(algo_avgs, key=lambda x: x[1])[0]
            else:
                best_vals[key] = min(algo_avgs, key=lambda x: x[1])[0]

    for ai, algo in enumerate(algos):
        col0 = domain_pretty[dom] if ai == 0 else ''
        cols = [col0, algo]
        for key, _, fmt, _ in metrics:
            if avg[algo][key]:
                mean_val = sum(avg[algo][key]) / len(avg[algo][key])
                val_str = fmt.format(mean_val)
                if best_vals.get(key) == algo:
                    val_str = bold(val_str)
                cols.append(val_str)
            else:
                cols.append('--')
        lines.append(' & '.join(cols) + r' \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

# ---------------------------------------------------------------------------
# Overall average table
# ---------------------------------------------------------------------------
lines.append('')
lines.append(r'% ============================================================')
lines.append(r'% Overall average performance summary')
lines.append(r'% ============================================================')
lines.append(r'\begin{table}[htbp]')
lines.append(r'\centering')
lines.append(r'\caption{Overall average performance across all 11 datasets. '
             r'Best values are in \textbf{bold}.}')
lines.append(r'\label{tab:overall_avg}')
lines.append(r'\small')
lines.append(r'\begin{tabular}{l' + 'r' * len(metrics) + '}')
lines.append(r'\toprule')
header_cols = ['Algorithm']
for _, label, _, _ in metrics:
    header_cols.append(label)
lines.append(' & '.join(header_cols) + r' \\')
lines.append(r'\midrule')

# Compute overall averages
overall = {algo: {key: [] for key, _, _, _ in metrics} for algo in algos}
for dom in data:
    for ds in data[dom]:
        for algo in algos:
            if algo in data[dom][ds]:
                for key, _, _, _ in metrics:
                    overall[algo][key].append(float(data[dom][ds][algo][key]))

best_overall = {}
for key, _, fmt, higher_better in metrics:
    algo_avgs = []
    for algo in algos:
        if overall[algo][key]:
            algo_avgs.append((algo, sum(overall[algo][key]) / len(overall[algo][key])))
    if algo_avgs:
        if higher_better:
            best_overall[key] = max(algo_avgs, key=lambda x: x[1])[0]
        else:
            best_overall[key] = min(algo_avgs, key=lambda x: x[1])[0]

for algo in algos:
    cols = [algo]
    for key, _, fmt, _ in metrics:
        if overall[algo][key]:
            mean_val = sum(overall[algo][key]) / len(overall[algo][key])
            val_str = fmt.format(mean_val)
            if best_overall.get(key) == algo:
                val_str = bold(val_str)
            cols.append(val_str)
        else:
            cols.append('--')
    lines.append(' & '.join(cols) + r' \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

# Write output
tex_content = '\n'.join(lines)
with open(OUT_PATH, 'w') as f:
    f.write(tex_content)

print(f"LaTeX tables written to {OUT_PATH}")
print(f"Total lines: {len(lines)}")
print()
print("=" * 80)
print(tex_content)
