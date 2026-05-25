"""Compute final per-algorithm summary numbers for paper writing."""
import csv
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# Real-world: compute mean per algorithm across 11 datasets
def real_world_means():
    rows = load(os.path.join(ROOT, 'real_world_study_v2', 'results_baselines',
                              'algorithm_comparison.csv')) \
         + load(os.path.join(ROOT, 'real_world_study_v2', 'results_evolved',
                              'algorithm_comparison.csv'))
    by_algo = defaultdict(list)
    for r in rows:
        try:
            by_algo[r['Algorithm']].append({
                'avg_r2': float(r['Avg_R2']),
                'avg_rmse': float(r['Avg_RMSE']),
                'ssr_red': float(r['SSR_Reduction']),
                'var_ratio': float(r['Variance_Ratio']),
                'time': float(r['Runtime_s']),
            })
        except (ValueError, KeyError):
            pass
    print('=' * 100)
    print('REAL-WORLD MEAN PERFORMANCE (across 11 datasets)')
    print('=' * 100)
    print(f"{'Algorithm':<32} {'AvgR2':>8} {'AvgRMSE':>9} {'SSR_red':>8} {'VarRatio':>8} {'Time(s)':>8}  Wins(R2/SSRr/VR)")
    print('-' * 100)
    # win count: per dataset, count which algo wins
    rows_baseline = load(os.path.join(ROOT, 'real_world_study_v2', 'results_baselines',
                                       'algorithm_comparison.csv'))
    rows_evolved = load(os.path.join(ROOT, 'real_world_study_v2', 'results_evolved',
                                      'algorithm_comparison.csv'))
    all_rows = rows_baseline + rows_evolved
    by_ds = defaultdict(dict)
    for r in all_rows:
        try:
            by_ds[r['Dataset']][r['Algorithm']] = {
                'avg_r2': float(r['Avg_R2']),
                'avg_rmse': float(r['Avg_RMSE']),
                'ssr_red': float(r['SSR_Reduction']),
                'var_ratio': float(r['Variance_Ratio']),
            }
        except ValueError:
            pass
    excluded = {'SHAP-based', 'GeoEvolve_DynamicRAG', 'GeoEvolve'}
    wins = defaultdict(lambda: [0, 0, 0])  # r2, ssr_red, var_ratio
    for ds, algo_perf in by_ds.items():
        for metric_idx, mk in enumerate(('avg_r2', 'ssr_red', 'var_ratio')):
            best_algo, best_v = None, -np.inf
            for a, perf in algo_perf.items():
                if a in excluded:
                    continue
                if perf[mk] > best_v:
                    best_v = perf[mk]; best_algo = a
            wins[best_algo][metric_idx] += 1

    order = ['AZP', 'RegionKModels', 'GWR+SKATER', 'SKATER-reg',
             '2kmodels (initial)', 'OpenEvolve_NoGeoKnow', 'OpenEvolve_SimpleGeoKnow',
             'OpenEvolve_SpecificGeoKnow', 'GeoEvolve_NoRAG', 'GeoEvolve_StaticRAG']
    for a in order:
        if a not in by_algo:
            continue
        recs = by_algo[a]
        w = wins[a]
        print(f"{a:<32} {np.mean([r['avg_r2'] for r in recs]):>8.3f} "
              f"{np.mean([r['avg_rmse'] for r in recs]):>9.3f} "
              f"{np.mean([r['ssr_red'] for r in recs]):>8.3f} "
              f"{np.mean([r['var_ratio'] for r in recs]):>8.3f} "
              f"{np.mean([r['time'] for r in recs]):>8.1f}  "
              f"{w[0]}/{w[1]}/{w[2]}")
    print()


# Simulation: read from per-pattern CSVs
def sim_summary():
    rows = []
    for path in [
        os.path.join(ROOT, 'simulation_study', 'results', 'baseline_per_pattern.csv'),
        os.path.join(ROOT, 'simulation_study', 'results', 'evolved_per_pattern.csv'),
    ]:
        rows.extend(load(path))
    by_algo = defaultdict(dict)
    for r in rows:
        by_algo[r['algorithm']][r['pattern']] = {
            'ssr': float(r['ssr_mean']),
            'randi': float(r['randi_mean']),
            'nmi': float(r['nmi_mean']),
            'mae': float(r['mae_mean']),
            'time': float(r['time_mean']),
        }
    print('=' * 110)
    print('SIMULATION PER-PATTERN MEAN PERFORMANCE (n=50 datasets per pattern, h-noise)')
    print('=' * 110)
    print(f"{'Algorithm':<30}  Rect-SSR  Vor-SSR   Arb-SSR  RectRandI  VorRandI  ArbRandI  AvgSSR  AvgRandI  Time")
    print('-' * 110)
    order_classical = ['AZP', 'RegionKModels', 'GWR+SKATER', 'SKATER-reg', 'SHAP-based']
    order_evolved = ['2kmodels (initial)', 'OpenEvolve_NoGeoKnow', 'OpenEvolve_SimpleGeoKnow',
                     'OpenEvolve_SpecificGeoKnow', 'GeoEvolve_NoRAG', 'GeoEvolve_StaticRAG',
                     'GeoEvolve_DynamicRAG']
    for a in order_classical + order_evolved:
        if a not in by_algo:
            continue
        d = by_algo[a]
        ssrs = [d[p]['ssr'] for p in ('Rectangular', 'Voronoi', 'Arbitrary') if p in d]
        randis = [d[p]['randi'] for p in ('Rectangular', 'Voronoi', 'Arbitrary') if p in d]
        times = [d[p]['time'] for p in ('Rectangular', 'Voronoi', 'Arbitrary') if p in d]
        print(f"{a:<30}  "
              f"{d['Rectangular']['ssr']:>7.2f}  "
              f"{d['Voronoi']['ssr']:>7.2f}  "
              f"{d['Arbitrary']['ssr']:>7.2f}  "
              f"{d['Rectangular']['randi']:>8.3f}   "
              f"{d['Voronoi']['randi']:>7.3f}   "
              f"{d['Arbitrary']['randi']:>7.3f}   "
              f"{np.mean(ssrs):>6.2f}   {np.mean(randis):>7.3f}   {np.mean(times):>5.1f}")


sim_summary()
print()
real_world_means()
