"""Regenerate the numbers in the report's coverage-gap tables from committed result files.

  Table tab:attackall   <- attack_all_points.csv
  Table tab:matched     <- matched_filtering.csv   (+ correct-minus-all paired contrasts)
  Table tab:cohort      <- results_imbalance.parquet (tree, standard, published grid by Aolin Yang),
                           robustness_joint_cohort[_moderate].csv, robustness_svm_cohort[_moderate].csv
  Clustering check      <- matched_cluster_check.csv
All values: mean paired (per-seed) difference, 95% normal interval 1.96 s / sqrt(n).

Usage (from repo root):  uv run python defect_expansion_experiment/report_tables.py
"""
from pathlib import Path
import numpy as np, polars as pl
H = Path(__file__).resolve().parent
ci = lambda a: 1.96 * np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))
mark = lambda d: '*' if np.nanmean(d) - ci(d) > 0 else ('r' if np.nanmean(d) + ci(d) < 0 else '')


def excess_pct(q, f, col='spread'):
    b = q.filter(pl.col('frac') == 0).sort('seed')[col].to_numpy()
    r = q.filter((pl.col('frac') == f) & (pl.col('structure') == 'random')).sort('seed')[col].to_numpy() / b
    g = q.filter((pl.col('frac') == f) & (pl.col('structure') == 'spatial')).sort('seed')[col].to_numpy() / b
    return g - r, r


print('== tab:attackall  (gap - random, % of random arm; recall at 90%)')
a = pl.read_csv(H / 'attack_all_points.csv')
for m in ['tree', 'svm']:
    for s in ['iris_pw', 'iris_pl', 'wine']:
        q = a.filter((pl.col('model') == m) & (pl.col('setting') == s)); cells = []
        for f in [0.4, 0.6, 0.8, 0.9]:
            d, r = excess_pct(q, f); cells.append(f'{100*np.nanmean(d)/np.nanmean(r):+.0f}{mark(d)}')
        rec = [q.filter((pl.col('frac') == 0.9) & (pl.col('structure') == st))['recall'].mean() for st in ('random', 'spatial')]
        print(f'  {m:4s} {s:8s} ' + '  '.join(cells) + f'   recall {rec[0]:.2f} / {rec[1]:.2f}')

print('\n== tab:matched  (C = correct only, A = every point)  and paired C - A contrast (percentage points)')
mf = pl.read_csv(H / 'matched_filtering.csv')
for m in ['tree', 'svm']:
    for s in ['iris_pw', 'iris_pl', 'wine']:
        out = {}
        for p in ('correct', 'all'):
            q = mf.filter((pl.col('model') == m) & (pl.col('setting') == s) & (pl.col('pipeline') == p))
            out[p] = [excess_pct(q, f) for f in [0.4, 0.6, 0.8, 0.9]]
        cl = '  '.join(f'{100*np.nanmean(d)/np.nanmean(r):+.0f}{mark(d)}' for d, r in out['correct'])
        al = '  '.join(f'{100*np.nanmean(d)/np.nanmean(r):+.0f}{mark(d)}' for d, r in out['all'])
        con = '  '.join(f'{100*np.nanmean(dc-da):+.1f}±{100*ci(dc-da):.1f}' for (dc, _), (da, _) in zip(out['correct'], out['all']))
        print(f'  {m:4s} {s:8s} C {cl}\n                A {al}\n                C-A {con}')

print('\n== tab:cohort')
imb = pl.read_parquet(H / 'results_imbalance.parquet').rename({'mean_dist': 'spread'})
for f in [0.25, 0.5, 0.7, 0.85, 0.95]:
    d, _ = excess_pct(imb, f); print(f'  tree standard (published) {int(f*100)}%: {np.nanmean(d):+.3f} ± {ci(d):.3f}')
for name, path, fr in [('tree fixed, moderate', 'robustness_joint_cohort_moderate.csv', [0.25, 0.5, 0.7]),
                       ('tree fixed, high', 'robustness_joint_cohort.csv', [0.5, 0.85, 0.95])]:
    c = pl.read_csv(H / path); b = c['random_0.0'].to_numpy()
    for f in fr:
        d = c[f'spatial_{f}'].to_numpy() / b - c[f'random_{f}'].to_numpy() / b
        print(f'  {name} {int(f*100)}%: {np.nanmean(d):+.3f} ± {ci(d):.3f}   cohort {c["n_cohort"].mean():.1f}')
for name, path, fr in [('svm, moderate', 'robustness_svm_cohort_moderate.csv', [0.25, 0.5, 0.7]),
                       ('svm, high', 'robustness_svm_cohort.csv', [0.5, 0.85, 0.95])]:
    c = pl.read_csv(H / path)
    for mode in ('std', 'fixed'):
        b = c[f'{mode}_random_0.0'].to_numpy()
        for f in fr:
            d = c[f'{mode}_spatial_{f}'].to_numpy() / b - c[f'{mode}_random_{f}'].to_numpy() / b
            print(f'  {name} {mode} {int(f*100)}%: {np.nanmean(d):+.3f} ± {ci(d):.3f}   cohort {c["n_cohort"].mean():.1f}')

print('\n== clustering check (attack every point): OPTICS vs unclustered excess, % of random arm')
cc = pl.read_csv(H / 'matched_cluster_check.csv').filter(pl.col('pipeline') == 'all')
for m in ['tree', 'svm']:
    for s in ['iris_pw', 'iris_pl', 'wine']:
        q = cc.filter((pl.col('model') == m) & (pl.col('setting') == s)); row = []
        for col in ('spread', 'spread_raw'):
            row.append(col + ' ' + '  '.join(f'{100*np.nanmean(d)/np.nanmean(r):+.0f}{mark(d)}' for d, r in [excess_pct(q, f, col) for f in [0.4, 0.6, 0.8, 0.9]]))
        print(f'  {m:4s} {s:8s} ' + ' | '.join(row))
