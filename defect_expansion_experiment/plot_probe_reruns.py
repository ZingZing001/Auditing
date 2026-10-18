"""Figures from Aolin Yang's every-point reruns (aolin12138/Auditing, commit 586a609; copies in
aolin_586a609/). Each run stores spread with every test point attacked and, in *_correct columns,
with only correctly classified points attacked. Ratios are paired per seed; bands are 95% intervals.
  probe_label_noise.pdf  spread vs noise-free run (tree + DTA, SVM + HSJ; iris, wine) and accuracy
  probe_outliers.pdf     spread vs matched no-outlier baseline over outlier distance k (depth-3/10 trees)
  probe_leakage.pdf      measured vs honest accuracy, and leaked / clean attack margin
Also prints the coverage-gap excess (% of random arm) for the SVM every-point rerun.

Usage (from repo root):  uv run python defect_expansion_experiment/plot_probe_reruns.py OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_style as rs

HERE = Path(__file__).resolve().parent; A = HERE / 'aolin_586a609'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'plots'; OUT.mkdir(parents=True, exist_ok=True)
ci = rs.ci
PROBES = [('', 'every'), ('_correct', 'correct')]
DATASETS = [('iris', 'Iris'), ('wine', 'Wine')]


def paired(q, key, levels, col, base_level=0):
    """Per-seed ratio of `col` at each level to the same seed at the base level (matched by seed)."""
    b = q.filter(pl.col(key) == base_level).select(['seed', pl.col(col).alias('_b')])
    return [(r[col] / r['_b']).to_numpy() for r in (q.filter(pl.col(key) == v).join(b, on='seed') for v in levels)]


# ---- label noise: rows = tree spread, SVM spread, accuracy; columns = dataset
ln = pl.concat([pl.read_parquet(A / 'results_label_noise.parquet'), pl.read_parquet(A / 'results_label_noise_svm.parquet')])
nv = sorted(ln['noise'].unique().to_list()); x = 100 * np.array(nv)
fig, axes = plt.subplots(3, 2, figsize=(rs.WIDTH, 7.0), sharex=True, layout='constrained')
for j, (ds, dlabel) in enumerate(DATASETS):
    for i, model in enumerate(['tree', 'svm']):
        q = ln.filter((pl.col('model') == model) & (pl.col('dataset') == ds))
        for suf, role in PROBES:
            rs.draw(axes[i, j], x, paired(q, 'noise', nv, 'mean_dist' + suf), role)
        rs.refline(axes[i, j], 1.0)
        rs.draw(axes[2, j], x, [q.filter(pl.col('noise') == v)['vacc'].to_numpy() for v in nv], model)
    rs.refline(axes[2, j], 1 / 3, chance=True)
    for i in range(3):
        rs.tidy(axes[i, j], dlabel, 'abcdef'[2 * i + j]); axes[i, j].set_xticks([0, 30, 60, 90])
for i in range(2):
    rs.share_ylim(axes[i], 0.5); axes[i, 0].set_ylabel('Normalised spread\n(× noise-free)')
rs.rowlabel(axes[0, 0], 'Tree, white-box'); rs.rowlabel(axes[1, 0], 'SVM, HopSkipJump')
for a in axes[2]:
    a.set_ylim(-0.03, 1.03)
axes[2, 0].set_ylabel('Test accuracy')
fig.supxlabel('Training labels flipped (%)', fontsize=11)
rs.legend(fig, [('Spread:', ['every', 'correct']), ('Accuracy:', ['tree', 'svm'])])
fig.savefig(OUT / 'probe_label_noise.pdf'); plt.close(fig)

# ---- outliers (k sweep against the matched 50-seed no-outlier baseline): columns = tree depth
kp, kb = pl.read_parquet(A / 'results_kplateau.parquet'), pl.read_parquet(A / 'results_kplateau_baseline.parquet')
kv = sorted(kp['k'].unique().to_list())
fig, axes = plt.subplots(1, 2, figsize=(rs.WIDTH, 2.9), sharey=True, layout='constrained')
for ax, (model, name), letter in zip(axes, [('tree_d3', 'Depth-3 tree'), ('tree_d10', 'Depth-10 tree')], 'ab'):
    q = pl.concat([kb.filter(pl.col('model') == model), kp.filter(pl.col('model') == model)])
    for suf, role in PROBES:
        rs.draw(ax, kv, paired(q, 'k', kv, 'mean_dist' + suf, 0.0), role)
    rs.refline(ax, 1.0); rs.tidy(ax, name, letter); ax.set_xticks([6, 12, 18, 24])
axes[0].set_ylabel('Normalised spread\n(× no outliers)')
fig.supxlabel('Outlier distance $k$ (class standard deviations)', fontsize=11)
rs.legend(fig, ['every', 'correct'])
fig.savefig(OUT / 'probe_outliers.pdf'); plt.close(fig)

# ---- leakage: rows = accuracy (measured vs honest), margin ratio (probe); columns = dataset
lk = pl.read_parquet(A / 'results_leakage.parquet'); lv = sorted(lk['n_leak'].unique().to_list())
fig, axes = plt.subplots(2, 2, figsize=(rs.WIDTH, 5.0), sharex=True, layout='constrained')
for j, (ds, dlabel) in enumerate(DATASETS):
    q = lk.filter(pl.col('dataset') == ds)
    for col, role in [('vacc', 'measured'), ('vacc_clean', 'honest')]:
        rs.draw(axes[0, j], lv, [q.filter(pl.col('n_leak') == v)[col].to_numpy() for v in lv], role)
    for suf, role in PROBES:
        rs.draw(axes[1, j], lv[1:], [(r['margin_leak' + suf] / r['margin_clean' + suf]).to_numpy()
                                     for r in (q.filter(pl.col('n_leak') == v) for v in lv[1:])], role)
    rs.refline(axes[1, j], 1.0)
    for i in range(2):
        rs.tidy(axes[i, j], dlabel, 'abcd'[2 * i + j]); axes[i, j].set_xticks([0, 5, 10, 15, 20])
rs.share_ylim(axes[0], 0.02); rs.share_ylim(axes[1], 0.05)
axes[0, 0].set_ylabel('Test accuracy'); axes[1, 0].set_ylabel('Attack-margin ratio\n(leaked / clean)')
fig.supxlabel('Training points copied into each test fold', fontsize=11)
rs.legend(fig, [('Accuracy:', ['measured', 'honest']), ('Margin ratio:', ['every', 'correct'])])
fig.savefig(OUT / 'probe_leakage.pdf'); plt.close(fig)
print('wrote', OUT / 'probe_label_noise.pdf', OUT / 'probe_outliers.pdf', OUT / 'probe_leakage.pdf')

# ---- SVM coverage-gap rerun: excess over random deletion, % of random arm (as in report_tables.py)
sv = pl.read_parquet(A / 'results_attack_all_svm.parquet')
for setting in ['iris', 'iris_plen', 'wine']:
    q = sv.filter(pl.col('dataset') == setting); b = q.filter(pl.col('frac') == 0).sort('seed')['spread_all'].to_numpy(); out = []
    for f in [0.4, 0.6, 0.8, 0.9]:
        g, r = (q.filter((pl.col('frac') == f) & (pl.col('structure') == s)).sort('seed')['spread_all'].to_numpy() / b
                for s in ('spatial', 'random'))
        d = g - r; mk = '*' if np.nanmean(d) - ci(d) > 0 else ('r' if np.nanmean(d) + ci(d) < 0 else ' ')
        out.append(f'{int(f * 100)}% {100 * np.nanmean(d) / np.nanmean(r):+.0f}{mk}')
    print(f'SVM every point, {setting}: ' + '  '.join(out))

# ---- every value quoted in the report from these reruns
print('\nLabel noise (tab:noise): spread / no-noise run, every point / correct only; accuracy at 40% / 80%')
for model in ['tree', 'svm']:
    for ds in ['iris', 'wine']:
        q = ln.filter((pl.col('model') == model) & (pl.col('dataset') == ds))
        cells = [f'{np.nanmean(a):.2f} / {np.nanmean(c):.2f}' for a, c in
                 zip(paired(q, 'noise', [0.2, 0.4, 0.6, 0.8], 'mean_dist'), paired(q, 'noise', [0.2, 0.4, 0.6, 0.8], 'mean_dist_correct'))]
        acc = [q.filter(pl.col('noise') == v)['vacc'].mean() for v in (0.4, 0.8)]
        rng_all = [np.nanmean(v) for v in paired(q, 'noise', nv[1:], 'mean_dist')]
        print(f'  {model} {ds}: ' + ' | '.join(cells) + f' | acc {acc[0]:.2f} / {acc[1]:.2f}'
              f'   every-point range {min(rng_all):.2f}-{max(rng_all):.2f}, correct-only at 90% '
              f'{np.nanmean(paired(q, "noise", [0.9], "mean_dist_correct")[0]):.2f}')
print('\nLeakage: paired margin ratio leaked / clean [95% interval], and global spread / no-leak run')
for ds in ['iris', 'wine']:
    q = lk.filter(pl.col('dataset') == ds)
    for v in lv[1:]:
        r = q.filter(pl.col('n_leak') == v).sort('seed'); out = []
        for suf in ('', '_correct'):
            x = (r['margin_leak' + suf] / r['margin_clean' + suf]).to_numpy(); d = (r['margin_leak' + suf] - r['margin_clean' + suf]).to_numpy()
            out.append(f'{suf or "every":8s} {np.nanmean(x):.3f} [{np.nanmean(x) - ci(x):.3f}, {np.nanmean(x) + ci(x):.3f}] diff {np.nanmean(d):+.3f}±{ci(d):.3f}')
        sp = paired(q, 'n_leak', [v], 'spread_m0')[0]
        print(f'  {ds} n={v:2d}: ' + ' | '.join(out) + f' | global spread {np.nanmean(sp):.3f}')
print('\nOutliers: spread / matched no-outlier baseline, every point / correct only')
for model in ['tree_d3', 'tree_d10']:
    q = pl.concat([kb.filter(pl.col('model') == model), kp.filter(pl.col('model') == model)])
    print(f'  {model}: ' + '  '.join(f'k{k:g} {np.nanmean(a):.2f}/{np.nanmean(c):.2f}' for k, a, c in
                                     zip(kv, paired(q, 'k', kv, 'mean_dist', 0.0), paired(q, 'k', kv, 'mean_dist_correct', 0.0))))
