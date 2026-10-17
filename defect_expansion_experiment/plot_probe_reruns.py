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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; A = HERE / 'aolin_586a609'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'plots'; OUT.mkdir(parents=True, exist_ok=True)
CORAL, TEAL, INK, MUTED, GRID = '#E2553A', '#1E9A8A', '#13233F', '#4F5D75', '#DDD7CB'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
ci = lambda v: 1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v)))
PIPES = [('', CORAL, 'every point'), ('_correct', TEAL, 'correct only')]


def tidy(ax, ref=None):
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=0.8, ls=(0, (2, 3)))


def band(ax, x, vals, colr, ls, label):
    m = np.array([np.nanmean(v) for v in vals]); c = np.array([ci(v) for v in vals])
    ax.fill_between(x, m - c, m + c, color=colr, alpha=0.12, lw=0)
    ax.plot(x, m, color=colr, ls=ls, lw=1.8, marker='o', ms=3.5, mec='white', mew=0.8, label=label)


def paired(q, key, levels, col, base_level=0):
    b = q.filter(pl.col(key) == base_level).sort('seed')[col].to_numpy()
    return [q.filter(pl.col(key) == v).sort('seed')[col].to_numpy() / b for v in levels]


# ---- label noise
ln = pl.concat([pl.read_parquet(A / 'results_label_noise.parquet'), pl.read_parquet(A / 'results_label_noise_svm.parquet')])
nv = sorted(ln['noise'].unique().to_list()); x = 100 * np.array(nv)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, model, title in [(axes[0], 'tree', 'Tree, white-box'), (axes[1], 'svm', 'SVM, HopSkipJump')]:
    for ds, ls in [('iris', '-'), ('wine', '--')]:
        q = ln.filter((pl.col('model') == model) & (pl.col('dataset') == ds))
        for suf, colr, lab in PIPES:
            band(ax, x, paired(q, 'noise', nv, 'mean_dist' + suf), colr, ls, f'{ds}, {lab}')
    ax.set_title(title, color=INK, fontsize=9); ax.set_ylabel('spread (× no noise)'); tidy(ax, 1.0)
for (model, colr), ds_ls in [((m, c), d) for m, c in [('tree', INK), ('svm', MUTED)] for d in [('iris', '-'), ('wine', '--')]]:
    q = ln.filter((pl.col('model') == model) & (pl.col('dataset') == ds_ls[0]))
    band(axes[2], x, [q.filter(pl.col('noise') == v)['vacc'].to_numpy() for v in nv], colr, ds_ls[1], f'{model}, {ds_ls[0]}')
axes[2].set_title('Test accuracy', color=INK, fontsize=9); tidy(axes[2], 1 / 3)
for ax in axes:
    ax.set_xlabel('% of training labels flipped')
axes[0].legend(frameon=False, fontsize=6.5, loc='upper left'); axes[2].legend(frameon=False, fontsize=6.5, loc='upper right')
fig.tight_layout(); fig.savefig(OUT / 'probe_label_noise.pdf'); plt.close(fig)

# ---- outliers (k sweep against the matched 50-seed no-outlier baseline)
kp, kb = pl.read_parquet(A / 'results_kplateau.parquet'), pl.read_parquet(A / 'results_kplateau_baseline.parquet')
kv = sorted(kp['k'].unique().to_list())
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for model, ls, name in [('tree_d3', '-', 'depth 3'), ('tree_d10', '--', 'depth 10')]:
    q = pl.concat([kb.filter(pl.col('model') == model), kp.filter(pl.col('model') == model)])
    for suf, colr, lab in PIPES:
        band(ax, kv, paired(q, 'k', kv, 'mean_dist' + suf, 0.0), colr, ls, f'{name}, {lab}')
ax.set_xlabel('outlier distance $k$ (class standard deviations)'); ax.set_ylabel('spread (× no outliers)')
tidy(ax, 1.0); ax.legend(frameon=False, fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(OUT / 'probe_outliers.pdf'); plt.close(fig)

# ---- leakage
lk = pl.read_parquet(A / 'results_leakage.parquet'); lv = sorted(lk['n_leak'].unique().to_list())
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ds, ls in [('iris', '-'), ('wine', '--')]:
    q = lk.filter(pl.col('dataset') == ds)
    for col, colr, lab in [('vacc', INK, 'measured'), ('vacc_clean', MUTED, 'honest')]:
        band(axes[0], lv, [q.filter(pl.col('n_leak') == v)[col].to_numpy() for v in lv], colr, ls, f'{ds}, {lab}')
    for suf, colr, lab in PIPES:
        band(axes[1], lv[1:], [(r['margin_leak' + suf] / r['margin_clean' + suf]).to_numpy()
                               for r in (q.filter(pl.col('n_leak') == v).sort('seed') for v in lv[1:])], colr, ls, f'{ds}, {lab}')
axes[0].set_ylabel('test accuracy'); axes[1].set_ylabel('margin, leaked / clean'); tidy(axes[0]); tidy(axes[1], 1.0)
for ax in axes:
    ax.set_xlabel('leaked points per test fold'); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'probe_leakage.pdf'); plt.close(fig)
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
