"""Redraw three legacy figures without printed conclusions, in the report's style.

Inputs (repo-relative):
  dtree_attack_experiment/results.parquet            tree + DTA, label noise rows (defect == 'label_noise')
  hsj_label_noise_experiment/results.parquet         tree / SVM + HopSkipJump, label noise
  defect_expansion_experiment/results_variance.parquet, results_variance_svm.parquet   (Aolin Yang, commit 350416a)
  defect_expansion_experiment/results_leakage.parquet                                  (Aolin Yang, commit 350416a)
Outputs: label_noise_redrawn.pdf, dims_redrawn.pdf, leakage_redrawn.pdf in OUT_DIR.

Usage (from repo root):  uv run python defect_expansion_experiment/plot_legacy_redraw.py OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / 'defect_expansion_experiment'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'plots'; OUT.mkdir(parents=True, exist_ok=True)
CORAL, TEAL, INK, MUTED, GRID = '#E2553A', '#1E9A8A', '#13233F', '#4F5D75', '#DDD7CB'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
ci = lambda v: 1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v)))


def tidy(ax):
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def line(ax, x, groups, col, colr, label, marker='o'):
    m = np.array([np.nanmean(g[col].to_numpy()) for g in groups]); c = np.array([ci(g[col].to_numpy()) for g in groups])
    ax.fill_between(x, m - c, m + c, color=colr, alpha=0.12, lw=0)
    ax.plot(x, m, color=colr, lw=2, marker=marker, ms=4, mec='white', mew=1, label=label)


# ---- 1. label noise
dt = pl.read_parquet(ROOT / 'dtree_attack_experiment' / 'results.parquet').filter(pl.col('defect') == 'label_noise')
hs = pl.read_parquet(ROOT / 'hsj_label_noise_experiment' / 'results.parquet')
lv = sorted(dt['level'].unique().to_list()); nv = sorted(hs['noise'].unique().to_list())
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
line(a1, lv, [dt.filter(pl.col('level') == v) for v in lv], 'mean_dist', INK, 'Tree, white-box', 's')
line(a1, nv, [hs.filter((pl.col('model') == 'tree') & (pl.col('noise') == v)) for v in nv], 'mean_dist', TEAL, 'Tree, HopSkipJump')
line(a1, nv, [hs.filter((pl.col('model') == 'svm') & (pl.col('noise') == v)) for v in nv], 'mean_dist', CORAL, 'SVM, HopSkipJump', 'D')
line(a2, nv, [hs.filter((pl.col('model') == 'tree') & (pl.col('noise') == v)) for v in nv], 'vacc', TEAL, 'Tree')
line(a2, nv, [hs.filter((pl.col('model') == 'svm') & (pl.col('noise') == v)) for v in nv], 'vacc', CORAL, 'SVM', 'D')
for a in (a1, a2):
    a.axvspan(0.5, max(nv) + 0.02, color=GRID, alpha=0.35, lw=0); tidy(a)
a2.axhline(1 / 3, color=MUTED, lw=0.8, ls=(0, (2, 3))); a2.text(0.12, 1 / 3 + 0.03, 'chance (1/3)', color=MUTED, fontsize=8)
a1.set_ylabel('adversarial spread'); a2.set_ylabel('test accuracy'); a2.set_xlabel('share of training labels flipped')
a1.legend(frameon=False, fontsize=8, loc='upper left'); fig.tight_layout(); fig.savefig(OUT / 'label_noise_redrawn.pdf'); plt.close(fig)

# ---- 2. dimensionality dot plot (original pipeline, train-only, 90% removal, paired per seed)
rows = []
for name, path in [('Tree, white-box', 'results_variance.parquet'), ('SVM, HopSkipJump', 'results_variance_svm.parquet')]:
    v = pl.read_parquet(HERE / path).filter(pl.col('protocol') == 'train_only')
    for ds, lab in [('iris', 'Iris (4-D)'), ('wine', 'Wine (13-D)')]:
        b = v.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.0)).select(['seed', pl.col('mean_dist').alias('b')])
        out = {}
        for st in ('random', 'spatial'):
            q = v.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.9) & (pl.col('structure') == st)).join(b, on='seed')
            r = (q['mean_dist'] / q['b']).to_numpy(); out[st] = (np.nanmean(r), ci(r))
        rows.append((f'{lab}\n{name}', out))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for i, (lab, out) in enumerate(rows[::-1]):
    for st, colr, dy, lb in [('random', TEAL, -0.12, 'Random deletion'), ('spatial', CORAL, 0.12, 'Coverage gap')]:
        m, c = out[st]; ax.errorbar(m, i + dy, xerr=c, fmt='o', color=colr, ms=6, mec='white', mew=1, capsize=0, lw=1.5,
                                   label=lb if i == 0 else None)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 3))); ax.set_xlabel('spread at 90% removal (× defect-free)')
ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc='lower right'); fig.tight_layout(); fig.savefig(OUT / 'dims_redrawn.pdf'); plt.close(fig)

# ---- 3. leakage
lk = pl.read_parquet(HERE / 'results_leakage.parquet'); nl = sorted(lk['n_leak'].unique().to_list())
fig, axes = plt.subplots(3, 2, figsize=(6.6, 6.0), sharex=True)
for j, ds in enumerate(['iris', 'wine']):
    g = [lk.filter((pl.col('dataset') == ds) & (pl.col('n_leak') == v)) for v in nl]
    line(axes[0, j], nl, g, 'vacc', CORAL, 'Measured accuracy'); line(axes[0, j], nl, g, 'vacc_clean', INK, 'Honest accuracy', 's')
    gl = [x for x, v in zip(g, nl) if v > 0]; xl = [v for v in nl if v > 0]
    line(axes[1, j], xl, gl, 'margin_leak', CORAL, 'Leaked points'); line(axes[1, j], nl, g, 'margin_clean', INK, 'Clean points', 's')
    line(axes[2, j], nl, g, 'spread_m0', MUTED, 'Global spread')
    axes[0, j].set_title({'iris': 'Iris (4-D)', 'wine': 'Wine (13-D)'}[ds], color=INK, fontsize=9.5)
    for i in range(3):
        tidy(axes[i, j])
    axes[2, j].set_xlabel('training points copied into the test fold')
axes[0, 0].set_ylabel('test accuracy'); axes[1, 0].set_ylabel('attack margin (L2)'); axes[2, 0].set_ylabel('adversarial spread')
axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left'); axes[1, 0].legend(frameon=False, fontsize=8, loc='lower left')
fig.tight_layout(); fig.savefig(OUT / 'leakage_redrawn.pdf'); plt.close(fig)
print('wrote', OUT / 'label_noise_redrawn.pdf', OUT / 'dims_redrawn.pdf', OUT / 'leakage_redrawn.pdf')
