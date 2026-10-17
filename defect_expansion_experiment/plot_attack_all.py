"""Figures for attack_all_points.csv: normalised spread and depleted-class recall, coverage gap vs
random deletion, small multiples (rows: tree + white-box, SVM + HopSkipJump; cols: settings).
Spread is normalised per seed by that seed's defect-free run; bands are 95% intervals.
Also shortcut_attack_all.csv: shortcut share, normalised spread and accuracy against strength.

Usage (from repo root):  uv run python defect_expansion_experiment/plot_attack_all.py OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
df = pl.read_csv(HERE / 'attack_all_points.csv')
GAP, RAND, INK, MUTED, GRID = '#E2553A', '#1E9A8A', '#13233F', '#4F5D75', '#DDD7CB'
SETTINGS = [('iris_pw', 'Iris, petal width'), ('iris_pl', 'Iris, petal length'), ('wine', 'Wine, proline')]
MODELS = [('tree', 'Tree, white-box'), ('svm', 'SVM, HopSkipJump')]
FR = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def series(q, structure, col, base=None):
    m, c = [], []
    for f in FR:
        st = 'random' if f == 0 else structure
        v = q.filter((pl.col('frac') == f) & (pl.col('structure') == st)).sort('seed')[col].to_numpy()
        if base is not None:
            v = v / base
        m.append(np.nanmean(v)); c.append(1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v))))
    return np.array(m), np.array(c)


def figure(col, ylabel, fname, norm, legend_loc):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey='row' if norm else True)
    x = np.array(FR) * 100
    for i, (mk, mlabel) in enumerate(MODELS):
        for j, (sk, slabel) in enumerate(SETTINGS):
            ax = axes[i, j]; q = df.filter((pl.col('model') == mk) & (pl.col('setting') == sk))
            base = q.filter(pl.col('frac') == 0).sort('seed')['spread'].to_numpy() if norm else None
            for structure, colr, lab in [('random', RAND, 'Random deletion'), ('spatial', GAP, 'Coverage gap')]:
                m, c = series(q, structure, col, base)
                ax.fill_between(x, m - c, m + c, color=colr, alpha=0.12, lw=0)
                ax.plot(x, m, color=colr, lw=2, marker='o', ms=4, mec='white', mew=1, label=lab)
            if norm:
                ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 3)))
            ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(slabel, color=INK, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f'{mlabel}\n{ylabel}')
            if i == 1:
                ax.set_xlabel('% of class removed')
            ax.set_xticks([0, 20, 40, 60, 80, 90])
    axes[0, 0].legend(frameon=False, fontsize=8, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(OUT / fname); plt.close(fig)
    print('wrote', OUT / fname)


figure('spread', 'spread (× defect-free)', 'attack_all_spread.pdf', norm=True, legend_loc='upper left')
figure('recall', 'recall, depleted class', 'attack_all_recall.pdf', norm=False, legend_loc='lower left')


def shortcut_figure(fname='shortcut_attack_all.pdf'):
    sc = pl.read_csv(HERE / 'shortcut_attack_all.csv'); corr = sorted(sc['corr'].unique().to_list())
    panels = [('spur_frac', 'shortcut share of movement'), ('spread', 'spread (× no shortcut)'),
              ('vacc', 'test accuracy')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for mk, ds, colr, ls, lab in [('tree', 'iris', GAP, '-', 'Tree, iris'), ('tree', 'wine', GAP, '--', 'Tree, wine'),
                                  ('svm', 'iris', RAND, '-', 'SVM, iris'), ('svm', 'wine', RAND, '--', 'SVM, wine')]:
        q = sc.filter((pl.col('model') == mk) & (pl.col('dataset') == ds))
        base = q.filter(pl.col('corr') == 0).sort('seed')['spread'].to_numpy()
        for ax, (col, ylabel) in zip(axes, panels):
            m, c = [], []
            for k in corr:
                r = q.filter(pl.col('corr') == k).sort('seed'); v = r[col].to_numpy()
                if col == 'spread':   # drop cells where under half the folds give a spread
                    v = v / base if r['valid_folds_spread'].sum() >= 2.5 * len(r) else np.full(len(r), np.nan)
                m.append(np.nanmean(v)); c.append(1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v))))
            m, c = np.array(m), np.array(c)
            ax.fill_between(corr, m - c, m + c, color=colr, alpha=0.12, lw=0)
            ax.plot(corr, m, color=colr, ls=ls, lw=1.8, marker='o', ms=3.5, mec='white', mew=0.8, label=lab)
            ax.set_ylabel(ylabel); ax.set_xlabel('shortcut strength'); ax.set_xticks(corr)
    axes[1].axhline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    for ax in axes:
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)
    print('wrote', OUT / fname)


shortcut_figure()
