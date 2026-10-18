"""Figures for the every-point runs, in the report style (report_style.py).
  attack_all_spread.pdf, attack_all_recall.pdf  (attack_all_points.csv): coverage gap vs random deletion,
      rows = model/attack, columns = deletion setting; spread normalised per seed by its defect-free run.
  shortcut_attack_all.pdf  (shortcut_attack_all.csv): tree vs SVM, rows = shortcut share / spread /
      accuracy, columns = iris / wine.
Bands are 95% intervals (mean +/- 1.96 standard errors over seeds).

Usage (from repo root):  uv run python defect_expansion_experiment/plot_attack_all.py OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_style as rs

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
SETTINGS = [('iris_pw', 'Iris, petal width'), ('iris_pl', 'Iris, petal length'), ('wine', 'Wine, proline')]
MODELS = [('tree', 'Tree, white-box'), ('svm', 'SVM, HopSkipJump')]
FR = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]


def per_seed(q, frac, structure, col, base=None):
    """Values for one cell, matched to the defect-free run by seed when normalising."""
    r = q.filter((pl.col('frac') == frac) & (pl.col('structure') == structure)).select(['seed', col])
    if base is None:
        return r[col].to_numpy()
    r = r.join(base, on='seed', how='inner')
    return (r[col] / r['base']).to_numpy()


def attack_all_figure(col, ylabel, fname, norm):
    df = pl.read_csv(HERE / 'attack_all_points.csv')
    fig, axes = plt.subplots(2, 3, figsize=(rs.WIDTH, 4.1), sharex=True, sharey=True, layout='constrained')
    x = np.array(FR) * 100
    for i, (mk, mlabel) in enumerate(MODELS):
        for j, (sk, slabel) in enumerate(SETTINGS):
            ax = axes[i, j]; q = df.filter((pl.col('model') == mk) & (pl.col('setting') == sk))
            base = q.filter(pl.col('frac') == 0).select(['seed', pl.col('spread').alias('base')]) if norm else None
            for role, structure in (('random', 'random'), ('gap', 'spatial')):
                vals = [per_seed(q, f, 'random' if f == 0 else structure, col, base) for f in FR]
                rs.line(ax, x, [np.nanmean(v) for v in vals], [rs.ci(v) for v in vals], role)
            if norm:
                rs.refline(ax, 1.0)
            rs.tidy(ax, slabel, 'abcdef'[3 * i + j])
            ax.set_xticks([0, 30, 60, 90])
        rs.rowlabel(axes[i, 0], mlabel)
    if norm:
        rs.share_ylim(axes.ravel(), 0.05)
    else:
        for a in axes.ravel():
            a.set_ylim(-0.03, 1.03)
        axes[0, 0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for a in axes[:, 0]:
        a.set_ylabel(ylabel)
    fig.supxlabel('Target class removed (%)', fontsize=11)
    rs.legend(fig, ['gap', 'random'])
    fig.savefig(OUT / fname); plt.close(fig); print('wrote', OUT / fname)


def shortcut_figure(fname='shortcut_attack_all.pdf'):
    sc = pl.read_csv(HERE / 'shortcut_attack_all.csv'); corr = sorted(sc['corr'].unique().to_list())
    rows = [('spur_frac', 'Shortcut share'), ('spread', 'Spread ratio'), ('vacc', 'Test accuracy')]
    fig, axes = plt.subplots(3, 2, figsize=(rs.WIDTH, 4.5), sharex=True, layout='constrained')
    for j, (ds, dlabel) in enumerate([('iris', 'Iris'), ('wine', 'Wine')]):
        for model in ('tree', 'svm'):
            q = sc.filter((pl.col('model') == model) & (pl.col('dataset') == ds))
            base = q.filter(pl.col('corr') == 0).select(['seed', pl.col('spread').alias('base')])
            for i, (col, _) in enumerate(rows):
                m, c = [], []
                for k in corr:
                    r = q.filter(pl.col('corr') == k)
                    if col == 'spread':   # drop cells where under half the folds give a spread
                        v = r.join(base, on='seed')
                        v = (v['spread'] / v['base']).to_numpy() if r['valid_folds_spread'].sum() >= 2.5 * len(r) else np.array([np.nan])
                    else:
                        v = r[col].to_numpy()
                    m.append(np.nanmean(v) if np.any(~np.isnan(v)) else np.nan)
                    c.append(rs.ci(v) if np.sum(~np.isnan(v)) > 1 else np.nan)
                rs.line(axes[i, j], corr, m, c, model)
        for i, (_, ylabel) in enumerate(rows):
            ax = axes[i, j]; rs.tidy(ax, dlabel, 'abcdef'[2 * i + j]); ax.set_xticks(corr)
            if j == 0:
                ax.set_ylabel(ylabel)
    for j in range(2):
        rs.refline(axes[1, j], 1.0)
        axes[0, j].set_ylim(-0.03, 1.03); axes[2, j].set_ylim(-0.03, 1.03)
    rs.share_ylim(axes[1], 0.05)
    fig.supxlabel('Shortcut strength', fontsize=11)
    rs.legend(fig, ['tree', 'svm'])
    fig.savefig(OUT / fname); plt.close(fig); print('wrote', OUT / fname)


if __name__ == '__main__':
    attack_all_figure('spread', 'Normalised spread\n(× defect-free)', 'attack_all_spread.pdf', norm=True)
    attack_all_figure('recall', 'Depleted-class recall', 'attack_all_recall.pdf', norm=False)
    shortcut_figure()
