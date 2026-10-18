"""Redraw the report's figures that come from earlier grids, in the report style (report_style.py).

Inputs (repo-relative):
  defect_expansion_experiment/results_variance.parquet, results_variance_svm.parquet   (Aolin Yang, commit 350416a)
  defect_expansion_experiment/results_imbalance.parquet                                (Aolin Yang, commit 350416a)
  defect_expansion_experiment/results_p1.parquet, results_p1_rf.parquet                (Aolin Yang, commit 350416a)
  mlp_experiment/results.parquet                                                       (coverage-gap rows)
Outputs in OUT_DIR:
  dims_redrawn.pdf       coverage gap vs random deletion at 90% removal, by dataset and model (paired per seed)
  imbalance_original.pdf original pipeline, overfit tree + white-box attack: spread, recall, accuracy
  imbalance_models.pdf   original pipeline under HopSkipJump: tree, SVM, random forest
  mlp_protocol.pdf       neural network: before-split vs training-only injection
Spread in the last three follows the report's ratio of seed means (eq. norm): the defect arm's mean over
the defect-free mean, with interval 1.96 s.d. / (sqrt(n) x defect-free mean); the defect-free point carries
its own interval. Other bands are 95% intervals over seeds.

Usage (from repo root):  uv run python defect_expansion_experiment/plot_legacy_redraw.py OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_style as rs

ROOT = Path(__file__).resolve().parents[1]; HERE = ROOT / 'defect_expansion_experiment'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'plots'; OUT.mkdir(parents=True, exist_ok=True)
ARMS = [('random', 'random'), ('gap', 'spatial')]


def norm_spread(base, cells):
    """Ratio of seed means and its interval for each cell; the defect-free point first."""
    b = np.nanmean(base); n = lambda v: np.sum(~np.isnan(v))
    m = [1.0] + [np.nanmean(v) / b if n(v) else np.nan for v in cells]
    h = [rs.ci(base) / b] + [1.96 * np.nanstd(v, ddof=1) / (np.sqrt(n(v)) * b) if n(v) > 1 else np.nan for v in cells]
    return m, h


def deletion_axes(ax, title, letter, xmax=95):
    rs.tidy(ax, title, letter); ax.set_xticks([0, 30, 60, 90]); ax.set_xlim(-4, xmax + 4)


# ---- 1. dataset/model dot-and-interval plot (original pipeline, train-only, 90% removal, paired per seed)
rows = []
for name, path in [('Tree, white-box', 'results_variance.parquet'), ('SVM, HopSkipJump', 'results_variance_svm.parquet')]:
    v = pl.read_parquet(HERE / path).filter(pl.col('protocol') == 'train_only')
    for ds, lab in [('iris', 'Iris (4 features)'), ('wine', 'Wine (13 features)')]:
        b = v.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.0)).select(['seed', pl.col('mean_dist').alias('b')])
        out = {}
        for role, st in ARMS:
            q = v.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.9) & (pl.col('structure') == st)).join(b, on='seed')
            r = (q['mean_dist'] / q['b']).to_numpy(); out[role] = (np.nanmean(r), rs.ci(r))
        rows.append((f'{lab}\n{name}', out))
fig, ax = plt.subplots(figsize=(rs.WIDTH, 3.2), layout='constrained')
for i, (lab, out) in enumerate(rows[::-1]):
    for role, dy in [('random', -0.12), ('gap', 0.12)]:
        rs.dot(ax, out[role][0], i + dy, out[role][1], role)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.tick_params(axis='y', length=0)
rs.refline(ax, x=1.0); rs.tidy(ax, ygrid=False)
ax.set_xlabel('Normalised spread at 90% removal (× defect-free)')
rs.legend(fig, ['gap', 'random'], whisker=True)
fig.savefig(OUT / 'dims_redrawn.pdf'); plt.close(fig)

# ---- 2. original pipeline, overfit tree + white-box attack (Table tab:holes): spread, recall, accuracy
im = pl.read_parquet(HERE / 'results_imbalance.parquet').filter((pl.col('tc') == 2) & (pl.col('feat') == 2))
fr = sorted(f for f in im['frac'].unique().to_list() if f > 0); x = 100 * np.array([0.0] + fr)
base = im.filter(pl.col('frac') == 0)
fig, axes = plt.subplots(1, 3, figsize=(rs.WIDTH, 3.0), sharex=True, layout='constrained')
for role, st in ARMS:
    cells = [im.filter((pl.col('frac') == f) & (pl.col('structure') == st)) for f in fr]
    rs.line(axes[0], x, *norm_spread(base['mean_dist'].to_numpy(), [c['mean_dist'].to_numpy() for c in cells]), role)
    for ax, col in [(axes[1], 'min_recall'), (axes[2], 'vacc')]:
        rs.draw(ax, x, [base[col].to_numpy()] + [c[col].to_numpy() for c in cells], role)
rs.refline(axes[0], 1.0)
for ax, (title, ylabel), letter in zip(axes, [('Spread', 'Normalised spread (× defect-free)'),
                                              ('Recall', 'Depleted-class recall'), ('Accuracy', 'Test accuracy')], 'abc'):
    deletion_axes(ax, title, letter); ax.set_ylabel(ylabel)
for ax in axes[1:]:
    ax.set_ylim(-0.03, 1.03)
fig.supxlabel('Target class removed (%)', fontsize=11)
rs.legend(fig, ['gap', 'random'])
fig.savefig(OUT / 'imbalance_original.pdf'); plt.close(fig)

# ---- 3. original pipeline under HopSkipJump (Table tab:models): columns tree, SVM, random forest
cols = ['model', 'structure', 'tc', 'frac', 'seed', 'recall_c2', 'mean_dist']
p1 = pl.concat([pl.read_parquet(HERE / 'results_p1.parquet').select(cols),
                pl.read_parquet(HERE / 'results_p1_rf.parquet').select(cols)], how='vertical_relaxed')
fig, axes = plt.subplots(2, 3, figsize=(rs.WIDTH, 4.8), sharex=True, layout='constrained')
for j, (model, title) in enumerate([('tree', 'Tree (10 seeds)'), ('svm', 'SVM (10 seeds)'), ('rf', 'Random forest (3 seeds)')]):
    q = p1.filter(pl.col('model') == model); b = q.filter(pl.col('frac') == 0)
    fr = sorted(f for f in q.filter(pl.col('tc') == 2)['frac'].unique().to_list()); x = 100 * np.array([0.0] + fr)
    for role, st in ARMS:
        cells = [q.filter((pl.col('tc') == 2) & (pl.col('frac') == f) & (pl.col('structure') == st)) for f in fr]
        rs.line(axes[0, j], x, *norm_spread(b['mean_dist'].to_numpy(), [c['mean_dist'].to_numpy() for c in cells]), role)
        rs.draw(axes[1, j], x, [b['recall_c2'].to_numpy()] + [c['recall_c2'].to_numpy() for c in cells], role)
    rs.refline(axes[0, j], 1.0)
    deletion_axes(axes[0, j], title, 'abc'[j]); deletion_axes(axes[1, j], title, 'def'[j])
rs.share_ylim(axes[0], 0.1)
for ax in axes[1]:
    ax.set_ylim(-0.03, 1.03)
axes[0, 0].set_ylabel('Normalised spread\n(× defect-free)'); axes[1, 0].set_ylabel('Depleted-class recall')
fig.supxlabel('Target class removed (%)', fontsize=11)
rs.legend(fig, ['gap', 'random'])
fig.savefig(OUT / 'imbalance_models.pdf'); plt.close(fig)

# ---- 4. neural network: before-split vs training-only coverage gap (Table tab:protocol)
mlp = pl.read_parquet(ROOT / 'mlp_experiment' / 'results.parquet').filter(
    (pl.col('defect') == 'coverage_gap') & (pl.col('attack') == 'hsj'))   # both attacks share the fitted model
fig, axes = plt.subplots(1, 2, figsize=(rs.WIDTH, 3.0), sharey=True, layout='constrained')
for role, protocol in [('before', 'before_split'), ('trainonly', 'train_only')]:
    q = mlp.filter(pl.col('protocol') == protocol); bv = sorted(b for b in q['bias'].unique().to_list() if b > 0)
    for ax, col in [(axes[0], 'vacc'), (axes[1], 'min_recall')]:
        rs.draw(ax, 100 * np.array(bv), [q.filter(pl.col('bias') == b)[col].to_numpy() for b in bv], role)
for ax, (title, ylabel), letter in zip(axes, [('Accuracy', 'Test accuracy'), ('Recall', 'Depleted-class recall')], 'ab'):
    rs.tidy(ax, title, letter); ax.set_xticks([10, 30, 50, 70, 90]); ax.set_ylabel(ylabel); ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_tick_params(labelleft=True)
fig.supxlabel('Target class removed (%)', fontsize=11)
rs.legend(fig, ['before', 'trainonly'])
fig.savefig(OUT / 'mlp_protocol.pdf'); plt.close(fig)
print('wrote', *(OUT / f for f in ['dims_redrawn.pdf', 'imbalance_original.pdf', 'imbalance_models.pdf', 'mlp_protocol.pdf']))
