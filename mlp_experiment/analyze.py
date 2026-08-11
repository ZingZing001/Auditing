"""Analysis for the MLP grid — answers H1-H7 from PLAN.md against results.parquet.

Spread is normalised PER SEED to that seed's own clean baseline (paired), which removes
between-seed variance in the baseline and is what makes 10 seeds enough to separate CIs.
1.0 = no effect, matching ../outlier_experiment convention.

Usage: python analyze.py
"""
from pathlib import Path
import numpy as np, polars as pl

HERE = Path(__file__).resolve().parent
DF = pl.read_parquet(HERE / 'results.parquet')

# Thread A reference values (normalised, strong cell k=8 n_out=10) for the H6 comparison
THREAD_A = {'tree d3 (DTA, white-box)': (1.36, 1.01), 'tree d10 (DTA, white-box)': (1.25, 1.00),
            'rbf-SVM (HSJ)': (0.98, 1.05), 'RandomForest (HSJ)': (0.92, 1.04)}


def ci95(v):
    """mean, half-width of the 95% CI. NaN-safe."""
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float(v[0]) if len(v) else np.nan), np.nan
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def cohens_d(a, b):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / s) if s > 0 else np.nan


def normalise(df, baseline):
    """Attach `norm` = mean_dist / that (attack, seed)'s clean baseline spread."""
    base = (baseline.group_by(['attack', 'seed']).agg(pl.col('mean_dist').mean().alias('_b')))
    return (df.join(base, on=['attack', 'seed'], how='left')
              .with_columns((pl.col('mean_dist') / pl.col('_b')).alias('norm')))


def fmt(m, h):
    return f'{m:.2f}x [{m - h:.2f},{m + h:.2f}]' if np.isfinite(h) else f'{m:.2f}x'


def verdict(m, h):
    if not np.isfinite(h):
        return '?'
    return 'SIGNAL' if m - h > 1.0 else ('compressed' if m + h < 1.0 else 'null')


# ───────────────────────────── coverage gap ─────────────────────────────
def coverage_gap():
    cg = DF.filter(pl.col('defect') == 'coverage_gap')
    if not len(cg):
        return
    clean = cg.filter(pl.col('bias') == 0.0)
    print('\n' + '=' * 78)
    print('COVERAGE GAP  —  H1 (signal survives a global smooth boundary), H2 (protocol)')
    print('=' * 78)
    for attack in sorted(cg['attack'].unique()):
        for proto in ['before_split', 'train_only']:
            sub = normalise(cg.filter((pl.col('attack') == attack) & (pl.col('bias') > 0) &
                                      (pl.col('protocol') == proto)), clean)
            if not len(sub):
                continue
            print(f'\n  {attack} / {proto}')
            print(f"    {'bias':>5} {'norm spread':>22} {'vacc':>6} {'min_recall':>11}  verdict")
            for b in sorted(sub['bias'].unique()):
                r = sub.filter(pl.col('bias') == b)
                m, h = ci95(r['norm'].to_list())
                va, _ = ci95(r['vacc'].to_list())
                mr, _ = ci95(r['min_recall'].to_list())
                print(f'    {b:>5} {fmt(m, h):>22} {va:>6.3f} {mr:>11.3f}  {verdict(m, h)}')
            lo = sub.filter(pl.col('bias') == sub['bias'].min())['mean_dist'].to_list()
            hi = sub.filter(pl.col('bias') == sub['bias'].max())['mean_dist'].to_list()
            va_all = [v for v in sub['vacc'].to_list() if np.isfinite(v)]
            print(f"    Cohen's d (lowest->highest bias) = {cohens_d(lo, hi):+.2f}"
                  f" | vacc range across bias = {max(va_all) - min(va_all):.3f}")
    print('\n  H2 read: before_split vacc range should be small (<=0.02, the accuracy-blind claim);')
    print('           train_only should drop materially. The MLP has no depth knob, so a drop here')
    print('           isolates PROTOCOL from the depth-3 -> overfit-tree switch made on 2026-08-11.')


# ───────────────────────────── outlier ─────────────────────────────
def outlier():
    ol = DF.filter(pl.col('defect') == 'outlier')
    if not len(ol):
        return
    clean = ol.filter(pl.col('n_out') == 0)
    inj = normalise(ol.filter(pl.col('n_out') > 0), clean)
    print('\n' + '=' * 78)
    print('OUTLIER  —  H6 (does a GLOBAL boundary see `outward`?), H7 (dose-response)')
    print('=' * 78)

    print('\n  Strong cell (k=8, n_out=10), normalised, 95% CI:')
    strong = inj.filter((pl.col('k') == 8.0) & (pl.col('n_out') == 10))
    print(f"    {'attack':>9} {'direction':>9} {'norm spread':>22} {'vacc':>6}  verdict")
    for attack in sorted(strong['attack'].unique()):
        for d in ['toward', 'outward', 'random']:
            r = strong.filter((pl.col('attack') == attack) & (pl.col('direction') == d))
            if not len(r):
                continue
            m, h = ci95(r['norm'].to_list())
            va, _ = ci95(r['vacc'].to_list())
            print(f'    {attack:>9} {d:>9} {fmt(m, h):>22} {va:>6.3f}  {verdict(m, h)}')

    print('\n  Reference — every model tested so far (same cell, from Thread A/B findings):')
    print(f"    {'model (attack)':>28} {'toward':>9} {'outward':>9}")
    for k, (tw, ow) in THREAD_A.items():
        print(f'    {k:>28} {tw:>8.2f}x {ow:>8.2f}x')
    print('    -> H6 fires if MLP `outward` CI clears 1.0; no other model does.')

    print('\n  H7 dose-response — normalised spread vs k (n_out=10):')
    for attack in sorted(inj['attack'].unique()):
        for d in ['toward', 'outward']:
            row = []
            for k in sorted(inj['k'].unique()):
                r = inj.filter((pl.col('attack') == attack) & (pl.col('direction') == d) &
                               (pl.col('k') == k) & (pl.col('n_out') == 10))
                m, h = ci95(r['norm'].to_list()) if len(r) else (np.nan, np.nan)
                row.append(f'k={k:.0f}:{m:.2f}' + ('*' if np.isfinite(h) and m - h > 1.0 else ''))
            print(f'    {attack:>9} {d:>8}  ' + '  '.join(row))
    print('    (* = 95% CI clears 1.0.)  Tree reference: fires only at k>=6, i.e. ~6 sigma.')
    print('    A global boundary reacting at k=2-3 would answer FINDINGS_phase05 F5,')
    print('    which flags realistic-magnitude outliers as invisible to trees.')


def main():
    print(f'rows={len(DF)}  cells={len(DF) // 2}  '
          f"nan_spread={int(DF['mean_dist'].is_nan().sum() + DF['mean_dist'].is_null().sum())}")
    print(f"attacks={sorted(DF['attack'].unique())}  defects={sorted(DF['defect'].unique())}")
    coverage_gap()
    outlier()
    print('\n' + '=' * 78)
    print('H4 (hang/convergence): asucc and nadv below — an MLP should not show the')
    print('flat-facet non-convergence that made RF/tree label noise intractable.')
    print('=' * 78)
    print(DF.group_by('attack').agg(
        pl.col('asucc').mean().round(3).alias('mean_attack_success'),
        pl.col('nadv').mean().round(1).alias('mean_n_adv'),
        pl.col('mean_dist').is_nan().sum().alias('nan_cells')).sort('attack'))


if __name__ == '__main__':
    main()
