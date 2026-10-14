"""Negative controls for the coverage-gap spread signal, run inside the run_imbalance.py harness.

Reuses run_imbalance.inject_delete + run_cell logic (overfit tree + white-box DTA, 5-fold,
train-only deletion, 30 seeds) and outlier_experiment.cluster_stats, unchanged.

Part 1  reproduce published cells of results_imbalance.parquet exactly (DTA is deterministic).
Part 2  controls, each scored as Cohen's d of spread (defect vs its own no-defect baseline):
  real        iris, virginica (tc=2), petal length       -> expect large +d
  separable   iris, setosa (tc=0), petal length          -> expect ~0
  synthetic   3 Gaussian blobs, 150 x 4, moderate overlap -> exploratory only, NOT reported:
              overlapping classes are contested, so a weak signal is expected here
  randlabel   iris with labels permuted per seed, tc=2   -> expect ~0

Reported in the final report, Section 4.1 ("The hole effect needs a contested boundary").
Usage (from repo root):  uv run python defect_expansion_experiment/crosscheck_controls.py
Writes crosscheck_reproduction.csv and crosscheck_controls.csv next to this script.
"""
import os, sys, time
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

WT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]   # repo root
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
imb = load('imb', WT / 'defect_expansion_experiment' / 'run_imbalance.py')
cluster_stats = imb.cluster_stats
SEEDS, N_FOLDS = imb.SEEDS, imb.N_FOLDS


def dataset(kind, seed):
    X, y = load_iris(return_X_y=True)
    if kind == 'synthetic':   # healthy: 3 isotropic Gaussian classes, iris-sized, moderate overlap
        X, y = make_blobs(n_samples=150, n_features=4, centers=3, cluster_std=1.5, random_state=seed)
    if kind == 'randlabel':
        y = np.random.default_rng(10_000 + seed).permutation(y)
    return X, y


def run_cell(kind, tc, feat, frac, structure, seed):
    X, y = dataset(kind, seed)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = imb.inject_delete(Xt, yt, tc, feat, frac, structure, rng)
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(m); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        mm = yv == tc
        rec = float((pv[mm] == tc).mean()) if mm.sum() else np.nan
        c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            adv = adv[np.argmax(art.predict(adv), axis=1) != yv[c]]
        else:
            adv = np.empty((0, X.shape[1]))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(vacc=vacc, min_recall=rec, nadv=len(adv), mean_dist=md))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(kind=kind, tc=tc, feat=feat, frac=frac, structure=structure, seed=seed)
    return r


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sp


if __name__ == '__main__':
    t0 = time.time(); rows = []
    # Part 1: reproduce published cells (frac 0 and 0.95, both arms, all 30 seeds)
    pub = pl.read_parquet(WT / 'defect_expansion_experiment' / 'results_imbalance.parquet')
    rep = []
    for f, s in [(0.0, 'random'), (0.95, 'random'), (0.95, 'spatial')]:
        for sd in SEEDS:
            r = run_cell('real', 2, 2, f, s, sd)
            p = pub.filter((pl.col('frac') == f) & (pl.col('structure') == s) & (pl.col('seed') == sd))
            rep.append(dict(frac=f, structure=s, seed=sd, ours=r['mean_dist'], published=float(p['mean_dist'][0]),
                            ours_rec=r['min_recall'], pub_rec=float(p['min_recall'][0])))
    rp = pl.DataFrame(rep)
    print('PART 1 reproduction: max |diff| spread =', float((rp['ours'] - rp['published']).abs().max()),
          '| max |diff| recall =', float((rp['ours_rec'] - rp['pub_rec']).abs().max()), f'({time.time()-t0:.0f}s)', flush=True)
    rp.write_csv(OUT / 'crosscheck_reproduction.csv')
    # Part 2: controls
    conds = [('real', 2, 2), ('separable', 0, 2), ('synthetic', 2, 0), ('randlabel', 2, 2)]
    for kind, tc, feat in conds:
        k = 'real' if kind == 'separable' else kind
        for f in [0.0, 0.85, 0.95]:
            for s in (['random'] if f == 0 else ['random', 'spatial']):
                for sd in SEEDS:
                    r = run_cell(k, tc, feat, f, s, sd); r['kind'] = kind; rows.append(r)
        print(f'  {kind} done ({(time.time()-t0)/60:.1f} min)', flush=True)
    df = pl.DataFrame(rows); df.write_csv(OUT / 'crosscheck_controls.csv')
    print('\nPART 2 controls (30 seeds, overfit tree + white-box DTA, train-only)')
    print(f'{"condition":<11}{"frac":>6}  {"base":>6} {"spat":>6} {"rand":>6}  {"d(gap vs base)":>15} {"d(gap vs rand)":>15}  {"rec base/gap":>14}')
    for kind, *_ in conds:
        b = df.filter((pl.col('kind') == kind) & (pl.col('frac') == 0.0))
        for f in [0.85, 0.95]:
            sp = df.filter((pl.col('kind') == kind) & (pl.col('frac') == f) & (pl.col('structure') == 'spatial'))
            ra = df.filter((pl.col('kind') == kind) & (pl.col('frac') == f) & (pl.col('structure') == 'random'))
            print(f'{kind:<11}{f:>6.2f}  {b["mean_dist"].mean():6.3f} {sp["mean_dist"].mean():6.3f} {ra["mean_dist"].mean():6.3f}'
                  f'  {cohens_d(sp["mean_dist"], b["mean_dist"]):15.2f} {cohens_d(sp["mean_dist"], ra["mean_dist"]):15.2f}'
                  f'  {b["min_recall"].mean():6.2f}/{sp["min_recall"].mean():.2f}')
    print(f'\nDONE in {(time.time()-t0)/60:.1f} min')
