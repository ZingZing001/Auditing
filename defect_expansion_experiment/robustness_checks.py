"""Robustness checks for the coverage-gap spread signal (overfit tree + white-box DTA, 30 seeds).

Reuses inject_delete (run_imbalance.py) and cluster_stats (outlier_experiment), unchanged.

Check 1  fold-fitted standardisation. The cross-dataset grid (run_variance.py) standardised
         features on the full dataset before splitting. Here the scaler is fitted on each
         training fold only. Same targets/features as that grid: iris virginica along petal
         width (f3), wine class 0 along proline (f12); train-only; frac 0 and 0.9.
Check 2  fixed attack cohort + stage counts. The standard pipeline attacks only test points
         the (defective) model classifies correctly, so the attacked set changes with severity.
         Here, per fold, every severity's model is trained first and only test points that ALL
         of them classify correctly are attacked, so the cohort is identical across severities.
         Stage counts (test, correct=attacked, successful, clustered) are logged for the
         standard pipeline. Iris virginica along petal length (Table 1 setting).
Check 3  one cohort shared by BOTH arms and all severities, compared paired per seed. This is
         the clean version of check 2 (whose cohorts differ between the spatial and random arms).

Usage (from repo root):  uv run python defect_expansion_experiment/robustness_checks.py
Writes robustness_standardise.csv, robustness_cohort.csv and robustness_joint_cohort.csv next to this script.
"""
import os, sys, time
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
imb = load('imb', HERE / 'run_imbalance.py')
inject_delete, cluster_stats, SEEDS, N_FOLDS = imb.inject_delete, imb.cluster_stats, imb.SEEDS, imb.N_FOLDS


def fit(Xt, yt, yoh):
    m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
    art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1, 1))); return art


def attack(art, X, y):
    """Attack X (assumed correctly classified); return successful adversarial points."""
    if len(X) == 0:
        return np.empty((0, X.shape[1]))
    adv = DecisionTreeAttack(classifier=art, verbose=False).generate(X)
    return adv[np.argmax(art.predict(adv), axis=1) != y]


def clustered(adv):
    cs = cluster_stats(adv) if len(adv) else (np.nan,) * 5
    n = 0 if np.isnan(cs[1]) or cs[1] == 0 else int(round(cs[1] * cs[3]))
    return cs[2], n


# ------------------------------------------------------------------ check 1: fold-fitted scaler
CFG = {'iris': (load_iris, 2, 3), 'wine': (load_wine, 0, 12)}

def cell_std(ds, structure, frac, seed):
    loader, tc, feat = CFG[ds]
    X, y = loader(return_X_y=True)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed); out = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])                     # training fold only
        Xt, Xv, yt, yv = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
        Xt, yt = inject_delete(Xt, yt, tc, feat, frac, structure, rng)
        art = fit(Xt, yt, yoh)
        pv = np.argmax(art.predict(Xv), axis=1); c = pv == yv; mm = yv == tc
        md, _ = clustered(attack(art, Xv[c], yv[c]))
        out.append(dict(mean_dist=md, min_recall=float((pv[mm] == tc).mean()) if mm.sum() else np.nan))
    r = {k: float(np.nanmean([o[k] for o in out])) for k in out[0]}
    r.update(dataset=ds, structure=structure, frac=frac, seed=seed); return r


# ------------------------------------------------------------------ check 2: fixed cohort
FRACS = [0.0] + [float(f) for f in os.environ.get('FRACS', '0.5,0.85,0.95').split(',')]   # env FRACS overrides severities

def cell_cohort(structure, seed, tc=2, feat=2):
    X, y = load_iris(return_X_y=True)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed); rows = {f: [] for f in FRACS}
    for tr, te in skf.split(X, y):
        Xt0, Xv, yt0, yv = X[tr], X[te], y[tr], y[te]
        models = {}
        for f in FRACS:
            Xt, yt = inject_delete(Xt0, yt0, tc, feat, f, structure, rng)
            models[f] = fit(Xt, yt, yoh)
        preds = {f: np.argmax(m.predict(Xv), axis=1) for f, m in models.items()}
        cohort = np.all([preds[f] == yv for f in FRACS], axis=0)
        for f, m in models.items():
            c = preds[f] == yv
            adv_std = attack(m, Xv[c], yv[c]); md_std, ncl = clustered(adv_std)
            adv_fix = attack(m, Xv[cohort], yv[cohort]); md_fix, _ = clustered(adv_fix)
            rows[f].append(dict(n_test=len(yv), n_attacked=int(c.sum()), n_success=len(adv_std), n_clustered=ncl,
                                spread_std=md_std, n_cohort=int(cohort.sum()), spread_fixed=md_fix))
    return [dict({k: float(np.nanmean([r[k] for r in rows[f]])) for k in rows[f][0]},
                 frac=f, structure=structure, seed=seed) for f in FRACS]


# ------------------------------------------------------------------ check 3: one cohort shared by both arms
def cell_joint(seed, tc=2, feat=2):
    """Train every (structure, frac) model per fold; attack only test points that ALL of them
    classify correctly, so spatial and random arms and all severities share one cohort."""
    X, y = load_iris(return_X_y=True)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed); keys = [('random', f) for f in FRACS] + [('spatial', f) for f in FRACS[1:]]
    acc = {k: [] for k in keys}; ncoh = []
    for tr, te in skf.split(X, y):
        models = {}
        for st, f in keys:
            Xt, yt = inject_delete(X[tr], y[tr], tc, feat, f, st, rng)
            models[(st, f)] = fit(Xt, yt, yoh)
        yv, Xv = y[te], X[te]
        cohort = np.all([np.argmax(m.predict(Xv), axis=1) == yv for m in models.values()], axis=0)
        ncoh.append(int(cohort.sum()))
        for k, m in models.items():
            acc[k].append(clustered(attack(m, Xv[cohort], yv[cohort]))[0])
    return dict({f'{st}_{f}': float(np.nanmean(v)) for (st, f), v in acc.items()}, seed=seed, n_cohort=float(np.mean(ncoh)))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sp


if __name__ == '__main__':
    t0 = time.time()
    s1 = pl.DataFrame([cell_std(ds, st, f, sd) for ds in CFG for f in [0.0, 0.9]
                       for st in (['random'] if f == 0 else ['random', 'spatial']) for sd in SEEDS])
    s1.write_csv(HERE / 'robustness_standardise.csv')
    print(f'CHECK 1  fold-fitted standardisation (frac 0.9, train-only, 30 seeds)  [{time.time()-t0:.0f}s]')
    for ds in CFG:
        b = s1.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0))['mean_dist'].mean()
        for st in ['random', 'spatial']:
            q = s1.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.9) & (pl.col('structure') == st))
            print(f'  {ds:5s} {st:8s} spread {q["mean_dist"].mean()/b:.3f}x   recall {q["min_recall"].mean():.3f}')
        sp = s1.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.9) & (pl.col('structure') == 'spatial'))['mean_dist']
        ra = s1.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.9) & (pl.col('structure') == 'random'))['mean_dist']
        print(f'  {ds:5s} d(spatial vs random) {cohens_d(sp, ra):+.2f}')
    s2 = pl.DataFrame([r for st in ['random', 'spatial'] for sd in SEEDS for r in cell_cohort(st, sd)])
    s2.write_csv(HERE / 'robustness_cohort.csv')
    print(f'\nCHECK 2  fixed cohort vs standard pipeline (iris, virginica, petal length, 30 seeds)  [{time.time()-t0:.0f}s]')
    print(f'{"structure":<9}{"frac":>5} {"test":>5} {"attacked":>9} {"success":>8} {"clustered":>10} {"cohort":>7} {"std x":>7} {"fixed x":>8}')
    for st in ['random', 'spatial']:
        q0 = s2.filter((pl.col('structure') == st) & (pl.col('frac') == 0))
        for f in FRACS:
            q = s2.filter((pl.col('structure') == st) & (pl.col('frac') == f))
            print(f'{st:<9}{f:>5.2f} {q["n_test"].mean():5.1f} {q["n_attacked"].mean():9.1f} {q["n_success"].mean():8.1f} '
                  f'{q["n_clustered"].mean():10.1f} {q["n_cohort"].mean():7.1f} '
                  f'{q["spread_std"].mean()/q0["spread_std"].mean():7.3f} {q["spread_fixed"].mean()/q0["spread_fixed"].mean():8.3f}')
    for f in [0.85, 0.95]:
        sp = s2.filter((pl.col('structure') == 'spatial') & (pl.col('frac') == f))['spread_fixed']
        ra = s2.filter((pl.col('structure') == 'random') & (pl.col('frac') == f))['spread_fixed']
        print(f'fixed cohort, frac {f}: d(spatial vs random) {cohens_d(sp, ra):+.2f}')

    s3 = pl.DataFrame([cell_joint(sd) for sd in SEEDS]); s3.write_csv(HERE / ('robustness_joint_cohort' + os.environ.get('OUT_SUFFIX', '') + '.csv'))
    print(f'\nCHECK 3  one cohort shared by both arms (mean cohort {s3["n_cohort"].mean():.1f} of 30 test points)')
    base = s3['random_0.0'].to_numpy()
    for f in FRACS[1:]:
        r = s3[f'random_{f}'].to_numpy() / base; sp = s3[f'spatial_{f}'].to_numpy() / base; dd = sp - r
        ci = lambda a: 1.96 * a.std(ddof=1) / np.sqrt(len(a))
        print(f'  frac {f:.2f}  random {r.mean():.3f}x ±{ci(r):.3f}   spatial {sp.mean():.3f}x ±{ci(sp):.3f}   '
              f'paired spatial-random {dd.mean():+.3f} ±{ci(dd):.3f}   d {cohens_d(sp, r):+.2f}')
    print(f'\nDONE in {(time.time()-t0)/60:.1f} min')
