"""Negative controls with Aolin Yang's definitions, at 50% and 70% deletion (standard pipeline).

Same harness as crosscheck_controls.py (overfit tree + white-box DTA, train-only, 30 seeds, the
run_imbalance.py deletion and outlier_experiment cluster_stats). Conditions:
  contested  iris, virginica (tc=2), petal length
  setosa     iris, setosa (tc=0), petal length
  blobs      make_blobs(n_samples=150, centers=3, n_features=3, cluster_std=1.0,
             center_box=(-10, 10), random_state=0), class 2, feature 0   (Aolin's run_healthy_control.py)
  randgauss  150 x 4 standard-normal features, balanced random labels redrawn per seed, class 2, feature 0
             (his description: "Gaussian features with random labels"; exact generator not yet pushed)
Reports Cohen's d of spread for the gap against the no-defect baseline and against random deletion.
Poster values to check (50% / 70%): contested +2.23 / +1.80, blobs -0.26 / -0.29,
random labels +0.29 / +0.26, setosa -0.05 / -0.12.

Usage (from repo root):  uv run python defect_expansion_experiment/crosscheck_controls_aolin_defs.py
                         ATTACK_ALL=1 ... attacks every test point (flips vs current prediction)
"""
import os, time, importlib.util, warnings
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('imb', HERE / 'run_imbalance.py')
imb = importlib.util.module_from_spec(spec); spec.loader.exec_module(imb)
SEEDS, N_FOLDS = imb.SEEDS, imb.N_FOLDS
SKIPPED = [0]                                     # points the DTA could not attack (ART IndexError)
ATTACK_ALL = bool(os.environ.get('ATTACK_ALL'))   # env ATTACK_ALL=1: attack every test point
CONDS = {'contested': (2, 2), 'setosa': (0, 2), 'blobs': (2, 0), 'randgauss': (2, 0)}


def data(kind, seed):
    if kind == 'blobs':
        return make_blobs(n_samples=150, centers=3, n_features=3, cluster_std=1.0,
                          center_box=(-10, 10), random_state=0)
    if kind == 'randgauss':
        r = np.random.default_rng(20_000 + seed)
        return r.standard_normal((150, 4)), r.permutation(np.repeat([0, 1, 2], 50))
    return load_iris(return_X_y=True)


def run_cell(kind, frac, structure, seed):
    tc, feat = CONDS[kind]; X, y = data(kind, seed)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed); md = []
    for tr, te in skf.split(X, y):
        Xa, ya = imb.inject_delete(X[tr], y[tr], tc, feat, frac, structure, rng)
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(m); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        Xv, yv = X[te], y[te]; pv = np.argmax(art.predict(Xv), axis=1)
        c = np.ones(len(yv), bool) if ATTACK_ALL else (pv == yv)       # ATTACK_ALL: every test point
        if c.sum():
            atk = DecisionTreeAttack(classifier=art, verbose=False)
            try:
                adv = atk.generate(Xv[c]); keep = np.ones(int(c.sum()), bool)
            except IndexError:                                          # ART path-indexing bug on some points:
                outs, keep = [], []                                     # attack point by point, skip failures
                for x in Xv[c]:
                    try:
                        outs.append(atk.generate(x[None])[0]); keep.append(True)
                    except IndexError:
                        outs.append(x); keep.append(False)
                adv, keep = np.array(outs), np.array(keep)
            SKIPPED[0] += int((~keep).sum())
            adv = adv[keep & (np.argmax(art.predict(adv), axis=1) != pv[c])]   # flip = change of current prediction
            md.append(imb.cluster_stats(adv)[2] if len(adv) else np.nan)
    return dict(kind=kind, frac=frac, structure=structure, seed=seed, mean_dist=float(np.nanmean(md)),
                skipped_total_so_far=SKIPPED[0])


def d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sp


if __name__ == '__main__':
    t0 = time.time()
    rows = [run_cell(k, f, st, sd) for k in CONDS for f in [0.0, 0.5, 0.7]
            for st in (['random'] if f == 0 else ['random', 'spatial']) for sd in SEEDS]
    df = pl.DataFrame(rows); df.write_csv(HERE / ('crosscheck_controls_aolin_defs' + ('_attack_all' if ATTACK_ALL else '') + '.csv'))
    print(f'{"condition":<10}{"frac":>6}  {"d vs baseline":>14} {"d vs random":>12}')
    for k in CONDS:
        b = df.filter((pl.col('kind') == k) & (pl.col('frac') == 0))['mean_dist']
        for f in [0.5, 0.7]:
            sp = df.filter((pl.col('kind') == k) & (pl.col('frac') == f) & (pl.col('structure') == 'spatial'))['mean_dist']
            ra = df.filter((pl.col('kind') == k) & (pl.col('frac') == f) & (pl.col('structure') == 'random'))['mean_dist']
            print(f'{k:<10}{f:>6.2f}  {d(sp, b):14.2f} {d(sp, ra):12.2f}')
    print(f'points skipped by DTA IndexError: {SKIPPED[0]}')
    (HERE / ('crosscheck_controls_aolin_defs' + ('_attack_all' if ATTACK_ALL else '') + '_skipped.txt')).write_text(f'{SKIPPED[0]}\n')
    print(f'DONE in {(time.time()-t0)/60:.1f} min')
