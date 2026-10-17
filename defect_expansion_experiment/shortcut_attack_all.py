"""Shortcut feature with EVERY test point attacked and fold-fitted scaling.

Same defect as run_shortcut.py: append one feature equal to corr*(y-1) + N(0,1) in training and
pure N(0,1) at test time (iris: 5 inputs, wine: 14). Differences from run_shortcut.py:
  * every test point is attacked (flips counted against the model's current prediction), so the
    attacked set does not shrink as accuracy collapses;
  * the real features are standardised on each training fold only.
Records spur_frac (mean share of adversarial L2 displacement on the shortcut axis), scalar spread
(OPTICS mean pairwise distance) and test accuracy. Tree + white-box DTA (30 seeds) and RBF-SVM +
HopSkipJump (15 seeds; HSJ is not reproducible run to run, see attack_all_points.py).

Usage (from repo root):  uv run python defect_expansion_experiment/shortcut_attack_all.py
Writes shortcut_attack_all.csv next to this script and prints a summary.
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
CORR = [0.0, 1.0, 2.0, 4.0, 8.0]
N_FOLDS = 5


def run_job(job):
    dataset, model, seed = job
    import warnings; warnings.filterwarnings('ignore')
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from art.estimators.classification import SklearnClassifier
    from art.attacks.evasion import DecisionTreeAttack, HopSkipJump
    spec = importlib.util.spec_from_file_location('imb', HERE / 'run_imbalance.py')
    imb = importlib.util.module_from_spec(spec); spec.loader.exec_module(imb)
    X, y = (load_iris if dataset == 'iris' else load_wine)(return_X_y=True)
    d = X.shape[1]; spur = d; z = y - 1
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for corr in CORR:
        rng = np.random.default_rng(seed); acc = dict(spur_frac=[], spread=[], vacc=[], nflip=[])
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xt = np.column_stack([sc.transform(X[tr]), corr * z[tr] + rng.normal(0, 1, len(tr))])
            Xv = np.column_stack([sc.transform(X[te]), rng.normal(0, 1, len(te))])
            yt, yv = y[tr], y[te]
            if model == 'tree':
                art = SklearnClassifier(DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt))
                art.fit(Xt, yoh.transform(yt.reshape(-1, 1)))
                attack = DecisionTreeAttack(classifier=art, verbose=False)
            else:
                art = SklearnClassifier(SVC(kernel='rbf', probability=True, random_state=42).fit(Xt, yt))
                attack = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
            pv = np.argmax(art.predict(Xv), axis=1)
            adv = attack.generate(Xv)
            flip = np.argmax(art.predict(adv), axis=1) != pv
            adv, x0 = adv[flip], Xv[flip]
            if len(adv):
                dx = adv - x0; l2 = np.linalg.norm(dx, axis=1); nz = l2 > 1e-12
                acc['spur_frac'].append(float(np.mean((dx[nz, spur] ** 2) / (l2[nz] ** 2))) if nz.any() else np.nan)
                acc['spread'].append(imb.cluster_stats(adv)[2])
            else:
                acc['spur_frac'].append(np.nan); acc['spread'].append(np.nan)
            acc['vacc'].append(float((pv == yv).mean())); acc['nflip'].append(int(flip.sum()))
        rows.append(dict(dataset=dataset, model=model, seed=seed, corr=corr,
                         **{k: float(np.nanmean(v)) for k, v in acc.items()}))
    return rows


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(ds, 'tree', sd) for ds in ['iris', 'wine'] for sd in SEEDS] + \
           [(ds, 'svm', sd) for ds in ['iris', 'wine'] for sd in SEEDS[:15]]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for out in ex.map(run_job, jobs):
            rows.extend(out)
    df = pl.DataFrame(rows); df.write_csv(HERE / 'shortcut_attack_all.csv')
    ci = lambda a: 1.96 * np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))
    for model in ['tree', 'svm']:
        for ds in ['iris', 'wine']:
            q = df.filter((pl.col('model') == model) & (pl.col('dataset') == ds))
            b = q.filter(pl.col('corr') == 0).sort('seed')['spread'].to_numpy()
            print(f'\n{model} / {ds}  (n seeds {q["seed"].n_unique()})')
            for c in CORR:
                r = q.filter(pl.col('corr') == c).sort('seed')
                sf = r['spur_frac'].to_numpy(); sp = r['spread'].to_numpy() / b
                print(f'  corr {c:>3.0f}  shortcut share {np.nanmean(sf):.3f} ±{ci(sf):.3f}   spread {np.nanmean(sp):.3f}x ±{ci(sp):.3f}'
                      f'   acc {r["vacc"].mean():.3f}   flipped/fold {r["nflip"].mean():.1f}')
    print(f'\nDONE in {(time.time()-t0)/60:.1f} min')
