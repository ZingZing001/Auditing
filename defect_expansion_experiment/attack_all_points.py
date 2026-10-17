"""Coverage gap vs random deletion with EVERY test point attacked (no correct-only filter).

The published pipeline attacks only test points the defective model classifies correctly, so the
attacked set shrinks and changes composition as the hole misclassifies its band. Here every test
point is attacked, as in the original project code: an untargeted attack moves each point across
the boundary relative to the model's CURRENT prediction, and a point counts as flipped if that
prediction changes. Every arm and severity therefore sees the same test points, and the hole's
misclassified points are kept.

Settings (target class, deletion feature):
  iris_pw   iris, virginica (2), petal width  (f3)
  iris_pl   iris, virginica (2), petal length (f2)
  wine      wine, class 0, proline (f12)
Models: overfit tree + white-box DecisionTreeAttack; RBF-SVM + HopSkipJump
  (norm=2, max_iter=10, max_eval=200, init_eval=50). Features standardised on each training fold (SCALE=full fits on all data, as a diagnostic only).
Train-only deletion via run_imbalance.inject_delete, matched count; spread via cluster_stats.
HopSkipJump is NOT reproducible run to run (ART seeds its initialisation from an unseeded
RandomState), so SVM numbers vary at the level of their confidence intervals.

Usage (from repo root):  uv run python defect_expansion_experiment/attack_all_points.py
Writes attack_all_points.csv next to this script and prints a summary.
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util, time, warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))    # 30
FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
SETTINGS = {'iris_pw': ('iris', 2, 3), 'iris_pl': ('iris', 2, 2), 'wine': ('wine', 0, 12)}
MODELS = ['tree', 'svm']
N_FOLDS = 5
SCALE = os.environ.get('SCALE', 'fold')   # 'fold' (default, correct) or 'full' (diagnostic: scaler fitted on all data)


def _imb():
    spec = importlib.util.spec_from_file_location('imb', HERE / 'run_imbalance.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def run_job(job):
    setting, model, seed = job
    import warnings; warnings.filterwarnings('ignore')
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from art.estimators.classification import SklearnClassifier
    from art.attacks.evasion import DecisionTreeAttack, HopSkipJump
    imb = _imb()
    ds, tc, feat = SETTINGS[setting]
    X, y = (load_iris if ds == 'iris' else load_wine)(return_X_y=True)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    keys = [('random', f) for f in FRACS] + [('spatial', f) for f in FRACS[1:]]
    acc = {k: dict(spread=[], recall=[], flipped=[], vacc=[]) for k in keys}
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr] if SCALE == 'fold' else X)
        Xt0, Xv, yt0, yv = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
        for st, f in keys:
            Xt, yt = imb.inject_delete(Xt0, yt0, tc, feat, f, st, rng)
            if model == 'tree':
                art = SklearnClassifier(DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt))
                art.fit(Xt, yoh.transform(yt.reshape(-1, 1)))
                attack = DecisionTreeAttack(classifier=art, verbose=False)
            else:
                art = SklearnClassifier(SVC(kernel='rbf', probability=True, random_state=42).fit(Xt, yt))
                attack = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
            pv = np.argmax(art.predict(Xv), axis=1)
            adv = attack.generate(Xv)                                  # every test point
            adv = adv[np.argmax(art.predict(adv), axis=1) != pv]       # flipped vs current prediction
            mm = yv == tc
            acc[(st, f)]['spread'].append(imb.cluster_stats(adv)[2] if len(adv) else np.nan)
            acc[(st, f)]['recall'].append(float((pv[mm] == tc).mean()) if mm.sum() else np.nan)
            acc[(st, f)]['flipped'].append(len(adv)); acc[(st, f)]['vacc'].append(float((pv == yv).mean()))
    return [dict(setting=setting, model=model, seed=seed, structure=st, frac=f,
                 **{m: float(np.nanmean(v)) for m, v in acc[(st, f)].items()}) for st, f in keys]


def summarise(df):
    ci = lambda a: 1.96 * np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))
    for model, setting in product(MODELS, SETTINGS):
        q = df.filter((pl.col('model') == model) & (pl.col('setting') == setting))
        base = q.filter(pl.col('frac') == 0).sort('seed')['spread'].to_numpy()
        print(f'\n{model} / {setting}   (attack all points; flipped/fold at 0%: '
              f'{q.filter(pl.col("frac") == 0)["flipped"].mean():.1f})')
        for f in FRACS[1:]:
            r = q.filter((pl.col('frac') == f) & (pl.col('structure') == 'random')).sort('seed')
            s = q.filter((pl.col('frac') == f) & (pl.col('structure') == 'spatial')).sort('seed')
            rr, ss = r['spread'].to_numpy() / base, s['spread'].to_numpy() / base; dd = ss - rr
            sep = '  SEPARATED' if np.nanmean(dd) - ci(dd) > 0 else ('  reversed' if np.nanmean(dd) + ci(dd) < 0 else '')
            print(f'  {int(f*100):>3}%  random {np.nanmean(rr):.3f}x  gap {np.nanmean(ss):.3f}x  '
                  f'gap-random {np.nanmean(dd):+.3f} ±{ci(dd):.3f} ({np.nanmean(dd)/np.nanmean(rr)*100:+.0f}% of random){sep}'
                  f'   recall random {r["recall"].mean():.2f} gap {s["recall"].mean():.2f}')


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(st, m, sd) for st in SETTINGS for m in MODELS for sd in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for i, out in enumerate(ex.map(run_job, jobs), 1):
            rows.extend(out)
            if i % 30 == 0:
                print(f'  [{i}/{len(jobs)}] {(time.time()-t0)/60:.1f} min', flush=True)
    df = pl.DataFrame(rows); df.write_csv(HERE / ('attack_all_points.csv' if SCALE == 'fold' else 'attack_all_points_fullscale.csv'))
    summarise(df)
    print(f'\nDONE in {(time.time()-t0)/60:.1f} min')
