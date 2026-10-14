"""Fixed-cohort check under a black-box attack: RBF-SVM + HopSkipJump (run_p1.py setting).

Iris, virginica (tc=2) deleted along petal length (f2), train-only, matched count, 10 seeds,
SVC(rbf, probability=True, random_state=42), HSJ(norm=2, max_iter=10, max_eval=200, init_eval=50).
Per fold every (structure, frac) model is trained first. Two scores per model:
  standard  attack the test points THAT model classifies correctly (the published pipeline)
  fixed     attack only the test points ALL models classify correctly (one cohort shared by
            both arms and all severities)
HSJ is seeded per (seed, fold, model) so the run is reproducible. Seeds run in parallel.

Usage (from repo root):  uv run python defect_expansion_experiment/robustness_svm_cohort.py
Writes robustness_svm_cohort.csv next to this script.
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + (list(range(300, 320)) if os.environ.get('SEEDS30') else [])
FRACS = [0.0] + [float(f) for f in os.environ.get('FRACS', '0.5,0.85,0.95').split(',')]   # env FRACS overrides severities
TC, FEAT, N_FOLDS = 2, 2, 5
KEYS = [('random', f) for f in FRACS] + [('spatial', f) for f in FRACS[1:]]


def _lib():
    spec = importlib.util.spec_from_file_location('imb', HERE / 'run_imbalance.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def run_seed(seed):
    import warnings; warnings.filterwarnings('ignore')
    from art.estimators.classification import SklearnClassifier
    from art.attacks.evasion import HopSkipJump
    imb = _lib()
    X, y = load_iris(return_X_y=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    acc = {(k, mode): [] for k in KEYS for mode in ('std', 'fixed')}
    cnt = {k: [] for k in KEYS}; ncoh = []

    def spread(art, Xa, ya, tag):
        if len(Xa) == 0:
            return np.nan, 0
        np.random.seed(tag)
        adv = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50,
                          verbose=False).generate(Xa)
        adv = adv[np.argmax(art.predict(adv), axis=1) != ya]
        return (imb.cluster_stats(adv)[2] if len(adv) else np.nan), len(adv)

    for fi, (tr, te) in enumerate(skf.split(X, y)):
        models = {}
        for st, f in KEYS:
            Xt, yt = imb.inject_delete(X[tr], y[tr], TC, FEAT, f, st, rng)
            models[(st, f)] = SklearnClassifier(SVC(kernel='rbf', probability=True, random_state=42).fit(Xt, yt))
        Xv, yv = X[te], y[te]
        preds = {k: np.argmax(m.predict(Xv), axis=1) for k, m in models.items()}
        cohort = np.all([p == yv for p in preds.values()], axis=0); ncoh.append(int(cohort.sum()))
        for ki, (k, m) in enumerate(models.items()):
            c = preds[k] == yv
            s_std, nsucc = spread(m, Xv[c], yv[c], seed * 1000 + fi * 10 + ki)
            s_fix, _ = spread(m, Xv[cohort], yv[cohort], seed * 1000 + fi * 10 + ki + 500)
            acc[(k, 'std')].append(s_std); acc[(k, 'fixed')].append(s_fix)
            cnt[k].append((int(c.sum()), nsucc))
    row = {'seed': seed, 'n_cohort': float(np.mean(ncoh))}
    for (st, f), mode in acc:
        row[f'{mode}_{st}_{f}'] = float(np.nanmean(acc[((st, f), mode)]))
    for (st, f), v in cnt.items():
        row[f'attacked_{st}_{f}'] = float(np.mean([a for a, _ in v])); row[f'success_{st}_{f}'] = float(np.mean([s for _, s in v]))
    return row


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sp


if __name__ == '__main__':
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count() or 4)) as ex:
        rows = list(ex.map(run_seed, SEEDS))
    df = pl.DataFrame(rows); df.write_csv(HERE / ('robustness_svm_cohort' + os.environ.get('OUT_SUFFIX', '') + '.csv'))
    ci = lambda a: 1.96 * np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))
    print(f'SVM + HopSkipJump, iris virginica / petal length, {len(SEEDS)} seeds, {(time.time()-t0)/60:.1f} min; '
          f'mean shared cohort {df["n_cohort"].mean():.1f} of 30')
    for mode in ('std', 'fixed'):
        base = df[f'{mode}_random_0.0'].to_numpy()
        print(f'\n[{mode}]' + ('  standard pipeline (published)' if mode == 'std' else '  one cohort shared by both arms'))
        for f in FRACS[1:]:
            r = df[f'{mode}_random_{f}'].to_numpy() / base; s = df[f'{mode}_spatial_{f}'].to_numpy() / base; dd = s - r
            extra = (f'   attacked random/spatial {df[f"attacked_random_{f}"].mean():.1f}/{df[f"attacked_spatial_{f}"].mean():.1f}'
                     if mode == 'std' else '')
            print(f'  frac {f:.2f}  random {np.nanmean(r):.3f}x ±{ci(r):.3f}   spatial {np.nanmean(s):.3f}x ±{ci(s):.3f}'
                  f'   paired spatial-random {np.nanmean(dd):+.3f} ±{ci(dd):.3f}   d {cohens_d(s, r):+.2f}{extra}')
