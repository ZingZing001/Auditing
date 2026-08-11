"""MLP + coverage gap + outlier, both attacks — see PLAN.md.

Model:   torch MLP (4-16-16-3, ReLU) wrapped in ART PyTorchClassifier. The FOURTH quadrant of
         the project's model taxonomy: a GLOBALLY-parameterised smooth boundary (tree/RF are
         local+flat, rbf-SVM is local+smooth). Thread A's outlier null was explained by rbf
         LOCALITY, so a global boundary is the outstanding test.
Attacks: HSJ (black-box, config IDENTICAL to every other grid) and DeepFool (white-box).
         BOTH attack the SAME fitted model per fold — that is what makes the white-box/black-box
         contrast clean (PLAN.md §2, §8).
Metric:  OPTICS within-cluster mean pairwise distance (spread), measured AFTER inverse-transform
         back to RAW iris units so it is comparable to the tree (~0.44) / SVM (~0.49) baselines.
         `spread_std_space` keeps the standardised-space value as a control (PLAN.md §1).

Defects:
  coverage_gap — spatial deletion, run under BOTH injection protocols (PLAN.md §3):
      before_split : inject into the full dataset, then CV-split   (grids 1-3, model_family)
      train_only   : CV-split first, inject into the train fold    (outlier, defect_expansion)
    The same deletion function is applied at two different points, so only WHEN varies.
  outlier — correctly-labeled points at k*class_std from the class centroid, train-fold only
    (Thread A's protocol). Reuses inject_outliers verbatim so results sit directly alongside
    ../outlier_experiment/FINDINGS_phase05.md and ../model_family_experiment/FINDINGS_outlier.md.

Usage:
  python run_experiment.py --smoke                 # pipeline check + timing, writes nothing
  python run_experiment.py --defect coverage_gap   # one defect
  python run_experiment.py --defect outlier
  python run_experiment.py                         # everything (resumable)
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import sys, time, warnings
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import HopSkipJump, DeepFool
import importlib.util

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results.parquet'
PROGRESS = HERE / 'progress.txt'


def _load(name, relpath):
    """Reuse by explicit path — several files in this repo are named run_experiment.py."""
    spec = importlib.util.spec_from_file_location(name, HERE.parent / relpath)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


_ol = _load('outlier_lib', 'outlier_experiment/run_experiment.py')
_mf = _load('modelfamily_lib', 'model_family_experiment/run_experiment.py')
cluster_stats = _ol.cluster_stats          # identical metric to every other grid
inject_outliers = _ol.inject_outliers      # identical injection to Thread A
inject_cg = _mf.inject_cg                  # identical spatial deletion to the headline CG grids

# ---- shared ----
TC = 2                       # virginica (contested); tc sweep is Phase 1
FEAT = 2                     # petal length — matches model_family_experiment
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777]     # 10 — n=3 flipped a conclusion before
N_FOLDS = 5
ATTACKS = ['hsj', 'deepfool']

# ---- coverage-gap grid (PLAN.md §4) ----
CG_BIAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
PROTOCOLS = ['before_split', 'train_only']

# ---- outlier grid — mirrors model_family OL_K/OL_N/OL_DIR so rows are directly comparable ----
OL_K = [2, 4, 6, 8]
OL_N = [5, 10]
OL_DIR = ['toward', 'outward', 'random']
# --ceiling: push the dose toward and past the class-size ceiling at k=8, the cell Thread A used.
# The train fold holds 40 points/class, so these are 50 / 75 / 100 / 150% of the target class.
# {20,30,40} mirror model_family's OL_N_CEILING exactly for comparability; 60 goes past anything
# tested so far. Motivation: rbf-SVM was null at low dose but rose to 1.09x at the ceiling with
# accuracy degrading 0.97->0.89 (FINDINGS_outlier.md F3) — the MLP is the other global model.
OL_N_CEILING = [20, 30, 40, 60]
OL_K_CEILING = 8.0
# --dose: FIX the distance, sweep the count. Motivated by F5 (FINDINGS_phase0.md): the MLP's only
# real response is at NEAR field and runs DOWNWARD (compression), not upward -- deepfool `toward`
# gives 0.85x at k=2, 0.96x at k=4, 1.01x at k=8, and is already dose-dependent at k=2
# (n=5 -> 0.90x, n=10 -> 0.85x). k=2 puts the injected points 0.88 cm out, 55% of the
# virginica->versicolor centroid gap, i.e. INSIDE versicolor: contamination, not an anomaly.
# k=4 (109% of the gap, just past versicolor's centre) is the contrast arm. `outward` at matched
# dose is the null control -- it isolates "points in the contested overlap" from "more points".
K_DOSE = [2.0, 4.0]
N_DOSE = [1, 3, 5, 10, 20, 30, 40, 60]
DIR_DOSE = ['toward', 'outward']

# ---- model + attack config (PLAN.md §2) ----
HIDDEN, EPOCHS, LR = (16, 16), 300, 1e-2
HSJ_KW = dict(norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
# DeepFool, NOT PGD, is the white-box arm. This project's metric is the geometry of
# MINIMUM-NORM adversarials: DTA walks to the nearest different leaf, HSJ binary-searches the
# nearest L2 crossing. PGD with a fixed eps is a *budget* attack — it stops anywhere inside the
# eps-ball, so its spread measures the budget, not the boundary (smoke test: perturbation 3.32
# against eps=3.0, i.e. saturated). DeepFool takes the minimal linearised step across the nearest
# boundary, making it the gradient analogue of DTA and directly comparable to every existing row.
DF_KW = dict(max_iter=100, epsilon=1e-6, nb_grads=3, batch_size=32, verbose=False)

COLS = ['defect', 'model', 'attack', 'protocol', 'bias', 'k', 'n_out', 'direction',
        'tc', 'feat', 'seed', 'tacc', 'vacc', 'min_recall', 'asucc', 'nadv', 'density',
        'nclust', 'mean_dist', 'clust_size', 'aiden_density', 'spread_std_space',
        'perturbation', 'spread_c0', 'spread_c1', 'spread_c2']
KEYCOLS = ['defect', 'protocol', 'attack', 'bias', 'k', 'n_out', 'direction', 'tc', 'feat', 'seed']


class MLP(nn.Module):
    def __init__(self, n_in=4, hidden=HIDDEN, n_out=3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))      # logits; CrossEntropyLoss applies softmax
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # ART's attack loops (HSJ binary search, DeepFool projection) hand back float64 arrays
        # even when fed float32, so cast at the boundary rather than chasing every call site.
        return self.net(x.to(self.net[0].weight.dtype))


def fit_mlp(Xs, y):
    """Deterministic fit on standardised features. Returns an ART-wrapped classifier.

    clip_values is left None to match the SklearnClassifier-based grids: HopSkipJump then infers
    its sampling bounds from the data (hop_skip_jump.py:164), exactly as it does for tree/SVM/RF.
    """
    torch.manual_seed(42)
    m = MLP(n_in=Xs.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossfn = nn.CrossEntropyLoss()
    xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    m.train()
    for _ in range(EPOCHS):                         # full-batch; iris is 150 rows
        opt.zero_grad()
        lossfn(m(xt), yt).backward()
        opt.step()
    m.eval()
    return PyTorchClassifier(model=m, loss=lossfn, optimizer=opt, input_shape=(Xs.shape[1],),
                             nb_classes=3, clip_values=None)


def run_attack(art_clf, Xv_s, yv, which):
    """Attack only correctly-classified test points; keep only successful flips.
    Returns (adv_scaled, x_orig_scaled, true_labels, attack_success_rate)."""
    p = np.argmax(art_clf.predict(Xv_s), axis=1)
    c = p == yv
    if c.sum() == 0:
        e = np.empty((0, Xv_s.shape[1]), dtype=np.float32)
        return e, e, np.empty(0, dtype=yv.dtype), 0.0
    x0 = Xv_s[c].astype(np.float32)
    atk = (HopSkipJump(classifier=art_clf, **HSJ_KW) if which == 'hsj'
           else DeepFool(classifier=art_clf, **DF_KW))
    adv = atk.generate(x0)
    fl = np.argmax(art_clf.predict(adv), axis=1) != yv[c]
    return adv[fl], x0[fl], yv[c][fl], float(c.mean())


def run_cell(X, y, job):
    """One grid cell -> ONE row per attack, sharing identical weights (PLAN.md §8)."""
    defect, bias, protocol = job['defect'], job['bias'], job['protocol']
    if defect == 'coverage_gap' and protocol == 'before_split' and bias > 0:
        np.random.seed(job['seed'])
        Xg, yg = inject_cg(X, y, TC, FEAT, bias)
    else:
        Xg, yg = X, y

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=job['seed'])
    rng = np.random.default_rng(job['seed'])
    folds = {a: [] for a in ATTACKS}
    for tr, te in skf.split(Xg, yg):
        Xt, Xv, yt, yv = Xg[tr], Xg[te], yg[tr], yg[te]
        if defect == 'coverage_gap' and protocol == 'train_only' and bias > 0:
            Xt, yt = inject_cg(Xt, yt, TC, FEAT, bias)      # same deletion, applied post-split
        elif defect == 'outlier' and job['n_out'] > 0:
            Xt, yt = inject_outliers(Xt, yt, TC, job['k'], job['n_out'],
                                     direction=job['direction'], rng=rng)

        sc = StandardScaler().fit(Xt)                        # train fold only — no leakage
        # float32 throughout: the torch model's weights are float32 and ART passes arrays
        # straight through, so a float64 array raises a dtype mismatch inside nn.Linear.
        Xt_s = sc.transform(Xt).astype(np.float32)
        Xv_s = sc.transform(Xv).astype(np.float32)
        art_clf = fit_mlp(Xt_s, yt)                          # ONE model, both attacks below

        tacc = float((np.argmax(art_clf.predict(Xt_s), axis=1) == yt).mean())
        pv = np.argmax(art_clf.predict(Xv_s), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == TC
        min_recall = float((pv[mmask] == TC).mean()) if mmask.sum() else np.nan

        for a in ATTACKS:
            adv_s, x0_s, labels, succ = run_attack(art_clf, Xv_s, yv, a)
            # back to RAW iris units so spread is comparable across the whole project (PLAN §1)
            adv = sc.inverse_transform(adv_s) if len(adv_s) else adv_s
            x0 = sc.inverse_transform(x0_s) if len(x0_s) else x0_s
            dens, nc, md, cs, aiden = cluster_stats(adv)
            rec = {'tacc': tacc, 'vacc': vacc, 'min_recall': min_recall, 'asucc': succ,
                   'nadv': len(adv), 'density': dens, 'nclust': nc, 'mean_dist': md,
                   'clust_size': cs, 'aiden_density': aiden,
                   'spread_std_space': cluster_stats(adv_s)[2] if len(adv_s) else np.nan,
                   'perturbation': (float(np.linalg.norm(adv - x0, axis=1).mean())
                                    if len(adv) else np.nan)}
            for cl in (0, 1, 2):
                rec[f'spread_c{cl}'] = cluster_stats(adv[labels == cl])[2] if len(adv) else np.nan
            folds[a].append(rec)

    rows = []
    for a in ATTACKS:
        r = {k: float(np.nanmean([f[k] for f in folds[a]])) for k in folds[a][0]}
        r.update(job); r.update(model='mlp', attack=a, tc=int(TC), feat=int(FEAT))
        rows.append(r)
    return rows


# ─────────────────────────── grid ───────────────────────────
def blank(**kw):
    r = {'defect': '', 'protocol': 'none', 'bias': -1.0, 'k': -1.0, 'n_out': -1,
         'direction': 'none', 'seed': -1}
    r.update(kw); return r


def build_grid():
    jobs = []
    for s in SEEDS:
        for b, p in product(CG_BIAS, PROTOCOLS):
            if b == 0.0 and p == 'train_only':
                continue                                  # bias=0 is protocol-independent
            jobs.append(blank(defect='coverage_gap', protocol=p, bias=round(b, 2), seed=s))
        # outliers are always train-fold-only (Thread A protocol)
        jobs.append(blank(defect='outlier', protocol='train_only', k=0.0, n_out=0, seed=s))
        for d, k, n in product(OL_DIR, OL_K, OL_N):
            jobs.append(blank(defect='outlier', protocol='train_only', k=float(k),
                              n_out=n, direction=d, seed=s))
        if '--ceiling' in sys.argv:
            for d, n in product(OL_DIR, OL_N_CEILING):
                jobs.append(blank(defect='outlier', protocol='train_only', k=OL_K_CEILING,
                                  n_out=n, direction=d, seed=s))
        if '--dose' in sys.argv:
            for d, k, n in product(DIR_DOSE, K_DOSE, N_DOSE):
                jobs.append(blank(defect='outlier', protocol='train_only', k=k,
                                  n_out=n, direction=d, seed=s))
    return jobs


def key(r):
    return tuple(round(float(r[c]), 2) if c in ('bias', 'k') else r[c] for c in KEYCOLS)


def done_keys():
    if not OUT.exists():
        return set()
    return {key(row) for row in pl.read_parquet(OUT).select(KEYCOLS).to_dicts()}


def append_rows(rows):
    new = pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows])
    merged = pl.concat([pl.read_parquet(OUT), new]) if OUT.exists() else new
    merged.write_parquet(OUT)


# ─────────────────────────── Phase 0a: smoke ───────────────────────────
def smoke():
    """Pipeline check: clean MLP accuracy, both attacks produce adversarials, H5 baseline sanity."""
    X, y = load_iris(return_X_y=True)
    print('=== Phase 0a: pipeline check (seed=42) ===', flush=True)
    cases = [('clean          ', blank(defect='coverage_gap', protocol='before_split', bias=0.0, seed=42)),
             ('outlier toward ', blank(defect='outlier', protocol='train_only', k=8.0, n_out=10,
                                       direction='toward', seed=42)),
             ('outlier outward', blank(defect='outlier', protocol='train_only', k=8.0, n_out=10,
                                       direction='outward', seed=42))]
    t0 = time.time()
    base = {}
    for name, job in cases:
        rows = run_cell(X, y, job)
        for r in rows:
            norm = f" norm={r['mean_dist'] / base[r['attack']]:.2f}x" if r['attack'] in base else ''
            print(f"  {name} {r['attack']:>8}: vacc={r['vacc']:.3f} asucc={r['asucc']:.3f} "
                  f"nadv={r['nadv']:.1f} spread(raw)={r['mean_dist']:.3f} "
                  f"pert={r['perturbation']:.3f}{norm}", flush=True)
        if name.startswith('clean'):
            base = {r['attack']: r['mean_dist'] for r in rows}
    dt = (time.time() - t0) / len(cases)
    print(f"\n  ~{dt:.1f}s/cell -> full grid ({len(build_grid())} cells) "
          f"~{dt * len(build_grid()) / 60:.0f} min", flush=True)
    print('  H5: tree baseline ~0.44, SVM ~0.49 — MLP spread(raw) should be same order.', flush=True)
    print('  Thread A reference (white-box DTA on tree): toward 1.25-1.36x, outward 1.00x (null).', flush=True)


def main():
    if '--smoke' in sys.argv:
        smoke(); return
    X, y = load_iris(return_X_y=True)
    grid = build_grid()
    if '--defect' in sys.argv:
        d = sys.argv[sys.argv.index('--defect') + 1]
        grid = [j for j in grid if j['defect'] == d]
    dk = done_keys()
    todo = [j for j in grid if key({**j, 'attack': 'hsj', 'tc': TC, 'feat': FEAT}) not in dk]
    print(f'MLP grid: {len(todo)}/{len(grid)} cells to run ({len(todo) * len(ATTACKS)} rows)', flush=True)
    t0 = time.time()
    for i, job in enumerate(todo, 1):
        append_rows(run_cell(X, y, job))
        el = (time.time() - t0) / 60
        desc = ' '.join(f'{k}={job[k]}' for k in ('defect', 'protocol', 'bias', 'k', 'n_out',
                                                  'direction', 'seed')
                        if job[k] not in (-1, -1.0, 'none', ''))
        msg = f'[{i}/{len(todo)}] {desc} elapsed={el:.1f}m ETA={el / i * (len(todo) - i):.1f}m'
        print(f'  {msg}', flush=True)
        PROGRESS.write_text(msg + '\n')
    print(f'DONE: {len(todo)} cells in {(time.time() - t0) / 60:.1f}m -> {OUT.name}', flush=True)


if __name__ == '__main__':
    main()
