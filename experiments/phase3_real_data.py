"""Phase 3 - does the boundary-contention pattern survive on REAL datasets?

Same protocol as Phases 1-2 (inject a KNOWN defect into a recorded region R,
train several models, attack them, score whether adversarial examples rank R
highly, race against cheap baselines) - but on three real datasets instead of
the synthetic moons toy:

  breast_cancer : 569 tumour records, 30 measurements each (sklearn, local)
  banknote      : 1372 banknote photos, 4 wavelet stats (UCI)
  adult         : census income records, 6 numeric columns, subsampled to 4000 (UCI)

Honesty notes (the point of this phase is to find where the story BREAKS):
- We still inject the damage ourselves. Real data, synthetic defect. That is
  unavoidable: grading a detector requires knowing where the damage is. What
  this phase tests is whether real data GEOMETRY (dimensions, scales, class
  shapes) preserves or destroys the pattern - not whether real-world defects
  behave like our injections.
- Features are standardized (zero mean, unit variance on the training split);
  attacks, distances and the region cutoff all live in that space.
- Region R = target-class points below a severity-quantile on ONE feature,
  chosen by a fixed rule (highest |correlation| with the label) to avoid
  cherry-picking per result.
- Attack success (flip rate) is recorded per config; low flip rates weaken the
  adversarial signal and are reported, not filtered away.
- METHOD FINDING (the hard way): HopSkipJump against a DECISION TREE in higher
  dimensions does not reliably terminate - the tree's piecewise-constant output
  gives the attack's step-size search no gradient signal to cross, and its
  epsilon-halving loop ran for 7.5 CPU-hours before we killed it. The black-box
  attack therefore runs against the smooth models (SVM, MLP) only; the tree
  keeps its native attack. This is itself evidence about the method's
  practicality on real data and is reported in the write-up.

Run:  .venv/Scripts/python experiments/phase3_real_data.py
"""

import contextlib
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import time

import numpy as np
import polars as pl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from art.attacks.evasion import DecisionTreeAttack, HopSkipJump  # noqa: E402
from art.estimators.classification import SklearnClassifier  # noqa: E402
from art.estimators.classification.scikitlearn import (  # noqa: E402
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnSVC,
)

from phase1_localization import (  # noqa: E402
    auroc,
    knn_disagreement,
    knn_mean_dist,
    model_uncertainty,
)
from phase2_defect_taxonomy import inject  # noqa: E402
from phase2_entropy_baseline import label_entropy  # noqa: E402

SEEDS = [42, 125, 58]
SEVERITIES = [0.3, 0.5, 0.7]
DEFECTS = ["coverage_gap", "label_noise_50", "label_noise_100"]
TEST_FRAC = 0.3
TARGET = 0
ATTACK_SUBSAMPLE = 150
K_NEIGHBOURS = 10
HSJ_KW = dict(max_iter=15, max_eval=1000, init_eval=30, verbose=False)
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase3_real_data.csv")


def load_datasets():
    """Returns {name: (X, y)} with y in {0,1}."""
    out = {}
    bc = load_breast_cancer()
    out["breast_cancer"] = (bc.data.astype(float), bc.target.astype(int))

    from ucimlrepo import fetch_ucirepo

    bn = fetch_ucirepo("banknote authentication")
    out["banknote"] = (
        bn.data.features.to_numpy().astype(float),
        bn.data.targets.to_numpy().ravel().astype(int),
    )

    ad = fetch_ucirepo("Adult")
    Xdf = ad.data.features
    num = Xdf.select_dtypes("number")
    y_raw = ad.data.targets.to_numpy().ravel()
    y = np.array([1 if str(v).strip().rstrip(".") == ">50K" else 0 for v in y_raw])
    X = num.to_numpy().astype(float)
    keep = ~np.isnan(X).any(axis=1)
    X, y = X[keep], y[keep]
    rng = np.random.default_rng(0)  # fixed subsample, stratified
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=2000, replace=False) for c in (0, 1)
    ])
    out["adult_numeric"] = (X[idx], y[idx])
    return out


def pick_feature(X, y):
    """Fixed rule: feature with highest |correlation| with the label."""
    corrs = [abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]
    return int(np.argmax(corrs))


def run():
    rows = []
    datasets = load_datasets()
    for ds_name, (X_all, y_all) in datasets.items():
        feature = pick_feature(X_all, y_all)
        print(f"\n=== {ds_name}: {X_all.shape[0]} rows x {X_all.shape[1]} features, "
              f"region feature index {feature} ===", flush=True)
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=TEST_FRAC, stratify=y_all, random_state=seed
            )
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
            clip = (float(X_tr.min()) - 1.0, float(X_tr.max()) + 1.0)
            sub_idx = np.random.default_rng(seed).choice(
                len(X_te), size=min(ATTACK_SUBSAMPLE, len(X_te)), replace=False
            )

            for defect in DEFECTS:
                for severity in SEVERITIES:
                    rng = np.random.default_rng(seed * 1000 + int(severity * 10))
                    Xb, yb, cutoff = inject(X_tr, y_tr, defect, TARGET, feature,
                                            severity, rng)
                    in_R = ((y_te == TARGET) & (X_te[:, feature] <= cutoff)).astype(int)
                    if in_R.sum() < 5 or in_R.sum() > len(in_R) - 5:
                        continue

                    base = {
                        "auroc_data_sparsity": auroc(
                            in_R, knn_mean_dist(X_te, Xb, K_NEIGHBOURS)),
                        "auroc_knn_disagree": auroc(
                            in_R, knn_disagreement(X_te, y_te, Xb, yb, K_NEIGHBOURS)),
                        "auroc_entropy": auroc(
                            in_R, label_entropy(X_te, Xb, yb, K_NEIGHBOURS)),
                    }

                    # NOTE: no HopSkipJump vs the tree - piecewise-constant
                    # models stall its step-size search (see module docstring).
                    models = {
                        "tree": DecisionTreeClassifier(
                            max_depth=10, random_state=seed).fit(Xb, yb),
                        "svm": SVC(kernel="rbf", random_state=seed).fit(Xb, yb),
                        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16),
                                             max_iter=500, random_state=seed).fit(Xb, yb),
                    }
                    attacks = {
                        ("tree", "DecisionTreeAttack"): DecisionTreeAttack(
                            classifier=ScikitlearnDecisionTreeClassifier(
                                model=models["tree"], clip_values=clip),
                            verbose=False),
                        ("svm", "HopSkipJump"): HopSkipJump(
                            classifier=ScikitlearnSVC(
                                model=models["svm"], clip_values=clip), **HSJ_KW),
                        ("mlp", "HopSkipJump"): HopSkipJump(
                            classifier=SklearnClassifier(
                                model=models["mlp"], clip_values=clip), **HSJ_KW),
                    }

                    X_sub = X_te[sub_idx]
                    for (model_name, attack_name), attack in attacks.items():
                        model = models[model_name]
                        t0 = time.time()
                        X_adv = attack.generate(x=X_sub)
                        attack_secs = time.time() - t0
                        flip = float(
                            (model.predict(X_adv) != model.predict(X_sub)).mean())
                        rows.append({
                            "dataset": ds_name,
                            "seed": seed,
                            "defect": defect,
                            "severity": severity,
                            "model": model_name,
                            "attack": attack_name,
                            "n_in_R": int(in_R.sum()),
                            "flip_rate": flip,
                            "attack_secs": round(attack_secs, 1),
                            "auroc_adv_density": auroc(
                                in_R, -knn_mean_dist(X_te, X_adv, K_NEIGHBOURS)),
                            "auroc_uncertainty": auroc(
                                in_R, model_uncertainty(model, X_te)),
                            **base,
                        })
                print(f"  seed={seed} {defect:>15} done", flush=True)
    return pl.DataFrame(rows)


def report(df: pl.DataFrame):
    for ds in df["dataset"].unique().sort().to_list():
        d = df.filter(pl.col("dataset") == ds)
        print(f"\n{'='*90}\n{ds}: mean localization AUROC (0.5 = chance)\n{'='*90}")
        agg = (
            d.group_by(["defect", "model", "attack"])
            .agg(
                pl.col("auroc_adv_density").mean().alias("adv"),
                pl.col("auroc_data_sparsity").mean().alias("sparsity"),
                pl.col("auroc_knn_disagree").mean().alias("disagree"),
                pl.col("auroc_entropy").mean().alias("entropy"),
                pl.col("auroc_uncertainty").mean().alias("uncert"),
                pl.col("flip_rate").mean().alias("flip"),
            )
            .sort(["defect", "model", "attack"])
        )
        print(f"{'defect':>16} {'model':>5} {'attack':>18} | {'adv':>6} | "
              f"{'sparse':>6} {'disagr':>6} {'entropy':>7} {'uncert':>6} | {'flip':>5}")
        for r in agg.iter_rows(named=True):
            print(f"{r['defect']:>16} {r['model']:>5} {r['attack']:>18} | "
                  f"{r['adv']:>6.3f} | {r['sparsity']:>6.3f} {r['disagree']:>6.3f} "
                  f"{r['entropy']:>7.3f} {r['uncert']:>6.3f} | {r['flip']:>5.2f}")

    print("\n--- SIGN PATTERN vs the moons result (gap<0.5<noise50, noise100<0.5) ---")
    for ds in df["dataset"].unique().sort().to_list():
        d = df.filter(pl.col("dataset") == ds)
        parts = []
        ok = 0
        for defect, expect_high in [("coverage_gap", False),
                                    ("label_noise_50", True),
                                    ("label_noise_100", False)]:
            m = d.filter(pl.col("defect") == defect)["auroc_adv_density"].mean()
            hit = (m > 0.5) == expect_high
            ok += hit
            parts.append(f"{defect}={m:.3f}[{'ok' if hit else 'BREAKS'}]")
        print(f"  {ds:>14}: " + "  ".join(parts) + f"   -> {ok}/3 match")
    print("\nNote: evidence only; interpretation and any claims are reviewed by humans.")


def main():
    warnings.filterwarnings("ignore")
    print("Phase 3 real-data run: 3 datasets x 3 defects x 3 severities x 3 seeds "
          "x 3 model-attack combos")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        df = run()
    df.write_csv(OUT_CSV)
    report(df)
    print(f"\nPer-config rows written to {OUT_CSV}")


if __name__ == "__main__":
    main()
