"""Phase 2 - defect taxonomy: does the SIGN of adversarial localization
discriminate defect type?

Phase 1 found robust ANTI-localization for coverage gaps and proposed the
mechanism: attacks move points to the nearest decision boundary; boundaries live
where data is dense and contested; deleting a region's data pushes the boundary
away, so adversarial examples avoid holes.

That mechanism makes a falsifiable prediction with the OPPOSITE sign for label
contamination: flipped labels inside a region create locally contested,
fragmented boundaries, which ATTRACT adversarial examples. If that holds, the
research claim redefines from "adversarial examples find weak regions" to
"adversarial localization is a *differential* diagnostic - its sign separates
boundary-attracting defects (contamination) from boundary-repelling ones
(coverage gaps), while density baselines see only the latter."

Design: identical harness, region definition R, model x attack matrix, scores
and baselines as phase1_localization.py (helpers imported from it), with the
defect injector varied:
  - coverage_gap      : delete training points in R (Phase 1 replication, same
                        code path - guards against config drift).
  - label_noise_50    : flip the labels of a random 50% of training points in R
                        (contested region / classic noise).
  - label_noise_100   : flip all training labels in R (systematic annotation
                        error / structured mislabeling).
Ground-truth R membership for test points always uses TRUE labels (the test set
is never contaminated).

Prediction table (mechanism -> expected adv-density AUROC):
  coverage_gap    < 0.5  (boundary repelled)
  label_noise_*   > 0.5  (boundary attracted)
  data-sparsity baseline: high for gap, ~0.5 (blind) for noise - the
  complementary blindness is the differential signature.

Run:  .venv/Scripts/python experiments/phase2_defect_taxonomy.py
"""

import contextlib
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import make_data  # noqa: E402

from art.attacks.evasion import (  # noqa: E402
    DecisionTreeAttack,
    FastGradientMethod,
    HopSkipJump,
)
from art.estimators.classification import SklearnClassifier  # noqa: E402
from art.estimators.classification.scikitlearn import (  # noqa: E402
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnSVC,
)

from phase1_localization import (  # noqa: E402  (identical scoring code path)
    auroc,
    knn_disagreement,
    knn_mean_dist,
    model_uncertainty,
)

# ---- configuration (mirrors phase1_localization) --------------------------------
SEEDS = [42, 125, 58]
SEVERITIES = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFECTS = ["coverage_gap", "label_noise_50", "label_noise_100"]
N_SAMPLES = 3000
NOISE = 0.15
TEST_FRAC = 0.3
TARGET = 0
FEATURE = 0
ATTACK_SUBSAMPLE = 300
K_NEIGHBOURS = 10
FGM_EPS = 0.3
HSJ_KW = dict(max_iter=15, max_eval=1000, init_eval=20, verbose=False)
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_defect_taxonomy.csv")


def inject(X, y, defect, target, feature, severity, rng):
    """Apply `defect` to the training set inside region R (the target-class slab
    at/below the severity-percentile cutoff). Returns (Xb, yb, cutoff)."""
    vals = np.sort(X[y == target, feature])
    cutoff = float(vals[int(len(vals) * severity)])
    region = (y == target) & (X[:, feature] <= cutoff)
    if defect == "coverage_gap":
        keep = ~region
        return X[keep], y[keep], cutoff
    if defect in ("label_noise_50", "label_noise_100"):
        frac = 0.5 if defect == "label_noise_50" else 1.0
        idx = np.flatnonzero(region)
        flip = rng.choice(idx, size=int(len(idx) * frac), replace=False)
        yb = y.copy()
        yb[flip] = 1 - yb[flip]
        return X, yb, cutoff
    raise ValueError(defect)


def run():
    rows = []
    for seed in SEEDS:
        X, y = make_data(N_SAMPLES, NOISE, "moons", SEED=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, stratify=y, random_state=seed
        )
        clip = (float(X.min()), float(X.max()))
        sub_rng = np.random.default_rng(seed)
        attack_idx = sub_rng.choice(
            len(X_te), size=min(ATTACK_SUBSAMPLE, len(X_te)), replace=False
        )

        for defect in DEFECTS:
            for severity in SEVERITIES:
                inj_rng = np.random.default_rng(seed * 1000 + int(severity * 10))
                Xb, yb, cutoff = inject(X_tr, y_tr, defect, TARGET, FEATURE, severity, inj_rng)
                in_R = ((y_te == TARGET) & (X_te[:, FEATURE] <= cutoff)).astype(int)
                if in_R.sum() < 5:
                    continue

                base_sparsity = auroc(in_R, knn_mean_dist(X_te, Xb, K_NEIGHBOURS))
                base_disagree = auroc(
                    in_R, knn_disagreement(X_te, y_te, Xb, yb, K_NEIGHBOURS)
                )

                models = {
                    "tree": DecisionTreeClassifier(max_depth=10, random_state=seed).fit(Xb, yb),
                    "svm": SVC(kernel="rbf", random_state=seed).fit(Xb, yb),
                    "mlp": MLPClassifier(
                        hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed
                    ).fit(Xb, yb),
                }
                wrappers = {
                    "tree": ScikitlearnDecisionTreeClassifier(model=models["tree"], clip_values=clip),
                    "svm": ScikitlearnSVC(model=models["svm"], clip_values=clip),
                    "mlp": SklearnClassifier(model=models["mlp"], clip_values=clip),
                }
                attacks = {
                    ("tree", "DecisionTreeAttack"): DecisionTreeAttack(
                        classifier=wrappers["tree"], verbose=False
                    ),
                    ("tree", "HopSkipJump"): HopSkipJump(classifier=wrappers["tree"], **HSJ_KW),
                    ("svm", "FGM"): FastGradientMethod(estimator=wrappers["svm"], eps=FGM_EPS),
                    ("svm", "HopSkipJump"): HopSkipJump(classifier=wrappers["svm"], **HSJ_KW),
                    ("mlp", "HopSkipJump"): HopSkipJump(classifier=wrappers["mlp"], **HSJ_KW),
                }

                X_sub = X_te[attack_idx]
                for (model_name, attack_name), attack in attacks.items():
                    model = models[model_name]
                    X_adv = attack.generate(x=X_sub)
                    flip_rate = float((model.predict(X_adv) != model.predict(X_sub)).mean())
                    adv_density = -knn_mean_dist(X_te, X_adv, K_NEIGHBOURS)
                    pert = np.linalg.norm(X_adv - X_sub, axis=1)

                    rows.append(
                        {
                            "seed": seed,
                            "defect": defect,
                            "severity": severity,
                            "model": model_name,
                            "attack": attack_name,
                            "n_in_R": int(in_R.sum()),
                            "flip_rate": flip_rate,
                            "auroc_adv_density": auroc(in_R, adv_density),
                            "auroc_pert_mag": auroc(in_R[attack_idx], -pert),
                            "auroc_data_sparsity": base_sparsity,
                            "auroc_knn_disagree": base_disagree,
                            "auroc_uncertainty": auroc(in_R, model_uncertainty(model, X_te)),
                        }
                    )
                print(f"seed={seed} {defect:>15} severity={severity:.1f} done "
                      f"(|R|={in_R.sum()}, sparsity={base_sparsity:.3f}, "
                      f"disagree={base_disagree:.3f})")
    return pl.DataFrame(rows)


def report(df: pl.DataFrame):
    print("\n" + "=" * 100)
    print("DEFECT TAXONOMY: mean AUROC over seeds x severities "
          "(0.5 = chance; <0.5 = adversarial examples AVOID R; >0.5 = concentrate in R)")
    print("=" * 100)
    agg = (
        df.group_by(["defect", "model", "attack"])
        .agg(
            pl.col("auroc_adv_density").mean().alias("adv_density"),
            pl.col("auroc_pert_mag").mean().alias("pert_mag"),
            pl.col("auroc_data_sparsity").mean().alias("sparsity"),
            pl.col("auroc_knn_disagree").mean().alias("disagree"),
            pl.col("auroc_uncertainty").mean().alias("uncertainty"),
            pl.col("flip_rate").mean().alias("flip_rate"),
        )
        .sort(["defect", "model", "attack"])
    )
    print(f"{'defect':>16} {'model':>5} {'attack':>18} | {'adv_dens':>8} {'pert':>6} | "
          f"{'sparsity':>8} {'disagree':>8} {'uncert':>7} | {'flip':>5}")
    for r in agg.iter_rows(named=True):
        print(f"{r['defect']:>16} {r['model']:>5} {r['attack']:>18} | "
              f"{r['adv_density']:>8.3f} {r['pert_mag']:>6.3f} | {r['sparsity']:>8.3f} "
              f"{r['disagree']:>8.3f} {r['uncertainty']:>7.3f} | {r['flip_rate']:>5.2f}")

    print("\n--- SIGN-FLIP TEST (mechanism prediction: gap < 0.5 < noise) ---")
    sign = (
        df.group_by(["defect", "model", "attack"])
        .agg(pl.col("auroc_adv_density").mean())
        .pivot(index=["model", "attack"], on="defect", values="auroc_adv_density")
        .sort(["model", "attack"])
    )
    for r in sign.iter_rows(named=True):
        gap = r.get("coverage_gap"); n50 = r.get("label_noise_50"); n100 = r.get("label_noise_100")
        flip50 = "FLIP" if (gap is not None and n50 is not None and gap < 0.5 < n50) else "no"
        flip100 = "FLIP" if (gap is not None and n100 is not None and gap < 0.5 < n100) else "no"
        print(f"  {r['model']:>5} x {r['attack']:<18} gap={gap:.3f}  "
              f"noise50={n50:.3f} [{flip50}]  noise100={n100:.3f} [{flip100}]")

    print("\n--- Baseline blindness (differential signature) ---")
    blind = (
        df.group_by("defect")
        .agg(
            pl.col("auroc_data_sparsity").mean().alias("sparsity"),
            pl.col("auroc_knn_disagree").mean().alias("disagree"),
        )
        .sort("defect")
    )
    for r in blind.iter_rows(named=True):
        print(f"  {r['defect']:>16}: data-sparsity={r['sparsity']:.3f}  "
              f"knn-disagree={r['disagree']:.3f}")

    print("\n--- Severity calibration (Spearman of adv_density AUROC vs severity) ---")
    for (defect, model, attack), g in sorted(
        df.group_by(["defect", "model", "attack"]), key=lambda kv: kv[0]
    ):
        rho, p = spearmanr(g["severity"].to_numpy(), g["auroc_adv_density"].to_numpy())
        print(f"  {defect:>16} {model:>5} x {attack:<18} rho={rho:+.3f} (p={p:.3f})")

    print("\nNote: evidence only - reframing the research claim on this result is a "
          "human call (docs/research_plan.md).")


def main():
    warnings.filterwarnings("ignore")
    print(f"Phase 2 defect taxonomy: {len(SEEDS)} seeds x {len(SEVERITIES)} severities "
          f"x {len(DEFECTS)} defects x 5 model-attack combos")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        df = run()
    df.write_csv(OUT_CSV)
    report(df)
    print(f"\nPer-config rows written to {OUT_CSV}")


if __name__ == "__main__":
    main()
