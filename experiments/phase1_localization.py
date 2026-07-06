"""Phase 1 - the decisive localization test (the GATE).

Question (docs/research_plan.md section 5): when a dataset has a KNOWN coverage
gap in region R, do adversarial examples spatially concentrate IN R - robustly
across models and attacks, and beyond what simple model-free baselines already
reveal?

Design
------
- Data: make_data moons (N=3000, noise=0.15), stratified 70/30 train/test.
- Injected defect: delete training points of the target class whose value on one
  feature falls at/below the severity-percentile cutoff (same mechanism as
  audit_tree_bias, but standalone and returning the cutoff so the ground-truth
  region R is recorded rather than discarded).
  R = { x : true class == target AND x[feature] <= cutoff }.
- Models x attacks (the model-mediation control):
    tree: DecisionTreeAttack (tree-specific) + HopSkipJump (black-box)
    svm (RBF): FastGradientMethod (gradient) + HopSkipJump
    mlp: HopSkipJump
- Adversarial localization score (primary): for every test point, minus the mean
  distance to its k nearest adversarial examples -> higher = adversarial points
  concentrate nearby. AUROC against membership of R.
  Secondary: minus the perturbation magnitude of attacked points (closer
  boundary = easier flip), AUROC on the attacked subset.
- Baselines (same AUROC, model-free unless noted):
    data_sparsity: mean distance to k nearest (biased) training points.
    knn_disagree: fraction of k nearest training neighbours whose label differs
      from the test point's true label.
    uncertainty (model-dependent): 1 - max predicted class probability (margin
      for the SVC, which has no predict_proba here).
- Severity sweep for calibration; multiple seeds.

Gate criteria evaluated on the output (evidence only; the go/no-go is a human
call):
  1. localize  - adv AUROC > 0.5 and >= best baseline.
  2. robust    - effect consistent across >=2 attacks on one model and across
                 >=2 models under the same attack (HSJ).
  3. calibrate - AUROC rises with severity (Spearman).

Run:  .venv/Scripts/python experiments/phase1_localization.py
"""

import contextlib
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
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

# ---- configuration -------------------------------------------------------------
SEEDS = [42, 125, 58]
SEVERITIES = [0.1, 0.3, 0.5, 0.7, 0.9]
N_SAMPLES = 3000
NOISE = 0.15
TEST_FRAC = 0.3
TARGET = 0          # class whose coverage is removed
FEATURE = 0         # feature along which the slab is deleted
ATTACK_SUBSAMPLE = 300
K_NEIGHBOURS = 10
FGM_EPS = 0.3       # ~data scale / 10 for moons
HSJ_KW = dict(max_iter=15, max_eval=1000, init_eval=20, verbose=False)
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_localization.csv")


def inject_coverage_gap(X, y, target, feature, severity):
    """Delete target-class points at/below the severity-percentile cutoff on one
    feature (audit_tree_bias semantics), returning the cutoff = ground truth R."""
    vals = np.sort(X[y == target, feature])
    cutoff = vals[int(len(vals) * severity)]
    keep = ~((y == target) & (X[:, feature] <= cutoff))
    return X[keep], y[keep], float(cutoff)


def knn_mean_dist(query, ref, k):
    nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(ref)
    dist, _ = nn.kneighbors(query)
    return dist.mean(axis=1)


def knn_disagreement(query, query_labels, ref, ref_labels, k):
    nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(ref)
    _, idx = nn.kneighbors(query)
    return (ref_labels[idx] != query_labels[:, None]).mean(axis=1)


def model_uncertainty(model, X):
    """1 - max class probability, or |margin|-based fallback for the SVC."""
    if hasattr(model, "predict_proba"):
        return 1.0 - model.predict_proba(X).max(axis=1)
    margin = np.abs(model.decision_function(X))
    return -margin  # smaller |margin| = less certain = higher score


def auroc(labels, scores):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return roc_auc_score(labels, scores)


def run():
    rows = []
    for seed in SEEDS:
        X, y = make_data(N_SAMPLES, NOISE, "moons", SEED=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, stratify=y, random_state=seed
        )
        clip = (float(X.min()), float(X.max()))
        rng = np.random.default_rng(seed)
        attack_idx = rng.choice(len(X_te), size=min(ATTACK_SUBSAMPLE, len(X_te)), replace=False)

        for severity in SEVERITIES:
            Xb, yb, cutoff = inject_coverage_gap(X_tr, y_tr, TARGET, FEATURE, severity)
            in_R = ((y_te == TARGET) & (X_te[:, FEATURE] <= cutoff)).astype(int)
            if in_R.sum() < 5:
                continue

            # model-free baselines (identical for every model/attack row)
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
                        "severity": severity,
                        "model": model_name,
                        "attack": attack_name,
                        "n_in_R": int(in_R.sum()),
                        "frac_in_R": float(in_R.mean()),
                        "flip_rate": flip_rate,
                        "auroc_adv_density": auroc(in_R, adv_density),
                        "auroc_pert_mag": auroc(in_R[attack_idx], -pert),
                        "auroc_data_sparsity": base_sparsity,
                        "auroc_knn_disagree": base_disagree,
                        "auroc_uncertainty": auroc(in_R, model_uncertainty(model, X_te)),
                    }
                )
            print(f"seed={seed} severity={severity:.1f} done "
                  f"(|R|={in_R.sum()}, sparsity_base={base_sparsity:.3f})")
    return pl.DataFrame(rows)


def gate_report(df: pl.DataFrame):
    print("\n" + "=" * 76)
    print("GATE EVIDENCE (mean AUROC over seeds x severities; higher = better)")
    print("=" * 76)
    agg = (
        df.group_by(["model", "attack"])
        .agg(
            pl.col("auroc_adv_density").mean().alias("adv_density"),
            pl.col("auroc_pert_mag").mean().alias("pert_mag"),
            pl.col("auroc_data_sparsity").mean().alias("base_sparsity"),
            pl.col("auroc_knn_disagree").mean().alias("base_disagree"),
            pl.col("auroc_uncertainty").mean().alias("base_uncertainty"),
            pl.col("flip_rate").mean().alias("flip_rate"),
        )
        .sort(["model", "attack"])
    )
    print(f"{'model':>5} {'attack':>18} | {'adv_dens':>8} {'pert':>6} | "
          f"{'sparsity':>8} {'disagree':>8} {'uncert':>7} | {'flip':>5}")
    for r in agg.iter_rows(named=True):
        print(f"{r['model']:>5} {r['attack']:>18} | {r['adv_density']:>8.3f} "
              f"{r['pert_mag']:>6.3f} | {r['base_sparsity']:>8.3f} "
              f"{r['base_disagree']:>8.3f} {r['base_uncertainty']:>7.3f} | "
              f"{r['flip_rate']:>5.2f}")

    print("\n--- Criterion 1: localize (adv_density > 0.5 and >= best baseline?) ---")
    for r in agg.iter_rows(named=True):
        best_base = max(r["base_sparsity"], r["base_disagree"], r["base_uncertainty"])
        verdict = "PASS" if (r["adv_density"] > 0.5 and r["adv_density"] >= best_base) else "fail"
        print(f"  {r['model']:>5} x {r['attack']:<18} adv={r['adv_density']:.3f} "
              f"best_base={best_base:.3f} -> {verdict}")

    print("\n--- Criterion 2: robust (consistency across attacks / models) ---")
    tree_att = {r["attack"]: r["adv_density"] for r in agg.iter_rows(named=True) if r["model"] == "tree"}
    hsj_mod = {r["model"]: r["adv_density"] for r in agg.iter_rows(named=True) if r["attack"] == "HopSkipJump"}
    print(f"  same model (tree), two attacks : {tree_att}")
    print(f"  same attack (HSJ), three models: {hsj_mod}")

    print("\n--- Criterion 3: calibrate (Spearman of adv_density AUROC vs severity) ---")
    for (model, attack), g in sorted(
        df.group_by(["model", "attack"]), key=lambda kv: kv[0]
    ):
        rho, p = spearmanr(g["severity"].to_numpy(), g["auroc_adv_density"].to_numpy())
        print(f"  {model:>5} x {attack:<18} rho={rho:+.3f} (p={p:.3f})")

    print("\nNote: evidence only - the go/no-go decision is a human call "
          "(docs/research_plan.md section 5).")


def main():
    warnings.filterwarnings("ignore")
    print("Phase 1 localization gate: "
          f"{len(SEEDS)} seeds x {len(SEVERITIES)} severities x 5 model-attack combos")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        df = run()
    df.write_csv(OUT_CSV)
    gate_report(df)
    print(f"\nPer-config rows written to {OUT_CSV}")


if __name__ == "__main__":
    main()
