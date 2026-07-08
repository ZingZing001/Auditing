"""Phase 3 control - does the attack method "point at" region R even when
NOTHING is wrong with it?

On real data, region R (the low tail of the most label-correlated feature for
one class) sits close to where the classes naturally overlap - which is where
decision boundaries live, and therefore where adversarial examples always land.
If the adversarial detection score is above chance with NO defect injected,
then the above-chance scores in phase3_real_data.py are (at least partly) a
region-placement artifact, not damage detection.

Same pipeline as phase3_real_data.py, severity fixed at 0.5 for the region
definition, but the training data is left UNDAMAGED.

Run:  .venv/Scripts/python experiments/phase3_no_defect_control.py
"""

import contextlib
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import polars as pl
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

from phase1_localization import auroc, knn_mean_dist  # noqa: E402
from phase3_real_data import (  # noqa: E402
    ATTACK_SUBSAMPLE,
    HSJ_KW,
    K_NEIGHBOURS,
    SEEDS,
    TARGET,
    TEST_FRAC,
    load_datasets,
    pick_feature,
)

REGION_SEVERITY = 0.5
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "phase3_no_defect_control.csv")


def run():
    rows = []
    for ds_name, (X_all, y_all) in load_datasets().items():
        feature = pick_feature(X_all, y_all)
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=TEST_FRAC, stratify=y_all, random_state=seed)
            scaler = StandardScaler().fit(X_tr)
            X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
            clip = (float(X_tr.min()) - 1.0, float(X_tr.max()) + 1.0)

            # region defined exactly as in the defect runs - but NO injection
            vals = np.sort(X_tr[y_tr == TARGET, feature])
            cutoff = float(vals[int(len(vals) * REGION_SEVERITY)])
            in_R = ((y_te == TARGET) & (X_te[:, feature] <= cutoff)).astype(int)

            models = {
                "tree": DecisionTreeClassifier(max_depth=10, random_state=seed
                                               ).fit(X_tr, y_tr),
                "svm": SVC(kernel="rbf", random_state=seed).fit(X_tr, y_tr),
                "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                                     random_state=seed).fit(X_tr, y_tr),
            }
            attacks = {
                ("tree", "DecisionTreeAttack"): DecisionTreeAttack(
                    classifier=ScikitlearnDecisionTreeClassifier(
                        model=models["tree"], clip_values=clip), verbose=False),
                ("svm", "HopSkipJump"): HopSkipJump(
                    classifier=ScikitlearnSVC(model=models["svm"], clip_values=clip),
                    **HSJ_KW),
                ("mlp", "HopSkipJump"): HopSkipJump(
                    classifier=SklearnClassifier(model=models["mlp"], clip_values=clip),
                    **HSJ_KW),
            }
            sub = np.random.default_rng(seed).choice(
                len(X_te), size=min(ATTACK_SUBSAMPLE, len(X_te)), replace=False)
            for (model_name, attack_name), attack in attacks.items():
                model = models[model_name]
                X_adv = attack.generate(x=X_te[sub])
                rows.append({
                    "dataset": ds_name, "seed": seed,
                    "model": model_name, "attack": attack_name,
                    "flip_rate": float(
                        (model.predict(X_adv) != model.predict(X_te[sub])).mean()),
                    "auroc_adv_density": auroc(
                        in_R, -knn_mean_dist(X_te, X_adv, K_NEIGHBOURS)),
                })
            print(f"  {ds_name} seed={seed} done", flush=True)
    return pl.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    print("No-defect control: does the attack signal point at region R when "
          "nothing is wrong?")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        df = run()
    df.write_csv(OUT_CSV)
    print("\nmean adversarial 'detection' score with NO damage injected "
          "(0.5 = chance):")
    agg = df.group_by("dataset").agg(
        pl.col("auroc_adv_density").mean().alias("adv_no_defect"),
        pl.col("flip_rate").mean().alias("flip"))
    for r in agg.sort("dataset").iter_rows(named=True):
        print(f"  {r['dataset']:>14}: {r['adv_no_defect']:.3f}   (flip {r['flip']:.2f})")
    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
