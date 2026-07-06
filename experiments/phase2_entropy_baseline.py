"""Honesty check for the Phase 2 sign-flip: does a no-attack kNN label-ENTROPY
baseline reproduce the adversarial signature (gap < 0.5, noise50 > 0.5,
noise100 < 0.5)?

Local label entropy of the (possibly contaminated) training set, evaluated at
test points, is the cheapest conceivable "boundary contention" statistic - no
model, no attack. If it matches or beats the adversarial signal per defect, the
adversarial geometry adds no *detection* value and the contribution is the
mechanistic characterization, not an instrument.

Run:  .venv/Scripts/python experiments/phase2_entropy_baseline.py
"""

import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_localization import auroc  # noqa: E402
from phase2_defect_taxonomy import (  # noqa: E402
    DEFECTS,
    FEATURE,
    K_NEIGHBOURS,
    N_SAMPLES,
    NOISE,
    SEEDS,
    SEVERITIES,
    TARGET,
    TEST_FRAC,
    inject,
)
from util import make_data  # noqa: E402


def label_entropy(query, ref, ref_labels, k):
    nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(ref)
    _, idx = nn.kneighbors(query)
    p = ref_labels[idx].mean(axis=1)  # binary: fraction of class-1 neighbours
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    res = {d: [] for d in DEFECTS}
    for seed in SEEDS:
        X, y = make_data(N_SAMPLES, NOISE, "moons", SEED=seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, stratify=y, random_state=seed
        )
        for defect in DEFECTS:
            for severity in SEVERITIES:
                rng = np.random.default_rng(seed * 1000 + int(severity * 10))
                Xb, yb, cutoff = inject(X_tr, y_tr, defect, TARGET, FEATURE, severity, rng)
                in_R = ((y_te == TARGET) & (X_te[:, FEATURE] <= cutoff)).astype(int)
                if in_R.sum() < 5:
                    continue
                res[defect].append(
                    auroc(in_R, label_entropy(X_te, Xb, yb, K_NEIGHBOURS))
                )

    print("kNN label-entropy baseline (no model, no attack) - mean AUROC vs region R:")
    for d, v in res.items():
        print(f"  {d:>16}: {np.mean(v):.3f}")
    print("\nAdversarial adv-density signature for comparison (Phase 2 run): "
          "gap ~0.35, noise50 ~0.71, noise100 ~0.38.")


if __name__ == "__main__":
    main()
