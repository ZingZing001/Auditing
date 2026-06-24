"""Axis A, step 1 - synthetic ground-truth monotonicity check.

Question: with the corrected density metric, does adversarial-point COMPACTNESS
(``adv distance`` density) and/or OPTICS cluster COUNT track the amount of
injected bias?

This uses a controlled, network-free synthetic dataset (``make_data`` moons) and
the *existing* bias mechanism in ``audit_tree_bias`` (delete a contiguous upper
slab of one class along one feature; larger ``bias`` -> bigger coverage gap). The
defect is known, so a clean monotonic relationship here is evidence the fixed
metric measures what it should, before trusting it on noisy real data.

Output: per-bias mean count/density across (seed, feature, target_class) and the
Spearman rho of count-vs-bias and density-vs-bias. This produces evidence; it
does NOT decide which metric becomes the project's headline (a human call).

Run:  .venv/Scripts/python.exe experiments/axis_a_synthetic.py
"""

import contextlib
import os
import sys
import warnings

# Win11 lacks wmic; loky's physical-core probe warns noisily. Pin it.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import audit_tree_bias, make_data  # noqa: E402

# ---- experiment configuration -------------------------------------------------
SEEDS = [42, 125, 58, 86, 138]
N_SAMPLES = 600
NOISE = 0.15
N_SPLITS = 10
MIN_SAMPLES = 5
# matches bias.py: np.linspace(0.1, 1, 9, endpoint=False) -> 0.1 .. 0.9
BIAS_LEVELS = np.linspace(0.1, 1, 9, endpoint=False)
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axis_a_synthetic.csv")


def run_sweep():
    """One row per (seed, feature, target, bias): mean count/density over folds."""
    rows = []
    for seed in SEEDS:
        X, y = make_data(N_SAMPLES, NOISE, "moons", SEED=seed)
        y = y.reshape(-1, 1)
        for feature in range(X.shape[1]):
            for target in np.unique(y):
                for bias in BIAS_LEVELS:
                    res = audit_tree_bias(
                        X,
                        y,
                        np.float64(bias),
                        stopping=feature,
                        SEED=seed,
                        n_splits=N_SPLITS,
                        dbscan=True,
                        target_bias=int(target),
                        min_samples=MIN_SAMPLES,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "feature": feature,
                            "target": int(target),
                            "bias": float(bias),
                            # mean across folds (nan-safe: some folds may find no cluster)
                            "count": float(np.nanmean(res["clusters"].to_numpy())),
                            "density": float(np.nanmean(res["adv distance"].to_numpy())),
                        }
                    )
    return pl.DataFrame(rows)


def summarise(df: pl.DataFrame):
    per_bias = (
        df.group_by("bias")
        .agg(
            pl.col("count").mean().alias("count_mean"),
            pl.col("count").std().alias("count_std"),
            pl.col("density").mean().alias("density_mean"),
            pl.col("density").std().alias("density_std"),
        )
        .sort("bias")
    )

    print("\n=== Axis A (synthetic moons): mean over "
          f"{len(SEEDS)} seeds x 2 features x 2 classes ===")
    print(f"{'bias':>5} | {'count_mean':>10} {'count_std':>9} | "
          f"{'density_mean':>12} {'density_std':>11}")
    for r in per_bias.iter_rows(named=True):
        print(f"{r['bias']:>5.1f} | {r['count_mean']:>10.3f} {r['count_std']:>9.3f} | "
              f"{r['density_mean']:>12.3f} {r['density_std']:>11.3f}")

    # Spearman trend of the per-config points (not just the means) vs bias.
    rho_c, p_c = spearmanr(df["bias"].to_numpy(), df["count"].to_numpy())
    rho_d, p_d = spearmanr(df["bias"].to_numpy(), df["density"].to_numpy())
    print("\n--- Spearman trend vs bias (all per-config points) ---")
    print(f"count   vs bias: rho = {rho_c:+.3f}  (p = {p_c:.2e})")
    print(f"density vs bias: rho = {rho_d:+.3f}  (p = {p_d:.2e})")
    print(
        "\nHypothesis: bigger coverage gap -> tighter adversarial clusters -> "
        "density rises (positive rho). Result above is the evidence; the "
        "count-vs-density headline call stays with the humans."
    )
    return per_bias


def main():
    warnings.filterwarnings("ignore")
    print("Running synthetic Axis-A sweep (suppressing ART/OPTICS stderr noise)...")
    # tqdm + sklearn RuntimeWarnings go to stderr; keep our stdout table clean.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        df = run_sweep()
    summarise(df)
    df.write_csv(OUT_CSV)
    print(f"\nPer-config results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
