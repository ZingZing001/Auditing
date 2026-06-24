"""Axis A, steps 2-3 - iris real-data pilot + OPTICS stability sweep.

Step 2: regenerate a scoped ``data/data_bias.parquet`` (iris only, 3 seeds, both
bias types) with the corrected density metric, so the notebook analysis runs
end-to-end on real data.

Step 3: at iris's small fold sizes (~15 adversarial points/fold) OPTICS with the
default ``min_samples`` finds very few clusters. Sweep ``min_samples in {3,5,8}``
and report how the count/density-vs-bias trend (and the mean cluster count) move,
to characterise how fragile the clustering signal is at this N (context-doc 6.7).

Run:  .venv/Scripts/python.exe experiments/axis_a_iris_pilot.py
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
from ucimlrepo import fetch_ucirepo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import audit_tree_bias, labelencoding  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_OUT = os.path.join(ROOT, "data", "data_bias.parquet")

SEEDS = [42, 125, 58]
BIAS_LEVELS = np.linspace(0.1, 1, 9, endpoint=False)
N_SPLITS = 10
DATASET = "iris"


def load_iris():
    d = fetch_ucirepo(DATASET)
    X = labelencoding(d.data.features.to_numpy().copy())
    y = labelencoding(d.data.targets.to_numpy().copy())
    return X, y


def spearman_block(label, df):
    rho_c, p_c = spearmanr(df["bias"].to_numpy(), df["count"].to_numpy())
    rho_d, p_d = spearmanr(df["bias"].to_numpy(), df["density"].to_numpy())
    print(f"  {label:>20}: count rho={rho_c:+.3f} (p={p_c:.1e}) | "
          f"density rho={rho_d:+.3f} (p={p_d:.1e}) | "
          f"mean_count={df['count'].mean():.2f}")


def generate_parquet(X, y):
    """Step 2: canonical parquet at min_samples=5, both bias types (as bias.py)."""
    res = pl.DataFrame()
    rows = []  # for the trend summary
    for i in np.unique(y):
        for feature in range(X.shape[1]):
            for j in SEEDS:
                for k in BIAS_LEVELS:
                    for over, tag in [(False, "undersampling"), (True, "oversampling")]:
                        out = audit_tree_bias(
                            X, y, np.float64(k), stopping=feature, SEED=j,
                            n_splits=N_SPLITS, dbscan=True, target_bias=int(i),
                            over=over, min_samples=5,
                        )
                        out = out.with_columns(
                            pl.lit(int(i)).alias("distribution"),
                            pl.lit(j).alias("seed"),
                            pl.lit(f"{DATASET}_feature{feature}").alias("dataset"),
                            pl.lit(tag).alias("bias_type"),
                        )
                        res = pl.concat([res, out])
                        rows.append({
                            "bias": float(k), "bias_type": tag,
                            "count": float(np.nanmean(out["clusters"].to_numpy())),
                            "density": float(np.nanmean(out["adv distance"].to_numpy())),
                        })
    os.makedirs(os.path.dirname(PARQUET_OUT), exist_ok=True)
    res.write_parquet(PARQUET_OUT)
    print(f"\n[step 2] wrote {res.height} rows -> {PARQUET_OUT}")
    trend = pl.DataFrame(rows)
    print("[step 2] trend vs bias (min_samples=5):")
    for tag in ["undersampling", "oversampling"]:
        spearman_block(tag, trend.filter(pl.col("bias_type") == tag))


def stability_sweep(X, y):
    """Step 3: min_samples in {3,5,8}, undersampling, reduced grid for speed."""
    print("\n[step 3] OPTICS min_samples stability (undersampling, 2 seeds x 2 feats):")
    for ms in [3, 5, 8]:
        rows = []
        for i in np.unique(y):
            for feature in [0, 1]:
                for j in SEEDS[:2]:
                    for k in BIAS_LEVELS:
                        out = audit_tree_bias(
                            X, y, np.float64(k), stopping=feature, SEED=j,
                            n_splits=N_SPLITS, dbscan=True, target_bias=int(i),
                            over=False, min_samples=ms,
                        )
                        rows.append({
                            "bias": float(k),
                            "count": float(np.nanmean(out["clusters"].to_numpy())),
                            "density": float(np.nanmean(out["adv distance"].to_numpy())),
                        })
        spearman_block(f"min_samples={ms}", pl.DataFrame(rows))


def main():
    warnings.filterwarnings("ignore")
    print("Loading iris and running pilot (stderr noise suppressed)...")
    X, y = load_iris()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        generate_parquet(X, y)
        stability_sweep(X, y)
    print("\nDone. Reminder: count-vs-density headline + OPTICS default are human calls.")


if __name__ == "__main__":
    main()
