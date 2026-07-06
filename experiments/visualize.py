"""Figures for the localization experiments (Phase 1, Phase 2, and reusable by
later phases).

Three figure families, all reading the CSVs the experiment scripts write (and,
for scenes, re-running one cheap configuration):

  scenes      - the mechanism, visually: training data after injection, the
                injected defect (deleted points hollow / flipped points x),
                the model's decision boundary, and adversarial examples.
                Shows boundary repulsion (coverage gap) vs attraction (noise).
  bars        - localization AUROC of the adversarial signal vs baselines per
                model x attack (0.5 = chance line).
  calibration - AUROC vs severity per model x attack.

Usage:
  .venv/Scripts/python experiments/visualize.py --phase 1
  .venv/Scripts/python experiments/visualize.py --phase 2
  .venv/Scripts/python experiments/visualize.py --scenes
  .venv/Scripts/python experiments/visualize.py --all

Outputs land in experiments/figures/. Later phases can reuse plot_bars /
plot_calibration on any CSV with the shared schema (auroc_* columns +
model/attack[/defect/severity]).
"""

import argparse
import contextlib
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
PHASE1_CSV = os.path.join(HERE, "phase1_localization.csv")
PHASE2_CSV = os.path.join(HERE, "phase2_defect_taxonomy.csv")

SCORE_COLS = {
    "auroc_adv_density": ("adversarial density", "#d62728"),
    "auroc_data_sparsity": ("data sparsity (baseline)", "#7f7f7f"),
    "auroc_knn_disagree": ("kNN disagreement (baseline)", "#bcbd22"),
    "auroc_uncertainty": ("model uncertainty (baseline)", "#17becf"),
}


# ---------------------------------------------------------------- scene figure
def draw_scene(ax, defect, model_name, seed=42, severity=0.5):
    """One panel: injected training set, defect footprint, decision boundary,
    adversarial examples. Re-runs a single cheap config."""
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import SklearnClassifier
    from art.estimators.classification.scikitlearn import (
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnSVC,
    )

    from phase2_defect_taxonomy import (
        ATTACK_SUBSAMPLE,
        FEATURE,
        HSJ_KW,
        N_SAMPLES,
        NOISE,
        TARGET,
        TEST_FRAC,
        inject,
    )
    from util import make_data

    X, y = make_data(N_SAMPLES, NOISE, "moons", SEED=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=seed
    )
    rng = np.random.default_rng(seed * 1000 + int(severity * 10))
    Xb, yb, cutoff = inject(X_tr, y_tr, defect, TARGET, FEATURE, severity, rng)
    region = (y_tr == TARGET) & (X_tr[:, FEATURE] <= cutoff)

    model = {
        "tree": DecisionTreeClassifier(max_depth=10, random_state=seed),
        "svm": SVC(kernel="rbf", random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed),
    }[model_name].fit(Xb, yb)

    clip = (float(X.min()), float(X.max()))
    wrapper = {
        "tree": lambda m: ScikitlearnDecisionTreeClassifier(model=m, clip_values=clip),
        "svm": lambda m: ScikitlearnSVC(model=m, clip_values=clip),
        "mlp": lambda m: SklearnClassifier(model=m, clip_values=clip),
    }[model_name](model)

    sub = np.random.default_rng(seed).choice(
        len(X_te), size=min(ATTACK_SUBSAMPLE, len(X_te)), replace=False
    )
    X_adv = HopSkipJump(classifier=wrapper, **HSJ_KW).generate(x=X_te[sub])

    # decision regions
    pad = 0.4
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300)
    gx, gy = np.meshgrid(xs, ys)
    Z = model.predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=1, colors=["#aec7e8", "#ffbb78"], alpha=0.35)
    ax.contour(gx, gy, Z, levels=1, colors="k", linewidths=0.8)

    # surviving training data
    ax.scatter(*Xb[yb == 0].T, s=4, c="#1f77b4", alpha=0.4, lw=0)
    ax.scatter(*Xb[yb == 1].T, s=4, c="#ff7f0e", alpha=0.4, lw=0)

    # the defect footprint
    if defect == "coverage_gap":
        ax.scatter(*X_tr[region].T, s=10, facecolors="none", edgecolors="0.45",
                   lw=0.6, label="deleted points (R)")
    else:
        flipped = region & (yb != y_tr)
        ax.scatter(*X_tr[flipped].T, s=14, marker="x", c="#2ca02c", lw=0.9,
                   label="flipped labels (R)")
    ax.axvline(cutoff, color="0.3", ls="--", lw=0.8)

    # adversarial examples
    ax.scatter(*X_adv.T, s=7, c="#d62728", alpha=0.85, lw=0, label="adversarial examples")

    ax.set_title(f"{defect}  |  {model_name} + HopSkipJump", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def fig_scenes(defects=("coverage_gap", "label_noise_50", "label_noise_100"),
               models=("tree", "mlp"), seed=42, severity=0.5, fname="scenes.png"):
    fig, axes = plt.subplots(len(defects), len(models),
                             figsize=(6.2 * len(models), 5.0 * len(defects)))
    axes = np.atleast_2d(axes)
    for i, defect in enumerate(defects):
        for j, model_name in enumerate(models):
            print(f"  scene: {defect} x {model_name}")
            draw_scene(axes[i, j], defect, model_name, seed=seed, severity=severity)
    fig.suptitle(
        "Where adversarial examples land vs where the defect is\n"
        f"(moons N=3000, severity={severity}, seed={seed}; dashed line = region-R cutoff)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- bar figure
def plot_bars(df: pl.DataFrame, ax, title):
    agg = (
        df.group_by(["model", "attack"])
        .agg([pl.col(c).mean() for c in SCORE_COLS])
        .sort(["model", "attack"])
    )
    combos = [f"{m}\n{a}" for m, a in zip(agg["model"], agg["attack"])]
    x = np.arange(len(combos))
    width = 0.8 / len(SCORE_COLS)
    for k, (col, (label, color)) in enumerate(SCORE_COLS.items()):
        ax.bar(x + k * width, agg[col].to_numpy(), width, label=label, color=color)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.text(len(combos) - 0.35, 0.505, "chance", fontsize=7)
    ax.set_xticks(x + width * (len(SCORE_COLS) - 1) / 2, combos, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("localization AUROC (of region R)")
    ax.set_title(title, fontsize=10)


def fig_bars(csv, by_defect, fname):
    df = pl.read_csv(csv)
    if by_defect and "defect" in df.columns:
        defects = df["defect"].unique().sort().to_list()
        fig, axes = plt.subplots(1, len(defects), figsize=(6.5 * len(defects), 4.6),
                                 sharey=True)
        for ax, defect in zip(np.atleast_1d(axes), defects):
            plot_bars(df.filter(pl.col("defect") == defect), ax, defect)
        axes[0].figure.legend(*axes[0].get_legend_handles_labels(),
                              loc="lower center", ncol=4, fontsize=8)
        fig.suptitle("Localization AUROC by defect type: adversarial signal vs baselines",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    else:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        plot_bars(df, ax, "coverage gap (Phase 1)")
        ax.legend(fontsize=8, loc="upper left")
        fig.suptitle("Phase 1: adversarial examples anti-localize the coverage gap",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------- calibration figure
def fig_calibration(csv, fname):
    df = pl.read_csv(csv)
    defects = df["defect"].unique().sort().to_list() if "defect" in df.columns else [None]
    fig, axes = plt.subplots(1, len(defects), figsize=(5.6 * len(defects), 4.4),
                             sharey=True)
    for ax, defect in zip(np.atleast_1d(axes), defects):
        d = df if defect is None else df.filter(pl.col("defect") == defect)
        for (model, attack), g in sorted(d.group_by(["model", "attack"]),
                                         key=lambda kv: kv[0]):
            line = (
                g.group_by("severity").agg(pl.col("auroc_adv_density").mean()).sort("severity")
            )
            ax.plot(line["severity"], line["auroc_adv_density"], marker="o", ms=3,
                    label=f"{model} x {attack}")
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xlabel("severity")
        ax.set_title(defect or "coverage gap (Phase 1)", fontsize=10)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("adv-density localization AUROC")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.suptitle("Severity calibration of the adversarial localization signal", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["1", "2"], default=None)
    p.add_argument("--scenes", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    os.makedirs(FIGDIR, exist_ok=True)

    if args.phase == "1" or args.all:
        fig_bars(PHASE1_CSV, by_defect=False, fname="phase1_auroc_bars.png")
        fig_calibration(PHASE1_CSV, fname="phase1_calibration.png")
    if args.phase == "2" or args.all:
        fig_bars(PHASE2_CSV, by_defect=True, fname="phase2_auroc_bars.png")
        fig_calibration(PHASE2_CSV, fname="phase2_calibration.png")
    if args.scenes or args.all:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            fig_scenes()


if __name__ == "__main__":
    main()
