"""Plain-language figures for the Phase 3 real-data experiment.

Design rule: every figure must be readable by a non-specialist. No metric
jargon on any axis - the localization AUROC is presented as a "detection
score" with the guessing level (0.5) marked, damage types are written in
plain words, and each figure states its takeaway in the title.

  verdict    - real_verdict.png : one coloured grid, datasets x damage types,
               "does the attack method point at the damage?" with the moons
               (toy) result as a reference row.
  bars       - real_bars.png : per dataset, attack method vs two simple
               checks, grouped by damage type.
  projection - real_projection.png : simplified 2-D map (PCA) per dataset -
               where the damaged region is vs where attack points land.

Run:  .venv/Scripts/python experiments/visualize_real.py
"""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
PHASE2_CSV = os.path.join(HERE, "phase2_defect_taxonomy.csv")
PHASE3_CSV = os.path.join(HERE, "phase3_real_data.csv")
CONTROL_CSV = os.path.join(HERE, "phase3_no_defect_control.csv")

DEFECT_WORDS = {
    "coverage_gap": "a chunk of data\ndeleted",
    "label_noise_50": "half the labels\nscrambled",
    "label_noise_100": "all labels flipped\n(consistently)",
}
DATASET_WORDS = {
    "moons (toy)": "moons (toy reference)",
    "breast_cancer": "tumour records\n(569 rows, 30 measurements)",
    "banknote": "banknote photos\n(1372 rows, 4 measurements)",
    "adult_numeric": "census income\n(4000 rows, 6 numbers)",
}
COMBOS = [("tree", "DecisionTreeAttack"), ("svm", "HopSkipJump"), ("mlp", "HopSkipJump")]


def adv_means(csv, extra_group=()):
    """Mean attack-based detection score per (defect [, dataset]) over the
    shared model/attack combos."""
    df = pl.read_csv(csv)
    mask = pl.any_horizontal([
        (pl.col("model") == m) & (pl.col("attack") == a) for m, a in COMBOS
    ])
    df = df.filter(mask)
    return df.group_by(["defect", *extra_group]).agg(
        pl.col("auroc_adv_density").mean().alias("adv"))


def score_color(v):
    """Blue = points away, gray = guessing, red = points at the damage."""
    if v is None:
        return "#eeeeee"
    if v < 0.45:
        return "#aec7e8"
    if v <= 0.55:
        return "#d9d9d9"
    return "#f4a9a9"


def fig_verdict(fname="real_verdict.png"):
    rows = []  # (display name, {defect: score}, control score or None)
    m2 = {r["defect"]: r["adv"] for r in adv_means(PHASE2_CSV).iter_rows(named=True)}
    rows.append(("moons (toy)", m2, None))
    p3 = adv_means(PHASE3_CSV, extra_group=("dataset",))
    ctrl = {
        r["dataset"]: r["adv"]
        for r in pl.read_csv(CONTROL_CSV)
        .group_by("dataset")
        .agg(pl.col("auroc_adv_density").mean().alias("adv"))
        .iter_rows(named=True)
    }
    for ds in sorted(p3["dataset"].unique().to_list()):
        m = {r["defect"]: r["adv"]
             for r in p3.filter(pl.col("dataset") == ds).iter_rows(named=True)}
        rows.append((ds, m, ctrl.get(ds)))

    defects = ["coverage_gap", "label_noise_50", "label_noise_100"]
    ncols = len(defects) + 1  # + the "nothing wrong" control column
    fig, ax = plt.subplots(figsize=(14.0, 2.3 + 1.15 * len(rows)))
    ax.set_xlim(0, ncols), ax.set_ylim(0, len(rows) + 0.9)
    ax.axis("off")
    for j, d in enumerate(defects):
        ax.text(j + 0.5, len(rows) + 0.08, DEFECT_WORDS[d], ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.text(len(defects) + 0.5, len(rows) + 0.08,
            "NOTHING wrong\n(placebo check)", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#8a5a00")

    def cell(x, y, v, is_control=False):
        fc = "#ffe9c7" if (is_control and v is not None) else score_color(v)
        ax.add_patch(plt.Rectangle((x + 0.03, y + 0.06), 0.94, 0.88,
                                   fc=fc, ec="white", lw=2))
        if v is None:
            ax.text(x + 0.5, y + 0.5, "n/a", ha="center", va="center",
                    fontsize=11, color="0.5")
            return
        arrow = "points AWAY" if v < 0.45 else (
            "can't tell" if v <= 0.55 else "points AT it")
        ax.text(x + 0.5, y + 0.58, f"{v:.2f}", ha="center", va="center",
                fontsize=15, fontweight="bold")
        ax.text(x + 0.5, y + 0.26, arrow, ha="center", va="center", fontsize=9)

    for i, (ds, m, cv) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(-0.06, y + 0.5, DATASET_WORDS.get(ds, ds), ha="right", va="center",
                fontsize=10.5)
        for j, d in enumerate(defects):
            cell(j, y, m.get(d))
        cell(len(defects), y, cv, is_control=True)

    ax.set_title(
        "Does the attack method point at the damaged part of the data?\n"
        "score: 1.0 = always points at the damage, 0.5 = random guessing, below 0.5 = points away.\n"
        "The placebo column is the same score when NOTHING was damaged - "
        "any real detection must clearly beat it.",
        fontsize=12.5, pad=30)
    fig.tight_layout()
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_bars(fname="real_bars.png"):
    df = pl.read_csv(PHASE3_CSV)
    mask = pl.any_horizontal([
        (pl.col("model") == m) & (pl.col("attack") == a) for m, a in COMBOS
    ])
    df = df.filter(mask)
    datasets = sorted(df["dataset"].unique().to_list())
    defects = ["coverage_gap", "label_noise_50", "label_noise_100"]
    series = [
        ("auroc_adv_density", "attack-based method", "#d62728"),
        ("auroc_data_sparsity", "simple check: count nearby data", "#7f7f7f"),
        ("auroc_entropy", "simple check: compare neighbour labels", "#bcbd22"),
    ]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.6 * len(datasets), 4.6),
                             sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        d = df.filter(pl.col("dataset") == ds)
        x = np.arange(len(defects))
        width = 0.8 / len(series)
        for k, (col, label, color) in enumerate(series):
            vals = [d.filter(pl.col("defect") == de)[col].mean() for de in defects]
            ax.bar(x + k * width, vals, width, label=label, color=color)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.text(len(defects) - 0.6, 0.515, "guessing", fontsize=8)
        ax.set_xticks(x + width, [DEFECT_WORDS[de] for de in defects], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_WORDS.get(ds, ds).replace("\n", " "), fontsize=11)
    np.atleast_1d(axes)[0].set_ylabel("detection score")
    np.atleast_1d(axes)[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Attack-based method vs two simple checks, on real data "
                 "(higher = better at finding the damage)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_projection(seed=42, severity=0.5, fname="real_projection.png"):
    """2-D simplified map per dataset x {deleted chunk, scrambled labels}:
    where the damaged region is vs where attack points land (mlp + HSJ)."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import SklearnClassifier

    from phase2_defect_taxonomy import inject
    from phase3_real_data import (
        ATTACK_SUBSAMPLE,
        HSJ_KW,
        TARGET,
        TEST_FRAC,
        load_datasets,
        pick_feature,
    )

    defects = ["coverage_gap", "label_noise_50"]
    datasets = load_datasets()
    fig, axes = plt.subplots(len(defects), len(datasets),
                             figsize=(5.6 * len(datasets), 4.8 * len(defects)))
    for j, (ds, (X_all, y_all)) in enumerate(datasets.items()):
        feature = pick_feature(X_all, y_all)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=TEST_FRAC, stratify=y_all, random_state=seed)
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
        clip = (float(X_tr.min()) - 1.0, float(X_tr.max()) + 1.0)
        pca = PCA(n_components=2, random_state=0).fit(X_tr)
        sub = np.random.default_rng(seed).choice(
            len(X_te), size=min(ATTACK_SUBSAMPLE, len(X_te)), replace=False)

        for i, defect in enumerate(defects):
            print(f"  projection: {ds} x {defect}")
            rng = np.random.default_rng(seed * 1000 + int(severity * 10))
            Xb, yb, cutoff = inject(X_tr, y_tr, defect, TARGET, feature, severity, rng)
            in_R = (y_te == TARGET) & (X_te[:, feature] <= cutoff)
            mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                                random_state=seed).fit(Xb, yb)
            X_adv = HopSkipJump(
                classifier=SklearnClassifier(model=mlp, clip_values=clip), **HSJ_KW
            ).generate(x=X_te[sub])

            ax = axes[i, j]
            P_te, P_R, P_adv = pca.transform(X_te), pca.transform(X_te[in_R]), pca.transform(X_adv)
            ax.scatter(*P_te.T, s=6, c="0.8", lw=0, label="normal data")
            ax.scatter(*P_R.T, s=18, facecolors="none", edgecolors="#1f77b4", lw=1.0,
                       label="damaged region")
            ax.scatter(*P_adv.T, s=10, c="#d62728", lw=0, alpha=0.85,
                       label="where attacks land")
            ax.set_title(f"{DATASET_WORDS.get(ds, ds).splitlines()[0]}  |  "
                         f"{DEFECT_WORDS[defect].replace(chr(10), ' ')}", fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0 and j == 0:
                ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Simplified 2-D view of each dataset: blue circles = the damaged region, "
        "red dots = where attack points land\n"
        "(top: data deleted - do red dots avoid the blue region?   "
        "bottom: labels scrambled - do they move into it?)",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(FIGDIR, exist_ok=True)
    fig_verdict()
    fig_bars()
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        fig_projection()


if __name__ == "__main__":
    main()
