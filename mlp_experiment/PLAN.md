# MLP Experiment — Plan

> **Question this thread answers:** do the established defect signals survive a model with a
> **globally-parameterised, smooth** decision boundary — and does the white-box/black-box split
> found on trees ([`../outlier_experiment/FINDINGS_phase05.md`](../outlier_experiment/FINDINGS_phase05.md) F1)
> replicate on a new model family, via **DeepFool vs HSJ**?

**Created:** 2026-08-11 · **Status:** Phase 0 running · **Owner:** research thread 4 (MLP)

Runs alongside the SVM/tree/RF work rather than replacing it. Phase 0 covers **two defects**:

- **coverage gap**, under both injection protocols (the anchor — proves the MLP pipeline
  reproduces known results, and isolates the protocol confound; §3);
- **outlier**, train-fold-only (supervisor request for wider defect coverage, 2026-08-11) —
  and the single most informative cell available, because it is where the *global* boundary
  should diverge from every model tested so far (§0).

---

## 0. Why MLP is the missing quadrant (not just "one more model")

The project's models currently occupy three cells of a 2×2. The fourth is empty:

| | **local** influence | **global** influence |
|---|---|---|
| **piecewise-flat boundary** | Tree, RandomForest | — |
| **smooth boundary** | rbf-SVM | **MLP ← this thread** |

This matters because Thread A's null result was explained *by locality*, and the findings doc
names the follow-up test explicitly
([`FINDINGS_phase05.md`](../outlier_experiment/FINDINGS_phase05.md) F4 / Caveats):

> "the **rbf kernel is local** — a far outlier's Gaussian influence decays to ~0, so it never
> reaches the decision boundary the attack probes … A **linear-kernel SVM** is the clean test of
> whether a *global* boundary sees the outward drag."

An MLP is a stronger version of that test than a linear SVM: **global *and* nonlinear**. Nothing
in the project currently has a globally-parameterised boundary, so every "the signal does/doesn't
survive model X" claim so far is confined to local-influence models.

Two further consequences:

1. **DeepFool is to the MLP what DTA is to the tree.** Thread A's headline is that the outlier
   signal is *white-box only* — deterministic DTA felt the tendril, stochastic HSJ averaged it
   away ([`FINDINGS_outlier.md`](../model_family_experiment/FINDINGS_outlier.md) F1). MLP+DeepFool
   vs MLP+HSJ reruns that contrast on a new model family with **identical weights**, which is why
   §2 insists on one torch model rather than two implementations.
2. **The label-noise hang trap may not apply here.** Thread B abandoned RF/tree label noise
   because HSJ hung on **53/90 cells** — flat-facet boundary fragmentation
   ([`FINDINGS_label_noise.md`](../model_family_experiment/FINDINGS_label_noise.md)). An MLP
   boundary is smooth, so HSJ should converge. Out of Phase 0 scope, but H4 measures the hang
   rate so a later thread knows whether that question is reopenable.

---

## 1. The measurement gotcha that shapes the whole design

**MLPs need standardised features; the rest of the repo trains on raw iris.** Spread is an L2
distance *in feature space*, so a `StandardScaler` silently changes the metric's units and MLP
spread would not be comparable to the tree (~0.44) and SVM (~0.49) baselines.

**Resolution — two layers, both required:**

1. **Inverse-transform adversarial points back to raw iris units before measuring.** The scaler
   is fit on the **training fold only** (no leakage) and is invertible, so `scaler.inverse_transform(adv)`
   puts the adversarial cloud back in the same space every other model in the project reports.
   OPTICS and `pdist` then run in raw space. This makes MLP spread **directly comparable in
   absolute terms**, not just in trend.
2. **Report normalised spread anyway** — spread ÷ that model's own clean (`bias=0`) baseline,
   1.0 = no effect — matching `outlier_experiment` convention. Absolute comparability is a bonus;
   the normalised number is what the hypotheses are stated in.

A raw-feature (unscaled) MLP arm in Phase 2 is the sanity check that scaling is not itself
driving any result.

---

## 2. Model + attacks

**One model, two attacks.** A single torch MLP wrapped in ART's `PyTorchClassifier` serves both
HSJ and PGD. Using sklearn's `MLPClassifier` for HSJ and a torch MLP for PGD would give the two
attacks *different weights*, confounding the white-box/black-box contrast that is half the point
of this thread.

| | config |
|---|---|
| architecture | `4 → 16 → 16 → 3`, ReLU, softmax out; Adam(lr=1e-2), 300 epochs, full-batch |
| determinism | `torch.manual_seed(42)` per fit; seed sweep is over the **CV split seed** |
| preprocessing | `StandardScaler` fit on train fold only |
| **HSJ** (black-box) | `norm=2, max_iter=10, max_eval=200, init_eval=50`, untargeted — **identical to every other grid in the project**, which is what keeps cross-model comparison valid |
| **DeepFool** (white-box) | `max_iter=100, epsilon=1e-6, nb_grads=3`, untargeted |

**Why DeepFool and not PGD.** This project's metric is the geometry of **minimum-norm**
adversarials: DTA walks to the nearest different leaf, HSJ binary-searches the nearest L2
crossing. PGD with a fixed `eps` is a *budget* attack — it stops anywhere inside the ε-ball, so
its spread measures the budget rather than the boundary. Measured, not assumed: a PGD arm at
`eps=3.0` returned mean perturbation **3.32** (saturated), versus HSJ's **1.33** on the same
folds. DeepFool takes the minimal linearised step across the nearest boundary and returns **1.29**
— matching HSJ, which is what "both are minimum-norm" should look like. A fixed-budget arm would
not have been comparable to any existing row in the project.

**Pipeline is otherwise unchanged** and must stay that way: attack only correctly-classified test
points, keep only successful (label-flipping) adversarials, cluster with
`OPTICS(min_samples=3, xi=0.05, min_cluster_size=3)`, average over clusters then folds.

---

## 3. The protocol arm (doubles as verification of the 2026-08-11 revision)

[`../defect_expansion_experiment/FINDINGS_imbalance.md`](../defect_expansion_experiment/FINDINGS_imbalance.md)
revises the project's headline claim: under **train-only** injection, coverage gap is *not*
accuracy-blind (0.95 → 0.71, minority recall → 0.14). But that run changed **two** things at once
versus the original grids — injection protocol (before-split → train-only) **and** model
(`max_depth=3` → `max_depth=None` overfit tree). The protocol explanation is therefore not yet
isolated.

**An MLP has no depth knob.** Running both protocols on identical MLP weights isolates the
protocol variable cleanly, on a model family where the tree-capacity explanation cannot apply.
This costs one extra factor level and settles an open question for the whole project.

| `protocol` | injection | matches |
|---|---|---|
| `before_split` | inject into the full dataset, then CV-split | grids 1–3, `model_family_experiment` |
| `train_only` | CV-split first, inject into the train fold, test fold stays clean | `outlier_experiment`, `defect_expansion_experiment` |

---

## 4. Grid (Phase 0)

Shared: `tc = 2` (virginica, contested; `{0,1,2}` sweep is Phase 1), `feat = 2` (petal length,
matching `model_family_experiment`), **10 CV-split seeds**, 5 folds, both attacks per cell.

**Coverage gap** — `bias ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9}` × `protocol ∈ {before_split, train_only}`,
dropping `(0, train_only)` since `bias=0` is protocol-independent → 11 × 10 = **110 cells**.

**Outlier** — always train-fold-only. Baseline `n_out=0`, plus
`direction ∈ {toward, outward, random}` × `k ∈ {2, 4, 6, 8}` × `n_out ∈ {5, 10}`
→ 25 × 10 = **250 cells**. `k`, `n_out` and `direction` levels mirror
`model_family_experiment`'s `OL_K`/`OL_N`/`OL_DIR` exactly, so every row is directly comparable to
[`FINDINGS_outlier.md`](../model_family_experiment/FINDINGS_outlier.md).

**360 cells → 720 rows** (one per attack), ~65 min measured.

## 5. Result schema

Reuses the established columns — `tacc, vacc, asucc, nadv, density, nclust, mean_dist,
clust_size, aiden_density, spread_c0..2, bias, tc, feat, seed, model, attack` — plus:

| column | meaning |
|---|---|
| `protocol` | `before_split` / `train_only` |
| `min_recall` | per-class recall of the depleted class `tc` — the clean discriminator identified by the imbalance findings |
| `perturbation` | mean `‖adv − x‖` in raw units, for the compression-ratio lineage |
| `spread_std_space` | spread measured *before* inverse-transform, to confirm §1 changes nothing qualitative |

Output: `mlp_experiment/results.parquet`, one row per (bias, protocol, attack, tc, seed) cell.

## 6. Hypotheses (pre-registered)

- **H1 (signal survives a global smooth boundary):** MLP+HSJ normalised spread rises
  monotonically with `bias` under `before_split`, CIs at 0.9 clearing the clean baseline.
  **Predicted TRUE** — a coverage gap is a *global structural hole*, which is exactly the class of
  defect that survived RandomForest ([`FINDINGS_coverage_gap.md`](../model_family_experiment/FINDINGS_coverage_gap.md)).
- **H2 (protocol isolation):** under `before_split`, `vacc` stays flat (within ~0.02 across the
  bias sweep); under `train_only`, `vacc` and `min_recall` fall. Confirming this on a model with
  **no depth parameter** isolates protocol from tree capacity and settles whether the
  accuracy-blind claim was a test-set artifact. Predicted TRUE.
- **H3 (white-box vs black-box):** DeepFool gives a cleaner/stronger coverage-gap trend than HSJ
  (higher Cohen's d, tighter CIs). **Genuinely uncertain** — unlike the outlier, coverage gap
  already survives black-box HSJ on RF, so this may come out null. A null here is informative:
  it would mean the white-box dependence found in Thread A is specific to *local* defects.
- **H6 (outlier — THE prediction, and the reason this thread is worth running):** the MLP shows a
  **non-null `outward` response**, unlike every model tested so far. Thread A's null is explained
  by *locality* — the tree never splits for a correctly-labeled point in already-correct empty
  space (Δleaves = 0), and the rbf kernel's influence decays to ~0 before reaching the probed
  boundary. An MLP's boundary is a global function of all weights, so a far same-labeled point
  should drag it. **Predicted: `outward` normalised spread clears 1.0 with CIs off the baseline** —
  a result no other model in the project can produce. Reference values to beat:

  | model (attack) | toward | outward |
  |---|---|---|
  | tree d3 / d10 (DTA, white-box) | 1.36× / 1.25× | **1.01× / 1.00× (null)** |
  | rbf-SVM (HSJ) | 0.98× (null) | **1.05× (null)** |
  | RandomForest (HSJ) | 0.92× (null) | **1.04× (null)** |

  Falsifier: if MLP `outward` also sits at ~1.0, locality is *not* the mechanism and the null is
  general — which would be a cleaner negative result than the current locality explanation.
- **H7 (outlier dose-response):** if H6 fires, the response rises with `k` and `n_out`. Thread A
  found trees only react at gross distance (`k ≥ 6`, ~6σ); a global boundary should react at
  *realistic* outlier magnitudes (`k = 2–3`), which is precisely the limitation
  [`FINDINGS_phase05.md`](../outlier_experiment/FINDINGS_phase05.md) F5 flags as motivating
  "more sensitive / global-boundary models".
- **H4 (no flat-facet pathology):** MLP+HSJ converges on ≥95% of cells, versus 53/90 hangs for
  RF/tree label noise. Would reopen ensemble-intractable questions for a later thread.
- **H5 (comparability):** after inverse-transform, MLP clean-baseline spread is the same order of
  magnitude as tree (~0.44) and SVM (~0.49). A wild value means §1 is wrong, not that MLPs are
  exotic — treat as a pipeline check, not a finding.

## 7. Success criteria (observable yes/no)

1. H1: normalised spread at `bias=0.9` vs `bias=0.1` under `before_split` — 95% CIs separate? ✅/❌
2. H2: `before_split` `vacc` range ≤ 0.02 **and** `train_only` `vacc` drop ≥ 0.10 with
   `min_recall` falling monotonically? ✅/❌
3. H3: Cohen's d reported for **both** attacks against the same cells, with a stated verdict on
   whether DeepFool > HSJ. ✅/❌
4. H4: hang/non-convergence rate recorded for both attacks. ✅/❌
5. H5: clean-baseline spread within ~2× of the tree/SVM baselines. ✅/❌
6. H6: outlier `outward` normalised spread at the strong cell (`k=8, n_out=10`) — 95% CI clears
   1.0, or does not. Either answer is reportable; state it against the tree/SVM/RF table. ✅/❌
7. H7: if H6 fires, the `k` at which the response switches on, versus the tree's `k ≥ 6`. ✅/❌

## 8. Confounds & controls

- **Scaler fit on the train fold only** — fitting on full data leaks test distribution into the model.
- **Identical weights across attacks** — HSJ and PGD must attack the *same* fitted model per fold,
  not two fits, or H3 is confounded by initialisation.
- **Clean `bias=0` baseline per (protocol, attack, seed)** — every normalised number needs its own
  denominator; do not share a baseline across protocols.
- **Never average accuracy over the injected factor** (repo-wide rule, `defect_expansion` PLAN §1).
- **PGD budget:** report the flip rate; if `asucc` is materially below the HSJ arm, `eps` was too
  small and the comparison is invalid.
- **10 seeds minimum.** Thread A's n=3 conclusion flipped when firmed to 10
  ([`FINDINGS_outlier.md`](../model_family_experiment/FINDINGS_outlier.md) F1) — do not repeat that.

## 9. Phased execution

- **Phase 0a — pipeline check.** Clean MLP, no defect: confirm `vacc` ≈ 0.95+, HSJ and PGD both
  produce adversarials, baseline spread passes H5. GO/NO-GO before any grid.
- **Phase 0b — coverage gap + outlier, both attacks** (the grid in §4). Answers H1–H7.
- **Phase 1** — `tc ∈ {0,1,2}` asymmetry; per-class recall plots; before-split vs train-only
  accuracy comparison written up as a standalone verification note for the wiki.
- **Phase 2** — raw-feature (unscaled) MLP sanity arm; architecture sensitivity (width/depth), since
  "capacity mediates the signal" is an established project theme (Finding 4). If H6 fires, width is
  the direct capacity test for the *global* mechanism.
- **Later / out of scope here** — label noise (H4 permitting: an MLP's smooth boundary may escape
  the flat-facet hang trap that made it intractable for RF/tree); the `k=2–3` realistic-magnitude
  regime if H7 shows the MLP reacts there.

## 10. Reuse / dependencies

- **New dep:** `torch` (added to `pyproject.toml`). ART's `PyTorchClassifier` +
  `ProjectedGradientDescent`.
- **Reuse by explicit path** (repo pattern — both files are named `run_experiment.py`, so
  `importlib.util.spec_from_file_location` avoids the name collision):
  `cluster_stats` from `../outlier_experiment/run_experiment.py`; `inject_cg` from
  `../model_family_experiment/run_experiment.py` as the `before_split` arm, and the train-only
  deletion from `../defect_expansion_experiment/run_imbalance.py` (`structure='spatial'`) as the
  `train_only` arm — reusing both keeps this thread's coverage gap *identical* to the ones it is
  being compared against.
- **Hang safety:** MLP is not expected to hang, so Phase 0 runs inline. If H4 fails, adopt the
  subprocess-timeout driver from `model_family_experiment/run_experiment.py` before scaling.
