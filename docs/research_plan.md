# Research Plan — Adversarial Learning as a Dataset-Quality Diagnostic

*Adopted direction (validity-first localization + discrimination study with
baselines and a go/no-go gate). Items marked **[ratify]** are research-design calls
for Aolin + research partner + supervisor (Joerg Wicker), not settled here.*

---

## ADDENDUM (post-gate results — 2026-07-06): the plan's empirical questions are
## answered; the claim redefines

Phases 1 and 2 have been run (full tables and figures:
`experiments/README.md`, `experiments/figures/`). Outcome:

1. **The gate failed for the original claim** — adversarial examples
   *anti-localize* coverage gaps (AUROC 0.24–0.40 across all 5 model×attack
   combos vs 0.98 for a trivial density baseline), robustly (= data-mediated,
   not a tree-attack artifact).
2. **The mechanism's predicted sign-flip is real** — contested label noise
   *attracts* adversarial examples (AUROC 0.63–0.82, 5/5 combos), coherent
   100%-flips do not (~0.37). Adversarial geometry reads **local boundary
   contention**: gaps repel boundaries, disagreement attracts them, coherent
   errors relocate them cleanly.
3. **Detection is dominated by cheap statistics on every defect tested**
   (sparsity 0.98 on gaps; kNN disagreement/label-entropy ~0.99 on noise;
   entropy also reproduces the noise-100 blindness with no model or attack).
4. **Coherent systematic mislabeling is invisible to every internal signal**
   — an epistemic limit, not a method failure.

**Redefined thesis claim [ratify]:** a controlled, mechanistic characterization
of what adversarial-example geometry actually measures — the model's boundary
contention, not dataset quality — with the sign-flip taxonomy as the central
result and the baseline-dominance analysis as the practical guidance
(audit datasets with cheap statistics; adversarial "quality" tools measure the
model). A rigorous result of this shape is publishable and thesis-complete.

**Remaining work under the redefined claim:** harden the taxonomy (feature
corruption, outliers, non-slab region shapes, a second dataset family,
a dimensionality spot-check), formalise the boundary-action framing, and write
up. The ACG strand stays available as a complementary scalar **[ratify]**.

The original plan below is retained for provenance; its Phase 2′
(characterization) branch is now the active path, entered with a *positive*
mechanistic finding rather than a bare negative.

---

## 1. Status

- Density metric was doubly-broken (Frobenius + pairing); fixed and tested.
- **Axis A (cheap falsification) came back negative:** on controlled synthetic *and*
  real iris data, cluster **count** does not track injected bias; **density** trends
  weakly, inconsistently, and in the wrong direction; the signal is OPTICS-fragile;
  the density definition is size-confounded.
- **But that test was underpowered for the real hypothesis:** it used a *global
  scalar*, *tiny* data (~15 adversarial pts/fold), and a *single model+attack*. The
  hypothesis is *spatial*. So it flags the premise as shaky without decisively
  testing it. The plan below runs the decisive version.

## 2. The scientific question (sharpened)

> When a dataset has a **known** structural weakness in a region R, do adversarial
> examples **spatially concentrate in R**, *robustly across models and attacks* and
> *beyond what simple model-free baselines already reveal*? And does the geometry
> **distinguish** structured coverage gaps from unstructured noise?

**Key insight that organises the whole study — model mediation.** A decision-tree
attack puts adversarial points on the tree's boundaries, so "adversarial clustering"
is partly the *model's* partition, not the *data*. That is not automatically fatal:
a data weakness legitimately *causes* a boundary shift that *causes* adversarial
concentration. The test that separates signal from artifact is therefore:

- If **many different models** (tree, SVM, NN) trained on the same weakened data all
  concentrate adversarial examples in R → the weakness is a **data property**
  (model-mediated) → real signal.
- If **only the tree + tree-specific attack** does → **artifact**.

This is why multi-model / multi-attack control is a *precondition*, not a later axis.

## 3. Measurement framework (define before experimenting)

**Ground truth.** Synthetically injected defect with a **known region R and known
severity s**. R is the feature-space footprint of the injected problem (e.g. the
deleted slab for a coverage gap; the perturbed indices for corruption).

**Primary metric — localization.** Per test point (or grid cell), ground-truth label
= "in R". Score each point by an adversarial-derived quantity (local adversarial-
example density; adversarial perturbation magnitude; adversarial-cluster membership).
Report **localization AUROC** = how well the adversarial score ranks R-membership.
Compare against the same AUROC for each baseline.

**Secondary metric — severity calibration.** Does localization AUROC (or an effect
size) increase monotonically with injected severity s? (This is the refined,
spatial version of what Axis A tried to do with a scalar.)

**Baselines (same localization metric) — the "compared to what".**
- data density (k-NN distance to / kernel density of training data),
- k-NN **label disagreement** rate,
- confident-learning self-confidence / **AUM**-style margin,
- (oracle reference) local test error.

**Adversarial method must clear all baselines to justify itself.** **[ratify]** which
baselines are in scope.

## 4. Phase 0 — define + scope (design; minimal code)

- Freeze the claim (localization? + calibration? + discrimination?), the R/s
  definitions per defect type, and the metrics above.
- Short lit-search/positioning pass (§8) to make sure the framing and baselines are
  right and novel.
- Choose the data regime: **synthetic-first** (recommended — gives exact R and s),
  UCI only later. **[ratify]**

## 5. Phase 1 — the decisive test (the GATE)

Adequately-sized, controllable **2D synthetic** data (e.g. moons / blobs /
checkerboard, **N ≈ 2000–5000** so each fold has hundreds of adversarial points, not
~15). Inject a known **coverage gap** (existing slab mechanism, but now recording R).

**Factors.**
- Attacks: `DecisionTreeAttack` (tree-specific) vs a decision-based **black-box**
  attack (HopSkipJump / Boundary); FGSM/PGD for differentiable models.
- Models: DecisionTree, RBF-SVM, small MLP.
- Severity sweep: bias 0.1 → 0.9.
- Baselines: as §3.

**Outputs.** Localization AUROC per (model × attack × severity), each with its
baseline AUROCs; calibration curves (AUROC vs severity).

**Gate criteria (proceed only if all hold):**
1. **Localizes:** adversarial localization AUROC clears chance and the best model-free
   baseline by a meaningful margin.
2. **Robust:** the effect holds (same sign, comparable magnitude) across ≥2 attacks
   **and** ≥2 models — i.e. it is not a tree+tree-attack artifact.
3. **Calibrated:** localization strength increases with severity.

If the gate **passes** → Phase 2. If it **fails** → Phase 2′.

## 6. Phase 2 — defect taxonomy (the positive contribution, if gate passes)

Hold the pipeline fixed; vary the *defect type*:
- structured **coverage gap** (slab deletion),
- **label noise** (random / systematic flips),
- **feature corruption** (perturb a fraction of values),
- **outliers** (out-of-distribution injections).

Claim to test: the adversarial-geometry signature **discriminates** structured
coverage gaps from unstructured noise (a sharper, more defensible claim than "measures
quality"). Report per-defect localization + a discrimination analysis.

## 7. Phase 2′ — pivot (if gate fails)

Both are legitimate thesis outcomes:
- **Write the negative / characterization:** *when and why* adversarial geometry does
  not reflect dataset quality, with the model/attack confound analysis as the
  contribution. Valuable in an over-claimed subfield.
- **Revive ACG:** the scalar performance gap between a model trained on original vs
  adversarially-augmented data, as a *dataset-difficulty* measure, benchmarked
  against dataset cartography / AUM. This is a cleaner scalar (no fragile clustering)
  and closer to the original brief's *model-certification* score. **[ratify]** keep
  or drop.

## 8. Phase 3 — scale / dimensionality / real data

Only if Phase 1–2 justify it. Use dimension-robust **relative** measures
(adversarial density vs data density in the same region; Hopkins; rank-based), since
absolute clustering degrades with dimension. Move to adequately-sized real datasets.

## 9. Threats to validity (state these explicitly in the write-up)

- **Model-mediation confound** (addressed by the multi-model/attack gate).
- **Boundary triviality:** adversarial perturbation is small near *any* boundary;
  localization must show the boundary coincides with R specifically, above baseline.
- **Small-N clustering instability** (addressed by N ≥ 2000 and by preferring
  density/rank measures over OPTICS cluster counts).
- **Metric artifacts:** the `n_pairs/(total_sep+1)` density is size-confounded; use a
  scale-invariant compactness (mean pairwise separation) or a per-point density.
- **Ground-truth leakage:** R is defined from the injection, so avoid using injection
  parameters in the score; score only from model/attack/data.

## 10. Scope for a Part 4 (Honours) timeline

Do **Phase 0 + Phase 1 well** (one rigorous, gated study with baselines) rather than
all four original axes thinly. That alone — localization + confound control +
baselines on controlled data — is a complete, defensible thesis, positive or negative.
Phase 2 is the stretch goal; Phase 3 is future work.

## 11. Related work to engage (verify + position against)

- **Confident Learning / Cleanlab** (Northcutt et al.) — label-error detection.
- **Dataset Cartography** (Swayamdipta et al., 2020) — training-dynamics region maps.
- **Area-Under-Margin / AUM** (Pleiss et al., 2020) — mislabel identification.
- **Influence functions** (Koh & Liang, 2017).
- **Katerina Dost** (Auckland) — bias / data-quality detection (local prior).
- Adversarial-robustness ↔ data-geometry (boundary tilting; adversarial spheres).

*(Pointers to review and cite, not verified claims; the lit-search is a Phase-0 task.)*

## 12. Open decisions for the supervisor

- Ratify the **reframe** (localization + discrimination + baselines, gated).
- Accept a **rigorous negative / characterization** as a valid outcome.
- **Baselines** in scope; **data regime** (synthetic-first vs UCI).
- Keep **ACG** as the pivot fallback or drop.
- Reconcile the thesis question with the **original brief** (model-certification score
  vs dataset-quality measurement) — pick one and state it.
