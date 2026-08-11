# MLP Experiment — Phase 0 Findings (2026-08-11)

Torch MLP (4-16-16-3, ReLU), iris, 5-fold, **10 seeds**, `tc=2` (virginica), `feat=2`.
Every fold fits **one** model attacked by **both** HopSkipJump (black-box) and DeepFool
(white-box), so the attack contrast shares identical weights. Spread is inverse-transformed to
raw iris units and normalised **per seed** to that seed's clean baseline (1.0 = no effect).

Data: `results.parquet` (1440 rows / 720 cells, **0 NaN**) — the base grid plus the `--ceiling`
(dose to 150% of class size at `k=8`) and `--dose` (count sweep at fixed `k=2`, `k=4`, with a
matched-count `outward` control) extensions. Runner: `run_experiment.py`.
Analysis: `python analyze.py` · figures: `analysis.ipynb`. Contract: `PLAN.md`.

Clean baseline: `vacc` 0.957, spread 0.600 (HSJ) / 0.622 (DeepFool), perturbation ~1.36.
**H5 ✅** — same order as tree (~0.44) and SVM (~0.49), so the standardise-then-invert pipeline
does not distort the metric.

---

## F1 — HEADLINE: the "accuracy-blind" coverage gap is a test-set artifact (H2 ✅, decisively)

The same defect, the same model, the same weights — only *when* the deletion happens changes:

| | `before_split` (grids 1–3, model_family) | `train_only` (correct protocol) |
|---|---|---|
| `vacc` 0.1 → 0.9 | 0.958 → **0.999** (*rises*) | 0.944 → **0.732** |
| minority recall 0.1 → 0.9 | 0.944 → **1.000** | 0.888 → **0.198** |
| accuracy range across bias | 0.055 | 0.233 |

Under before-split injection accuracy does not merely *stay* flat — it **improves to 0.999 and
minority recall reaches 1.000**. That is the confound in its purest form: deleting the
low-petal-length virginica from the training set removes them from the *test* set too, so the
model is scored only on the easy remainder. "Accuracy stays flat while spread rises" was
substantially "we deleted the hard test cases."

**This confirms the 2026-08-11 revision and removes its confound.**
[`defect_expansion_experiment/FINDINGS_imbalance.md`](../defect_expansion_experiment/FINDINGS_imbalance.md)
reported train-only coverage gap costing accuracy (0.95 → 0.71) and minority recall (→ 0.14) —
but that run changed protocol **and** switched `max_depth=3` → overfit tree simultaneously, so
tree capacity was never excluded. **An MLP has no depth knob**, and reproduces the effect almost
exactly (0.944 → 0.732, recall → 0.198). The cause is the **protocol**, not tree capacity.

Consequence for the write-up: the headline claim in `README.md`, `.wiki/04-findings.md`,
`midyear_report.docx` and `presentation/deck.html` — "spread rises while test accuracy stays flat
(~0.96)" — does not survive a clean test set, on any model tested.

## F2 — The coverage-gap spread signal itself is real and survives a global boundary (H1 ✅)

Importantly, the signal does **not** depend on the confound. Under the *correct* train-only
protocol both attacks still rise monotonically with bias:

| attack / protocol | 0.1 → 0.9 (normalised) | Cohen's d |
|---|---|---|
| HSJ / `train_only` | 0.96× → **1.16×** | **+2.81** |
| DeepFool / `train_only` | 1.05× → 1.16× | +1.00 |
| DeepFool / `before_split` | 1.12× → 1.24× | +1.40 |
| HSJ / `before_split` | 0.99× → 1.02× (non-monotone) | +0.24 |

So the geometry genuinely tracks the defect on a globally-parameterised smooth boundary — the
fourth model family, after tree, rbf-SVM and RandomForest. What changes is the *claim*: the
signal is real, but it is **no longer accuracy-blind**, so it must be argued as complementary to
accuracy rather than as seeing what accuracy cannot.

## F3 — The bias-0.9 HSJ drop is confirmed as test-set depletion (a previously unverified claim)

HSJ/`before_split` reproduces the exact non-monotonic drop seen on Tree+HSJ: **1.19× at bias 0.7
→ 1.02× at 0.9**. Under `train_only` — where the test fold keeps its virginica — the drop
**disappears** and the curve keeps rising (1.11× → 1.16×).

The 2026-07-30 session flagged this mechanism as asserted but unverified ("the per-point tracing
needed to verify this mechanism was not completed", and the DTA-determinism story was retracted
as within-CI). Toggling only the protocol isolates it: **the drop is caused by the depleted class
running out of attackable test points**, and it is attack-specific (HSJ's stochastic init, not
DeepFool's deterministic gradient step — DeepFool/`before_split` moves 1.27× → 1.24×, within CI).
Confirmed on a model family that shares no structure with the decision tree.

## F4 — H6: `outward` is refuted robustly; `toward` fires, but only at ~5× the tree's dose

Pre-registered prediction: a global boundary should be dragged by a far same-labeled point, so
`outward` should fire where every local model is null.

**`outward` does not fire — at any distance, at any dose, under either attack.** Now tested to
`n_out=60`, i.e. **150% of the target class**, at `k=8` (3.53 cm, genuine far field):

| dose (% of class) | 12% | 25% | 50% | 75% | 100% | 150% |
|---|---|---|---|---|---|---|
| DeepFool `outward` | 1.00 | 0.97 | 1.03 | 0.92 | 0.95 | 1.02 |

**The locality explanation is therefore wrong.**
[`outlier_experiment/FINDINGS_phase05.md`](../outlier_experiment/FINDINGS_phase05.md) F4 attributes
the rbf-SVM `outward` null to the Gaussian kernel's influence decaying to ~0 before reaching the
probed boundary, and names a global-boundary model as the clean test. The MLP is that test, it is
null under a white-box gradient attack, and it stays null when the "outliers" outnumber the class
itself. The `outward` null is **general**, not a kernel artifact. The proposed linear-SVM
follow-up is not worth running.

### ⚠️ Correction — `toward` DOES fire on the MLP; this document's first draft got it wrong

An earlier version of F4 concluded that "the outlier signal is specific to axis-aligned partition
models under white-box attack", because MLP `toward` is null at the strong cell. **That cell is
`n_out=10` — 25% of the class — which is simply too small a dose for this model.** At ≥50% dose
DeepFool `toward` reaches a stable **~1.10×** with CIs clearing 1.0, while test accuracy holds at
0.945–0.951 against a clean 0.957. The MLP needs roughly **5× the dose a tree needs** (the tree
responds from `n_out≈1` and saturates by ~5); it is not blind to outliers.

Generalising from one cell was the error — the exact dose limitation flagged when F4 was written.
**`outward` is unaffected and the locality refutation stands.** HSJ remains null at every dose,
so Thread A's white-box dependence holds here too.

Full account, including the fourth ceiling shape and its caveats:
[`FINDINGS_dose.md`](FINDINGS_dose.md) **D1**.

**Correction (H7).** An earlier draft also recorded H7 as "no dose-response at all, flat in `k`
for both attacks and both directions". Also wrong: it scanned only for spread *above* 1.0 and
filed everything else as null, discarding a clean trend running in the **compression** direction.
See F5 and [`FINDINGS_dose.md`](FINDINGS_dose.md) D2.

## F5 — Near-field CONTAMINATION produces the opposite sign: compression, not expansion (H7 ↺)

At small `k` the `toward` injection does not produce an outlier at all. Measured geometry:

| | distance | as % of the virginica→versicolor centroid gap (1.62 cm) |
|---|---|---|
| `k=2` | 0.88 cm | **55%** — deep inside the versicolor cloud |
| `k=4` | 1.77 cm | **109%** — past versicolor's centre |
| `k=8` | 3.53 cm | 218% — out the far side, genuine empty space |

So `k=2, toward` injects 10 points **labelled virginica but physically sitting inside
versicolor**. That is contamination — label noise placed precisely in the contested overlap —
not an anomaly. DeepFool responds, and responds *downward*:

| attack | k=2 | k=4 | k=8 |
|---|---|---|---|
| **DeepFool** (`n_out=10`) | **0.85× [0.79, 0.91]** | 0.96× [0.90, 1.01] | 1.01× [0.91, 1.11] |
| HSJ (`n_out=10`) | 1.02× [0.96, 1.09] | 1.02× [0.90, 1.14] | 1.01× [0.91, 1.11] |

Monotonic in `k`. `nclust` (~3.8) and `clust_size` (~5.7) barely move, so these are genuinely
tighter clusters, not a clustering artifact.

**A matched-count `outward` control confirms this is about *where* the points land, not *how
many*** — `toward` sits at 0.85–0.91 across the dose sweep while `outward` at the identical count
is flat at ~1.02. The compression is also a **threshold, not a dial**: it trips at `n_out=3` and
then saturates rather than deepening. Full dose sweep and the accuracy-blindness check:
[`FINDINGS_dose.md`](FINDINGS_dose.md) **D2**, **D3**.

**Mechanism.** Contamination drags the boundary into the dense contested band and anchors it
there. DeepFool walks each test point to its *nearest* boundary, so every adversarial point piles
into that band → tighter cloud. HSJ starts from a random far point and estimates direction by
Monte-Carlo sampling, so it does not reliably find the nearest boundary and the compression is
smeared away. Same white-box/black-box asymmetry as Thread A's `toward` signal on trees, but
expressed as compression rather than expansion.

**Accuracy is an inverted U, which corroborates the mechanism:** clean 0.957 → `k=2` 0.953 →
**`k=4` 0.907 (worst)** → `k=8` 0.951. At `k=2` the bad points sit where the boundary already is,
so the model absorbs them cheaply; at `k=4` they are planted in the heart of versicolor, maximum
confusion; at `k=8` they are isolated again and get carved off without disturbing the real
boundary.

This maps the outlier/contamination frontier that
[`FINDINGS_phase05.md`](../outlier_experiment/FINDINGS_phase05.md) flagged as "itself worth
mapping" after observing accuracy dip at extreme `toward` injection. The frontier for `tc=2` sits
near **`k ≈ 3.7`** (where `k·s` equals the centroid gap).

**Consequence for the F5 question in Thread A** — whether a more sensitive model detects
*realistic* 2–3σ outliers: **still no**. At `k=2–3` the injected point is not a realistic outlier,
it is contamination in a neighbouring class, and what fires there is a different defect wearing
the outlier's clothes. The MLP's genuine far-field outlier response (F4) needs `k=8` *and* ≥50%
dose — far outside anything a real dataset would contain.

**But the contamination defect it *does* detect is the more valuable finding.** Per
[`FINDINGS_dose.md`](FINDINGS_dose.md) D3, `k=2, toward` stays invisible to accuracy at every dose
tested (minority recall 0.942 → 0.942 at 150% dose) while producing a stable, control-validated
geometric signal. That is the project's central claim in its strongest form — and unlike the
coverage gap, whose accuracy-blindness F1 showed to be a test-set artifact, this one survives a
clean test set.

**The sign is the diagnostic.** Taken with F4, the MLP separates two defects by the *direction*
of the geometric change: contamination in the overlap → **compression** (0.85×); far-field
same-labelled outliers at high dose → **expansion** (1.10×). Both are white-box-only and both are
accuracy-blind-ish, so a scalar magnitude alone would conflate them; the sign does not.

## F6 — `random` direction fires at low-to-mid dose, and now has a dose curve

| `k=8`, DeepFool `random` | 12% | 25% | 50% | 75% | 100% | 150% |
|---|---|---|---|---|---|---|
| normalised spread | **1.10** ✅ | **1.15** ✅ | **1.12** ✅ | 1.07 | 1.05 | 1.07 |

Three consecutive doses clear 1.0, then it decays as dose rises — the opposite shape to `toward`,
which switches on at 50% and holds. HSJ is null throughout. This echoes the one non-null RF cell
in Thread B (`random` 1.13× [1.02,1.24],
[`FINDINGS_outlier.md`](../model_family_experiment/FINDINGS_outlier.md) F2), so two different
models now show it.

**Upgraded from "lead" to "weak but replicated result"** — it survives a dose sweep rather than
sitting in one cell, which was the multiple-comparisons worry. Still the weakest effect here, and
the decay at high dose is unexplained: scattering `n` points isotropically probably stops looking
like anomalies and starts looking like added variance once `n` approaches the class size.

## F7 — H4 ✅: the MLP escapes the flat-facet hang trap entirely

**0 NaN cells out of 720**, attack success 0.944 for both attacks, ~27.9 adversarials per fold.
Compare RF/tree + HSJ on label noise, where **53 of 90 cells hung** and Thread B was abandoned
([`FINDINGS_label_noise.md`](../model_family_experiment/FINDINGS_label_noise.md)). A smooth
boundary gives HSJ a well-conditioned direction estimate, so **label noise via HSJ is tractable on
an MLP** — a question the project closed as a hang trap is reopened for this model family.

---

## Verdict against PLAN.md §7

| # | criterion | result |
|---|---|---|
| 1 | H1 — CG spread separates low vs high bias | ✅ both attacks, both protocols |
| 2 | H2 — protocol isolates the accuracy effect | ✅ decisively; F1 |
| 3 | H3 — DeepFool vs HSJ verdict stated | ✅ DeepFool cleaner on `before_split` (d +1.40 vs +0.24); **HSJ far stronger on `train_only` (d +2.81 vs +1.00)** — no consistent white-box advantage for a *global* defect, unlike the outlier case |
| 4 | H4 — hang rate recorded | ✅ 0/720 |
| 5 | H5 — baseline comparable to tree/SVM | ✅ 0.60 vs 0.44 / 0.49 |
| 6 | H6 — outward clears 1.0 | ❌ **refuted, robustly** — null at every distance and every dose out to 150% of the class; F4 |
| 7 | H7 — dose-response onset vs tree's k≥6 | ⚠️ **two responses, neither as predicted** — far-field `toward` needs `k=8` **and ≥50% dose** (1.10×, accuracy-blind, DeepFool only); near-field contamination compresses to 0.85× with a matched-count `outward` control; F4, F5 |

## Next

1. **Propagate F1.** The accuracy-blind claim needs revising in `README.md`,
   `.wiki/04-findings.md`, the report and the deck. F1 supplies the controlled evidence the
   imbalance finding was missing. *(Held pending your call — you asked to verify first.)*
2. **Phase 1** — `tc ∈ {0,1,2}` asymmetry, and per-class recall curves for the protocol contrast.
3. **Label noise on the MLP** — newly tractable per F7, and the one defect this thread has not
   touched. F5 makes this more interesting, not less: `k=2, toward` *is* spatially-localised label
   noise, and it produced the thread's only clean dose-response. Random (unlocalised) label noise
   is the natural contrast.
4. **Resolve `k` in units of the class gap, not class σ.** F5 shows `k` conflates two defects:
   below `k ≈ 3.7` it is contamination, above it is a true outlier. Reporting `k·s / centroid-gap`
   alongside `k` would stop this being rediscovered.
5. **Re-check the tree and RF at ≥50% dose on `toward`.** This is the generalisable lesson from
   the F4 correction: MLP `toward` reads null at 25% and fires at 50%+. RandomForest was recorded
   as "flat null throughout" — but if its ceiling grid was concentrated at low dose, it may have
   the same blind spot this document did. Cheap to verify and it would change a published claim.
6. **Report the sign, not just the magnitude.** F4 and F5 give opposite-signed responses to two
   different defects on the same model and attack. A scalar "spread went up by X" conflates them;
   `>1` vs `<1` separates them. This is the closest thing this thread has to a positive
   methodological contribution.
7. **Do not** run the linear-SVM outward test — F4 makes it uninformative now that `outward` is
   null out to 150% dose on a global model.
