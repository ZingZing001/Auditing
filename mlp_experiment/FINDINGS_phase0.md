# MLP Experiment — Phase 0 Findings (2026-08-11)

Torch MLP (4-16-16-3, ReLU), iris, 5-fold, **10 seeds**, `tc=2` (virginica), `feat=2`.
Every fold fits **one** model attacked by **both** HopSkipJump (black-box) and DeepFool
(white-box), so the attack contrast shares identical weights. Spread is inverse-transformed to
raw iris units and normalised **per seed** to that seed's clean baseline (1.0 = no effect).

Data: `results.parquet` (720 rows / 360 cells, **0 NaN**). Runner: `run_experiment.py`.
Analysis: `python analyze.py`. Contract: `PLAN.md`.

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

## F4 — H6 REFUTED: the MLP does *not* see outward outliers, so locality is not the explanation

Pre-registered prediction: a global boundary should be dragged by a far same-labeled point, so
`outward` should fire where every local model is null. It does not.

Strong cell (`k=8, n_out=10`), normalised, 95% CI:

| model (attack) | toward | outward |
|---|---|---|
| tree d3 (DTA, white-box) | **1.36×** | 1.01× null |
| tree d10 (DTA, white-box) | **1.25×** | 1.00× null |
| rbf-SVM (HSJ) | 0.98× null | 1.05× null |
| RandomForest (HSJ) | 0.92× null | 1.04× null |
| **MLP (HSJ)** | **1.01× [0.91,1.11] null** | **1.01× [0.92,1.10] null** |
| **MLP (DeepFool, white-box)** | **1.01× [0.91,1.11] null** | **0.97× [0.90,1.03] null** |

Two conclusions, both cleaner than the hypothesis they replace:

1. **The locality explanation is wrong, or at least insufficient.**
   [`outlier_experiment/FINDINGS_phase05.md`](../outlier_experiment/FINDINGS_phase05.md) F4
   attributes the rbf-SVM `outward` null to the Gaussian kernel's influence decaying to ~0 before
   reaching the probed boundary, and names a global-boundary model as the clean test. The MLP is
   that test, and it is null too — **including under a white-box gradient attack**. The
   `outward` null is *general*, not a kernel-locality artifact. The proposed linear-SVM follow-up
   is unlikely to be worth running.
2. **The outlier signal looks specific to axis-aligned partition models under white-box attack.**
   The MLP shows no `toward` response either (1.01×), where trees show 1.25–1.36×. Combined with
   RF's null, the tree result now looks like a property of *how a tree carves space* — a lone far
   point forces a dedicated axis-aligned tendril that the minimal-perturbation attack walks into —
   rather than a general property of adversarial geometry.

The H6 refutation is unaffected by F5 below: `outward` is null at *every* distance, and `toward`
at `k=8` — genuine far-field, 3.53 cm from the centroid — is null too.

**Correction (H7).** An earlier draft of this document recorded H7 as "no dose-response at all,
flat in `k` for both attacks and both directions". That was wrong. It scanned only for spread
*above* 1.0 and filed everything else as null, discarding a clean monotonic trend running in the
**compression** direction. See F5.

## F5 — The MLP detects near-field CONTAMINATION by compressing, not far-field outliers (H7 ↺)

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

Monotonic in `k`, and dose-dependent at fixed `k=2`: `n_out=5` → 0.90× [0.80, 1.01],
`n_out=10` → **0.85×**. `nclust` (~3.8) and `clust_size` (~5.7) barely move, so these are
genuinely tighter clusters, not a clustering artifact.

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
*realistic* 2–3σ outliers: still **no**. A realistic outlier is far-field and same-labelled; the
MLP is null there. What it detects at `k=2` is a different defect wearing the outlier's clothes.

## F6 — Tentative: `random` direction is the only far-field outlier cell that moves

DeepFool `random` at the strong cell is **1.15× [1.04,1.27]** while HSJ `random` is 1.01× (null).
This echoes the one non-null RF cell in Thread B (`random` **1.13× [1.02,1.24]`,
[`FINDINGS_outlier.md`](../model_family_experiment/FINDINGS_outlier.md) F2) — two different models
showing a weak effect in the same direction.

**Treat as a lead, not a result.** It is a single cell among six direction × attack combinations
with no dose-response behind it, so it is exactly the shape of a multiple-comparisons artifact.
Thread A's own lesson (an n=3 conclusion that flipped at n=10) argues for a dedicated
`random`-only grid before anything is claimed.

## F7 — H4 ✅: the MLP escapes the flat-facet hang trap entirely

**0 NaN cells out of 360**, attack success 0.944 for both attacks, ~27.5 adversarials per fold.
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
| 4 | H4 — hang rate recorded | ✅ 0/360 |
| 5 | H5 — baseline comparable to tree/SVM | ✅ 0.60 vs 0.44 / 0.49 |
| 6 | H6 — outward clears 1.0 | ❌ **refuted**; F4 |
| 7 | H7 — dose-response onset vs tree's k≥6 | ⚠️ **not as predicted, but not null** — no far-field response at any `k`; a monotonic *compression* response to near-field contamination at `k=2` (0.85×, DeepFool only); F5 |

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
5. **Ceiling extension** (`--ceiling`, `n_out ∈ {20,30,40,60}` at `k=8`) — the honest scope limit
   on F4: our maximum dose was 25% of the class, and Thread A found the tree's response plateaus
   between 25–75% and collapses at 100%, while rbf-SVM was null at low dose but rose to 1.09× at
   the ceiling *with accuracy degrading*. The MLP is the other global model, so its ceiling
   behaviour is the direct test.
6. **Do not** run the linear-SVM outward test (F4 makes it uninformative); consider a dedicated
   `random`-direction grid instead (F6).
