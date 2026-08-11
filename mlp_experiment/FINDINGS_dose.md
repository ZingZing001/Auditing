# MLP — Outlier Dose Findings (2026-08-12)

Two follow-ups to `FINDINGS_phase0.md`, both asking **what happens when you inject more points**:

- **Ceiling** — distance fixed at `k=8` (far field), count swept to and past the class-size
  ceiling: `n_out ∈ {20, 30, 40, 60}` = 50/75/100/150% of the 40-point train-fold class.
  `{20,30,40}` mirror `model_family_experiment`'s `OL_N_CEILING` for direct comparability.
- **Fixed distance** — distance held at `k=2` and `k=4` (near field, where F5 found the response),
  count swept `n_out ∈ {1, 3, 5, 10, 20, 30, 40, 60}`, with `outward` at **matched count** as the
  control that separates *where the points land* from *how many there are*.

10 seeds throughout, `tc=2`, train-fold-only injection, 0 NaN.
Runner: `run_experiment.py --defect outlier [--ceiling|--dose]`. Analysis: `analyze.py`.

---

## D1 — Dose rescues the far-field signal, and this one IS accuracy-blind

`k=8`, DeepFool, `toward` (baseline `vacc` 0.957):

| dose | 5 (12%) | 10 (25%) | 20 (50%) | 30 (75%) | 40 (100%) | 60 (150%) |
|---|---|---|---|---|---|---|
| normalised spread | 1.10 | 1.01 | **1.09** ✱ | **1.11** ✱ | **1.10** ✱ | **1.10** ✱ |
| test accuracy | 0.949 | 0.951 | 0.947 | 0.945 | 0.948 | 0.946 |

(✱ = 95% CI clears 1.0.) HSJ is null at every dose (0.96–1.03) — white-box only, as in Phase 0.

**This qualifies the H6 refutation in `FINDINGS_phase0.md` F4.** That test used `n_out=10`
because it is the cell Thread A used, and the MLP is genuinely null there (1.01×). It is *not*
null at higher dose. The correct statement is that **the MLP needs roughly 5× the dose a tree
needs**, not that it cannot see outliers. F4's `outward` conclusion is unaffected — `outward` is
null at every dose too (0.92–1.03), so the locality refutation stands.

**A fourth ceiling shape, and the only accuracy-blind one:**

| model | shape at the ceiling | accuracy |
|---|---|---|
| tree + DTA | rises, then **collapses** to 1.04× at 100% | flat |
| rbf-SVM + HSJ | rises to 1.09× | **degrades 0.97 → 0.89** |
| RandomForest + HSJ | flat null throughout | flat |
| **MLP + DeepFool** | **rises to a stable ~1.10× plateau from 50% on** | **flat (0.945–0.951)** |

The SVM only obtained its ceiling signal by breaking accuracy — that is contamination, not a
hidden defect, and it fails the accuracy-blindness test the project rests on. The MLP holds
accuracy at 0.946 while absorbing 60 points at 8σ.

**Caveats.** The effect is modest (1.10× vs the tree's 1.36×) with CIs of roughly ±0.06. The
`n_out=10` dip (1.10 → 1.01 → 1.09) is not monotone; since `n=5`'s CI straddles 1.0, the honest
read is that **below 50% dose the effect is unstable**, not that there is a real dip. And at 150%
of class size "outlier" is the wrong word — 60 points at 8σ labelled virginica are a **second
virginica mode**, so the plateau may be measuring "this class now has two clusters".

## D2 — The near-field compression is a THRESHOLD, not a dial

`k=2` (0.88 cm out — 55% of the virginica→versicolor centroid gap, i.e. *inside* versicolor),
DeepFool, normalised spread:

| dose | 1 | 3 | 5 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|---|
| **toward** | 1.02 | **0.88** ▾ | 0.90 | **0.85** ▾ | **0.91** ▾ | **0.85** ▾ | **0.86** ▾ | 0.93 |
| `outward` (control) | 1.04 | 1.03 | 1.02 | 1.01 | 1.01 | 1.01 | 1.05 | 1.03 |

(▾ = 95% CI entirely **below** 1.0.) HSJ is null in both directions at every dose (0.95–1.07).

**The compression switches on at `n_out=3` and then flattens.** It does not deepen: n=3 gives
0.88, n=40 gives 0.86, and n=60 actually weakens to 0.93. So the answer to "does more
contamination compress further" is **no** — the signal is a threshold that trips almost
immediately and then saturates. Three points labelled virginica inside versicolor are enough; the
next thirty-seven add nothing.

**The matched-dose control is what makes this interpretable.** `outward` sits at 1.01–1.05 across
the entire sweep while `toward` sits at 0.85–0.91. Same number of injected training points, same
distance from the centroid, opposite direction — so the effect is about the points landing **in
the contested overlap**, not about enlarging the training set. This rules out the obvious
alternative explanation (more data → better-fit model → tighter geometry).

## D3 — And it stays accuracy-blind across the whole range

`k=2`, `toward` (clean baseline `vacc` 0.957, minority recall 0.942):

| dose | 0 | 3 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|
| `vacc` | 0.957 | 0.960 | 0.952 | 0.940 | 0.944 | **0.935** |
| minority recall | 0.942 | 0.942 | 0.940 | 0.932 | 0.944 | **0.942** |
| `nclust` | 3.96 | 4.08 | 3.86 | 3.44 | 3.76 | 4.12 |
| `clust_size` | 5.62 | 5.55 | 5.40 | 6.20 | 5.68 | 5.30 |

**A pre-registered prediction failed here, in the informative direction.** The expectation was
that past `n_out≈20` these points would stop being contamination and become a relabelled class,
breaking accuracy the way the SVM ceiling does. It does not happen: at 150% dose, accuracy has
fallen only 0.022 and **minority recall has not moved at all** (0.942 → 0.942). Overall accuracy
absorbs about a 6.6% dent in versicolor recall and nothing else.

So this is a defect that is **invisible to accuracy at any dose tested** while producing a stable,
control-validated geometric signal. That is the strongest form of the project's central claim —
and notably it is a *contamination* defect, not the coverage gap, whose accuracy-blindness
`FINDINGS_phase0.md` F1 showed to be a test-set artifact.

`nclust` (3.4–4.1) and `clust_size` (5.3–6.4) are flat throughout, so the compression is genuine
geometry and not OPTICS re-partitioning.

## D4 — At `k=4` the two attacks diverge

| `k=4`, `toward` | 1 | 3 | 5 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|---|
| DeepFool | **0.84** ▾ | 0.92 | 0.97 | 0.96 | **0.92** ▾ | 0.94 | 0.94 | **0.94** ▾ |
| HSJ | 1.03 | 0.99 | 1.01 | 1.02 | 1.05 | 1.08 | 1.06 | **1.11** ▴ |

At 109% of the centroid gap — points sitting just past versicolor's centre — DeepFool compresses
weakly while HSJ *expands*, reaching 1.11× at the top dose. Flagged, not claimed: it is one
significant cell at the end of a noisy trend, and the two attacks disagreeing is exactly what
Phase 0 F4 warned about with the `random` direction. Worth a dedicated grid if the divergence
matters; not worth a sentence in a report yet.

---

## What this changes

1. **`FINDINGS_phase0.md` F4/H6 needs the dose qualification** — "the MLP is null for outliers" is
   only true at `n_out ≤ 10`. The `outward` locality refutation is unaffected.
   *(Applied 2026-08-12: F4 now carries the correction and points here.)*
2. **The MLP has two distinct outlier responses, in opposite directions**: far-field dispersion
   (`k=8`, dose ≥ 50%, ~1.10×) and near-field compression (`k=2`, dose ≥ 3, ~0.86×). A scalar
   spread metric with a one-sided "is it above 1.0" test detects the first and **actively hides
   the second** — which is how H7 was misread in the first place.
3. **Matched-dose directional controls should be standard** for every future injection defect.
   Without `outward`, D2 would have been indistinguishable from "more training data".

## Next

- Sweep `k` finely between 2 and 4 to locate the compression→dispersion crossover (the geometry
  says it sits near `k=3.67`, where the injection crosses the versicolor centroid).
- Record **per-class recall for all three classes**, not just `tc` — D3 can only infer the
  versicolor dent from overall accuracy.
- Dedicated `random`-direction grid (Phase 0 **F6** — renumbered when the contamination finding
  was inserted as F5; it now has a dose curve, firing at 12–50% and decaying above that).
