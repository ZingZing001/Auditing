# Experiments

Stress-tests of the adversarial-clustering hypothesis, built on the corrected
density metric (`util.cluster_density` / `util.adv_cluster_stats`). Two studies are
implemented: **Axis A** (metric settle: count vs density) and **Phase 1** (the
decisive localization gate from `docs/research_plan.md`).

These scripts **produce evidence** for research-design decisions; they do not make
them. The go/no-go gate call, the count-vs-density headline choice, and the OPTICS
default remain human calls.

## Setup

The repo targets Python >=3.12 with the deps in `pyproject.toml` (use `uv sync`).
A clean `uv` environment ships scikit-learn >=1.8 and threadpoolctl >=3, which avoids
the Win11 `wmic`/threadpoolctl crash seen with older stacks.

## Axis A - metric settle: cluster COUNT vs DENSITY

### `axis_a_synthetic.py` (network-free)
Controlled ground truth: `make_data` moons + the existing contiguous-slab bias
mechanism, sweeping bias 0.1->0.9. Reports per-bias mean count/density and the
Spearman rho of count-vs-bias and density-vs-bias.

```
.venv/Scripts/python experiments/axis_a_synthetic.py
```

### `axis_a_iris_pilot.py` (fetches UCI iris)
1. Regenerates a scoped `data/data_bias.parquet` (iris, 3 seeds, both bias types)
   with the corrected metric so `eda_bias.ipynb` runs end-to-end on real data.
2. Sweeps OPTICS `min_samples in {3,5,8}` and reports how the count/density-vs-bias
   trend and the mean cluster count move - i.e. how fragile the signal is at iris's
   small fold sizes.

```
.venv/Scripts/python experiments/axis_a_iris_pilot.py
```

## What Axis A found (this run)

- **Cluster count does not track bias.** Synthetic: rho=+0.07 (n.s.). iris: ~0 both
  sampling modes.
- **Density tracks bias but weakly / inconsistently, and not in the hypothesised
  direction.** Synthetic: rho=-0.41 (clusters get *looser* under more bias, the
  opposite of "tighter clusters in weak regions"). iris: rho=-0.18 (undersampling)
  vs +0.16 (oversampling) - sign flips.
- **The clustering is OPTICS-hyperparameter fragile** at this N: count rho swings
  -0.35 (min_samples=3) -> ~0 (5) -> undefined/single-cluster collapse (8).
- **The density definition `n_pairs / (total_sep + 1)` is size-confounded** at small
  clusters even after the Frobenius/pairing fix. Mean pairwise separation is a
  scale-invariant alternative worth evaluating.

Net: on this evidence the "denser adversarial clusters in biased regions" hypothesis
is not supported. A clean negative is a legitimate result; it argues for settling the
compactness definition and the OPTICS parameterisation before the heavier axes.

## Phase 1 - the decisive localization gate

### `phase1_localization.py` (network-free)

The spatial version of the question Axis A tested as a scalar (design:
`docs/research_plan.md` section 5). Moons (N=3000), a known coverage gap (slab
deletion with recorded region R), severity sweep 0.1-0.9, 3 seeds. For every test
point, an adversarial localization score (minus mean distance to the k nearest
adversarial examples) is ranked against ground-truth R-membership (AUROC), across
the model-mediation control matrix - tree x {DecisionTreeAttack, HopSkipJump},
RBF-SVM x {FGM, HopSkipJump}, MLP x HopSkipJump - and against model-free baselines
(data sparsity, kNN label disagreement) plus model uncertainty.

```
.venv/Scripts/python experiments/phase1_localization.py
```

### What Phase 1 found (this run)

Mean localization AUROC over 3 seeds x 5 severities (0.5 = chance):

| model x attack | adv density | data sparsity | kNN disagree | uncertainty |
|---|---|---|---|---|
| tree x DecisionTreeAttack | **0.237** | 0.982 | 0.768 | 0.499 |
| tree x HopSkipJump        | **0.286** | 0.982 | 0.768 | 0.499 |
| svm x FGM                 | **0.401** | 0.982 | 0.768 | 0.543 |
| svm x HopSkipJump         | **0.385** | 0.982 | 0.768 | 0.543 |
| mlp x HopSkipJump         | **0.397** | 0.982 | 0.768 | 0.349 |

1. **Adversarial examples ANTI-localize the coverage gap.** AUROC is well below
   chance for every model and attack: adversarial examples systematically *avoid*
   the weak region rather than concentrating in it.
2. **The anti-localization is robust** (consistent across two attacks on the same
   tree and three models under the same attack) - so by the plan's own
   model-mediation logic it is a *data-mediated* effect, just in the opposite
   direction of the hypothesis. Mechanism: attacks move points to the nearest
   decision boundary; boundaries live where data is dense and contested; deleting
   a region's data moves the boundary *away* from it. Coverage gaps repel
   boundaries, so adversarial examples mark dense contested regions, not holes.
3. **Baselines dominate.** Trivial data sparsity is near-oracle (0.982) by
   construction; even inverting the adversarial signal (~0.7 after 1-AUROC) would
   add nothing over it.
4. **Calibration is mostly flat** (only svm x FGM trends with severity, and its
   absolute AUROC is still below chance). Attack success is not the issue
   (HopSkipJump flips 100%, DecisionTreeAttack 90%).

**Gate verdict on this evidence: criteria 1 and 3 fail; criterion 2 holds but for
an inverted effect.** Per the plan this points to Phase 2' (pivot): either the
rigorous characterization - adversarial geometry reflects *boundary placement*,
which responds to coverage gaps by moving away, making adversarial examples the
wrong instrument for gap detection (simple density is the right one) - and/or the
ACG scalar strand. One motivated follow-up before closing the door: *label noise*
attracts boundary fragments rather than repelling them, so the taxonomy prediction
flips - adversarial examples may localize noise even though they cannot localize
gaps. Whether to run that is a human call.

Figures: `figures/phase1_auroc_bars.png`, `figures/phase1_calibration.png`,
`figures/scenes.png` (mechanism, all defect types).

## Phase 2 - defect taxonomy: the sign-flip test

### `phase2_defect_taxonomy.py` (network-free)

Same harness, region definition R, model x attack matrix and scoring code path as
Phase 1 (helpers imported from it), with the defect injector varied over the SAME
region: `coverage_gap` (delete training points in R - Phase 1 replication),
`label_noise_50` (flip 50% of training labels in R - contested region), and
`label_noise_100` (flip all - coherent systematic mislabeling). Mechanism
prediction: gaps repel boundaries (AUROC < 0.5), contested regions attract them
(AUROC > 0.5).

```
.venv/Scripts/python experiments/phase2_defect_taxonomy.py
.venv/Scripts/python experiments/phase2_entropy_baseline.py   # honesty check
```

### What Phase 2 found (this run)

Mean adv-density localization AUROC (3 seeds x 5 severities):

| model x attack | coverage_gap | label_noise_50 | label_noise_100 |
|---|---|---|---|
| tree x DecisionTreeAttack | 0.237 | **0.678** | 0.303 |
| tree x HopSkipJump        | 0.307 | **0.820** | 0.368 |
| svm x FGM                 | 0.401 | **0.655** | 0.506 |
| svm x HopSkipJump         | 0.389 | **0.627** | 0.366 |
| mlp x HopSkipJump         | 0.397 | **0.774** | 0.367 |

1. **The sign flips for contested noise - 5/5 combos** (gap < 0.5 < noise50),
   robust across models and attacks, i.e. data-mediated. The mechanism holds:
   adversarial examples read out *local boundary contention*.
2. **Coherent mislabeling (noise100) does NOT flip** (~0.37) - with 100% flips
   there is no contest; the boundary redraws cleanly around the region's edge
   (see `figures/scenes.png`, bottom row). Adversarial geometry detects
   *disagreement*, not "wrongness".
3. **Baseline blindness is complementary:** data sparsity sees gaps (0.98) but is
   blind to both noise types (0.51); kNN disagreement sees both noise types
   (0.99) but cannot separate contested from coherent; only the adversarial
   signal's sign separates noise50 from noise100 among the attack-based signals.
4. **Honesty check (`phase2_entropy_baseline.py`):** a no-model, no-attack kNN
   label-entropy statistic scores gap 0.63 / noise50 0.99 / noise100 0.41 - it
   *detects contested noise far better than the adversarial signal* (0.99 vs
   ~0.71) and reproduces the noise100 blindness. Detection, for every defect
   tested, is dominated by cheap training-set statistics.
5. Severity calibration: noise50 localization strengthens with severity for 4/5
   combos; noise100 weakens with severity (more coherent = cleaner redraw) -
   both consistent with the mechanism.

### Where this leaves the research claim

- The original hypothesis ("adversarial examples cluster in weak regions") is
  false as stated: the direction depends on the defect's *boundary action* -
  gaps repel, contested labels attract, coherent flips relocate cleanly.
- The sign-flip is a real, robust, mechanism-predicted phenomenon - but as a
  *detector*, adversarial geometry is dominated by cheap statistics (sparsity
  for gaps; disagreement/entropy/uncertainty for noise) on every defect tested.
- Coherent systematic mislabeling is invisible to EVERY internal signal tested
  (adversarial, sparsity, entropy, uncertainty ~ 0.4-0.5; disagreement only sees
  it because the test labels are clean) - detecting it requires external
  knowledge, an epistemic limit worth stating in the write-up.
- The defensible contribution is therefore the **mechanistic characterization**:
  adversarial-example geometry measures the model's boundary contention, not
  dataset quality; adversarial-based "data quality" tools should be expected to
  inherit exactly this signature. Whether to frame the thesis this way is a
  human call.

Figures: `figures/phase2_auroc_bars.png`, `figures/phase2_calibration.png`,
`figures/scenes.png`.

## Phase 3 - real datasets: where the story breaks

### `phase3_real_data.py`, `phase3_no_defect_control.py` (UCI fetches + sklearn)

Same protocol as Phase 2 (inject a known defect into a recorded region, attack
three model families, score localization against baselines), on three real
datasets: breast-cancer tumour records (569x30), banknote authentication
(1372x4), and a numeric subset of Adult census income (4000x6). Features
standardized; region = low tail of the most label-correlated feature (fixed
rule, no per-result cherry-picking); severities {0.3,0.5,0.7}; 3 seeds.

```
.venv/Scripts/python experiments/phase3_real_data.py
.venv/Scripts/python experiments/phase3_no_defect_control.py   # placebo check
.venv/Scripts/python experiments/visualize_real.py             # plain-language figures
```

### What Phase 3 found (this run) - the skeptical read

1. **The toy sign taxonomy does NOT transfer.** Census income: everything is
   chance (0.47-0.54) - no signal either way. Banknote and tumour records: the
   gap-avoidance REVERSES (adv 0.62-0.78 - adversarial points land *near*
   deleted regions). Only "contested labels attract" survives directionally on
   all three datasets, and weakly (0.54-0.75 means, not every combo).
2. **The placebo control deflates most of the apparent real-data "detection".**
   With NOTHING injected, the adversarial score against the same region is
   already 0.53 / 0.585 / 0.564 (census / banknote / tumour). The region sits in
   the low tail of the most label-correlated feature - i.e. near the natural
   class overlap, which is where adversarial points land anyway. Tumour-record
   "detections" (0.59-0.62) are barely above their 0.56 placebo.
3. **Attack practicality degrades on real data.** HopSkipJump vs the DECISION
   TREE does not reliably terminate in 30 dimensions (piecewise-constant output
   starves its step-size search; one run burned 7.5 CPU-hours before being
   killed - the black-box attack now runs only vs smooth models). Flip rates
   crater: SVM x HSJ succeeds on only 11-27% of points on census, 25-61% on
   tumour records (vs ~100% on the toy).
4. **Cheap statistics still dominate every dataset and defect** - neighbour-label
   disagreement 0.78-1.00, sparsity finds gaps, entropy finds contested labels,
   and even plain model confidence beats the adversarial signal almost
   everywhere it has signal at all.

**Bottom line: on real data the adversarial signal's direction is
dataset-dependent, much of its apparent magnitude is a region-placement
artifact, its attack machinery is fragile, and it never beats trivial
baselines.** This *strengthens* the central characterization - adversarial
examples map the model's boundary, and what the boundary does depends on the
dataset's local geometry - while closing the door on "adversarial localization
as a practical dataset auditor". Limitations of our own study: injected (not
naturally occurring) defects, one region shape, binary tasks, three datasets,
3 seeds.

Figures (plain-language, for non-specialists): `figures/real_verdict.png`
(the verdict grid incl. placebo column), `figures/real_bars.png` (vs simple
checks), `figures/real_projection.png` (2-D view of where attacks land).

### Aiden's metric, made visible

`visualize.py --density` renders the original OPTICS + pair-density method on the
controlled defects (`figures/density_clusters.png`,
`figures/density_vs_severity.png`). What it shows, concretely:

- The tree attack snaps adversarial points onto the tree's axis-aligned
  thresholds, so they form thin straight lines; OPTICS shatters those lines into
  13-46 micro-clusters, and both the count and the mean density swing wildly
  with `min_samples` (e.g. coverage-gap mean density 12.7 / 20.5 / 36.9 at
  min_samples 3 / 5 / 8) - the fragility, visually.
- Per-cluster density values span 2.6 to 877 within a single panel, and the
  severity chart is dominated by a ~1750 spike: when many attacked points snap
  to the SAME threshold, pairwise separations approach zero and the
  pairs/(separation+1) score explodes. The metric rewards near-duplicate
  stacking, not meaningful tightness.
- Neither cluster count nor mean density orders the defect types or tracks
  severity coherently - while the spatial layout in the same panels plainly
  shows the contested-region concentration that the localization AUROC captures.
  The information is in WHERE the points are; the scalar clustering summary
  destroys it.
