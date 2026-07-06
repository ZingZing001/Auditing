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
