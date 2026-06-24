# Experiments

Stress-tests built on the corrected adversarial-density metric (`util.cluster_density`
/ `util.adv_cluster_stats`). The first axis ("metric settle") is implemented here;
later axes (multi-attack falsification gate, defect-type sweep, dimensionality) are
described in the project roadmap.

These scripts **produce evidence** for research-design decisions; they do not make
them. The count-vs-density headline choice and the OPTICS default remain human calls.

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
