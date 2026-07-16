# Auditing

## Black-box adversarial-cluster reliability study

`notebooks/black_box_density_reliability.ipynb` is a fresh, self-contained
experiment for measuring the compactness of successful adversarial-example
clusters. It trains a decision tree, MLP, and RBF SVM on two-moons data, then
attacks only a label-returning black-box API with HopSkipJump and Boundary
Attack.

The notebook deliberately does not use ART's scikit-learn estimator wrappers
for attacks. The attack receives neither gradients, model parameters, tree
structure, nor confidence scores.

To make ART's stochastic attacks repeatable, each attack is initialized with a
deterministically selected point from the same attack batch that the
label-only oracle already predicts as another class. This replaces ART's
unseeded random initialization; all model interactions still use the
label-only oracle.

### Run it

With [uv](https://docs.astral.sh/uv/) installed:

```powershell
uv sync
uv run jupyter nbconvert --to notebook --execute notebooks\black_box_density_reliability.ipynb --output black_box_density_reliability.executed.ipynb --ExecutePreprocessor.timeout=900
```

The notebook defaults to a bounded quick run. Set `QUICK_RUN = False` in its
configuration cell for the larger query budget and sample count. It writes
per-sample outcomes, attack and cluster summaries, and figures to
`data\black_box_density\`.

### What it measures

For each model and attack, the notebook retains only successful adversarial
outputs from initially correct examples, then clusters them with OPTICS. For
each non-noise cluster `C_j` of size `n_j`, it calculates:

```text
cluster_density(C_j) =
  choose(n_j, 2) / (1 + sum_{a < b} ||z_a - z_b||_2)
```

It reports the unweighted mean across clusters. This is Aidan's intended
`pair_count / (total_separation + 1)` adversarial-cluster density, corrected
to use each unique pair's separation. The prior implementation used the norm
of two stacked coordinates, which measured their position relative to the
origin rather than the distance between them; it also repeated and self-paired
points.

Higher adversarial-cluster density means the found adversarial outputs are
more tightly concentrated. It is an attack-outcome diagnostic, not a
pre-attack vulnerability predictor or a reliability score.

The notebook reports:

- clean accuracy and query count;
- attack success rate and empirical robust accuracy at the configured L2
  budget;
- median successful boundary distance;
- successful adversarial point, cluster, clustered-point, and noise-point
  counts, together with corrected adversarial-cluster density; and
- the conservative per-model reliability score: the worst empirical robust
  accuracy across the two black-box attacks.

The score is an empirical result for the specified data distribution, attack
budgets, and query budgets. It is not a formal robustness certificate.

## Real-data replication: Forest CoverType

`notebooks/real_data_cluster_density.ipynb` replicates the cluster-density
experiment using Forest CoverType, a real-world cartographic dataset with
581,012 observations. It selects a fixed, stratified subset of 15,000 examples
from cover types 1 and 2, and uses the first ten continuous terrain features:
elevation, aspect, slope, hydrology distances, road distance, hillshade
measurements, and fire-point distance.

The discrete wilderness-area and soil-type indicators are excluded because a
continuous L2 perturbation could make one-hot fields invalid. The attack domain
is the scaled `[0, 1]^10` feature box; this preserves observed feature ranges
but not all physical feature correlations.

The model suite includes:

- a decision tree;
- an MLP; and
- an SVM explicitly configured with the non-linear `kernel="rbf"`.

Like the synthetic notebook, it exposes only a label-returning API to the
attacks and computes corrected adversarial-cluster density after an attack
succeeds. It writes results to `data\real_data_cluster_density\`.

For a deliberately distinct white-box reference, the decision tree is also
attacked with Aidan's ART `DecisionTreeAttack` using `offset=0.001`. That
attack receives the trained tree and follows its paths and thresholds; it is
labelled `white_box_tree_paths` in all output CSVs, visualized separately, and
excluded from the black-box conservative reliability score. Its adversarial
outputs are clipped to the same normalized `[0, 1]^10` domain before
measurement. It finds an adversarial tree path rather than optimizing the
globally smallest L2 perturbation, so it is a structural reference only and
must not be ranked directly against the query-bounded black-box attacks.

Run it with:

```powershell
uv sync
uv run jupyter nbconvert --to notebook --execute notebooks\real_data_cluster_density.ipynb --output real_data_cluster_density.executed.ipynb --ExecutePreprocessor.timeout=1800
```

The first run downloads CoverType to `data\sklearn_cache\`. Set
`QUICK_RUN = False` in the notebook to use a 45,000-example subset and 200
attack inputs; this is substantially slower, particularly for the RBF SVM.

### Representation-bias versus density trend

The same real-data notebook includes a controlled representation-bias sweep.
It retains 100%, 75%, 50%, or 25% of cover type 2 in training while retaining
all cover type 1 training examples. Evaluation data, attack inputs, feature
scaling, attack budgets, and clustering configuration remain fixed. The
resulting `bias_vs_density_trend.png` plots each model/attack pair's corrected
adversarial-cluster density and empirical robust accuracy against the
percentage of cover type 2 deliberately under-represented.

This is a label-representation stress test, **not** a demographic fairness
evaluation: forest-cover labels are not protected characteristics. It also
changes both class balance and total training examples. The plot should be
used to inspect trends and disagreement across attacks, not to claim that
representation bias causes cluster density or reliability. Detailed data are
saved as `bias_trend_summary.csv`, `bias_reliability_summary.csv`, and
`bias_attack_outcomes.csv` under `data\real_data_cluster_density\`.
The dotted decision-tree line is the white-box `DecisionTreeAttack` reference;
all other lines use only label-returning black-box access.
