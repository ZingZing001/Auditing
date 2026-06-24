from collections import deque

import numpy as np
import polars as pl
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import accuracy_score, f1_score

# from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def make_data(n_samples, noise, type="moons", SEED=42):
    if type == "moons":
        X, y = make_moons(
            n_samples=n_samples, noise=noise, random_state=SEED, shuffle=True
        )
        # X, y = make_blobs(n_samples=n_samples, random_state=SEED, n_features=2, centers=2)
    elif type == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=noise, random_state=SEED, shuffle=True
        )
    else:
        # jax is only needed for the checkerboard generator; import lazily so the
        # rest of util (tree/SVM auditing) loads without a working jax install.
        import jax
        import jax.numpy as jnp

        key = jax.random.PRNGKey(SEED)
        key1, key2 = jax.random.split(key)
        X = jax.random.uniform(
            key1, (n_samples, 2), minval=-10, maxval=10, dtype=jnp.float32
        )
        y = ((jnp.floor(X[:, 0]) + jnp.floor(X[:, 1])) % 2).astype(int)
        noise_idx = jax.random.choice(
            key2, a=n_samples, shape=(int(n_samples * noise),), replace=False
        )
        y = y.at[noise_idx].set(1 - y[noise_idx])
        X = np.array(X)
        y = np.array(y)
    return X, y


def labelencoding(X):
    for i in range(X.shape[1]):
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
    X = X.astype(int)
    return X


def cluster_density(points):
    """Compactness/DENSITY score for a single cluster of points.

    Returns ``n_pairs / (total_pairwise_separation + 1)`` over the true
    upper-triangular set of unique point pairs. Higher = tighter cluster.
    Returns 0.0 for clusters with fewer than two points. This is a density,
    not a distance, despite the historical ``adv distance`` column name.
    """
    points = np.asarray(points).reshape(len(points), -1)
    n = points.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            total += np.linalg.norm(points[a] - points[b]).item()
            count += 1
    return count / (total + 1)


def adv_cluster_stats(x_adv, min_samples: int = 5):
    """Cluster adversarial points with OPTICS and summarise compactness.

    Returns ``(n_clusters, densities)`` where ``n_clusters`` is the number of
    OPTICS clusters (noise label -1 excluded) and ``densities`` is the list of
    per-cluster :func:`cluster_density` scores. ``min_samples`` is exposed so the
    clustering can be stability-swept at the small fold sizes used here.
    """
    clustering = OPTICS(min_samples=min_samples).fit(x_adv)
    labels = np.unique(clustering.labels_[clustering.labels_ != -1])
    densities = [cluster_density(x_adv[clustering.labels_ == lab]) for lab in labels]
    return len(labels), densities


def audit_tree(
    X,
    y,
    SEED: int = 42,
    max_features: int = 2,
    stopping: int = 5,
    n_splits: int = 5,
    top_k: int = 5,
):
    acc_queue = deque(maxlen=stopping)
    acc_queue.extend([0, 0.1])
    d = 1
    results = pl.DataFrame()
    # for d in max_depths:

    while np.std(acc_queue).item() >= 0.001:
        model = DecisionTreeClassifier(
            random_state=SEED,
            max_features=max_features,
            max_depth=d,
            min_samples_split=2,
            min_samples_leaf=1,
        )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        train_acces = []
        train_f1s = []
        test_acces = []
        test_f1s = []
        avg_acces = []
        avg_f1s = []
        avg_distances = []

        for train_idx, test_idx in skf.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            model.fit(X_train, y_train)

            classifier = ScikitlearnDecisionTreeClassifier(model)

            y_pred, test_acc, test_f1 = eval_model(
                X_test, y_test.reshape(y_test.shape[0], 1), classifier
            )
            y_train_pred, train_acc, train_f1 = eval_model(
                X_train, y_train.reshape(y_train.shape[0], 1), classifier
            )

            attack = DecisionTreeAttack(classifier=classifier)
            try:
                x_attack_adv = attack.generate(x=X_test)
            except:
                break

            y_adv_pred, acc_adv, f1_adv = eval_model(
                x_attack_adv, y_test.reshape(y_test.shape[0], 1), classifier
            )
            distances = []
            for idx, attack_data in enumerate(x_attack_adv):
                true_label = y_test[idx]
                attack_class_points = X_test[y_test.T.squeeze(0) != true_label]
                distance_vector = np.sqrt(
                    np.sum((attack_data - attack_class_points) ** 2, axis=1)
                )
                indices = np.argsort(distance_vector)[:top_k]
                knn_distance = distance_vector[indices]
                distances.append(np.mean(knn_distance).item())

            adv_distance = np.mean(distances)
            train_acces.append(train_acc)
            train_f1s.append(train_f1)
            test_acces.append(test_acc)
            test_f1s.append(test_f1)
            avg_acces.append(acc_adv)
            avg_f1s.append(f1_adv)
            avg_distances.append(adv_distance.tolist())

        acc_queue.append(np.mean(test_acces).item())

        res = {
            "train acc": train_acces,
            "train f1": train_f1s,
            "test acc": test_acces,
            "test f1": test_f1s,
            "adv acc": avg_acces,
            "adv f1": avg_f1s,
            "adv distance": avg_distances,
            "depth": d,
        }
        results = pl.concat([results, pl.from_dict(res)])
        d += 1

    return results


def audit_tree_bias(
    X,
    y,
    bias,
    depth: int = 10,
    SEED: int = 42,
    max_features: int = 2,
    stopping: int = 5,
    n_splits: int = 5,
    top_k: int = 5,
    dbscan: bool = False,
    target_bias: int = 0,
    over: bool = False,
    min_samples: int = 5,
):
    # acc_queue = deque(maxlen=stopping)
    # acc_queue.extend([0,.1])
    # d = 1
    results = pl.DataFrame()
    # for d in max_depths:

    # while np.std(acc_queue).item() >=.001:
    model = DecisionTreeClassifier(
        random_state=SEED,
        max_features=max_features,
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_acces = []
    train_f1s = []
    test_acces = []
    test_f1s = []
    avg_acces = []
    avg_f1s = []
    avg_distances = []
    clusters = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        X_sort = np.sort(X_train[:, stopping][y_train.squeeze(1) == target_bias])
        bias_idx = int(X_sort.shape[0] * bias.item())

        mask = (X_train[:, stopping] > X_sort[bias_idx])[
            y_train.squeeze(1) == target_bias
        ]
        X_train_b = X_train[y_train.squeeze(1) == target_bias][mask]
        y_train_b = y_train[y_train.squeeze(1) == target_bias][mask]

        for i in np.unique(y_train):
            if i != target_bias:
                X_train_b = np.vstack((X_train_b, X_train[y_train.squeeze(1) == i]))
                y_train_b = np.vstack((y_train_b, y_train[y_train.squeeze(1) == i]))

        if over:
            if mask.sum() <= 5:
                # seeded RNG so the oversampling top-up is reproducible
                rng = np.random.default_rng(SEED)
                extra = rng.choice(
                    X_train[y_train.squeeze(1) == target_bias].shape[0],
                    replace=False,
                    size=5,
                )
                X_train_b = np.vstack(
                    (X_train_b, X_train[y_train.squeeze(1) == target_bias][extra])
                )
                y_train_b = np.vstack(
                    (y_train_b, y_train[y_train.squeeze(1) == target_bias][extra])
                )
            X_train_b, y_train_b = SMOTE(random_state=SEED, k_neighbors=4).fit_resample(
                X_train_b, y_train_b
            )
            X_train_b = np.vstack(
                (X_train_b, X_train[y_train.squeeze(1) == target_bias][~mask])
            )
            y_train_b = np.vstack(
                (
                    y_train_b.reshape(y_train_b.shape[0], -1),
                    y_train[y_train.squeeze(1) == target_bias][~mask],
                )
            )

        model.fit(X_train_b, y_train_b)

        classifier = ScikitlearnDecisionTreeClassifier(model)

        y_pred, test_acc, test_f1 = eval_model(
            X_test, y_test.reshape(y_test.shape[0], 1), classifier
        )
        y_train_pred, train_acc, train_f1 = eval_model(
            X_train, y_train.reshape(y_train.shape[0], 1), classifier
        )

        attack = DecisionTreeAttack(classifier=classifier)
        try:
            x_attack_adv = attack.generate(x=X_test)
        except:
            break

        y_adv_pred, acc_adv, f1_adv = eval_model(
            x_attack_adv, y_test.reshape(y_test.shape[0], 1), classifier
        )
        distances = []

        if dbscan:
            # OPTICS-cluster the adversarial points; `distances` holds the
            # per-cluster compactness/DENSITY scores (see adv_cluster_stats /
            # cluster_density). `min_samples` is passed through so the clustering
            # can be stability-swept at the small fold sizes used here.
            n_clusters, distances = adv_cluster_stats(
                x_attack_adv, min_samples=min_samples
            )
            clusters.append(n_clusters)
        else:
            for idx, attack_data in enumerate(x_attack_adv):
                true_label = y_test[idx]
                attack_class_points = X_test[y_test != true_label]
                distance_vector = np.sqrt(
                    np.sum((attack_data - attack_class_points) ** 2, axis=1)
                )
                indices = np.argsort(distance_vector)[:top_k]
                knn_distance = distance_vector[indices]
                distances.append(np.mean(knn_distance).item())

        adv_distance = np.mean(distances)
        train_acces.append(train_acc)
        train_f1s.append(train_f1)
        test_acces.append(test_acc)
        test_f1s.append(test_f1)
        avg_acces.append(acc_adv)
        avg_f1s.append(f1_adv)
        avg_distances.append(adv_distance.tolist())

        # acc_queue.append(np.mean(test_acces).item())

    res = {
        "train acc": train_acces,
        "train f1": train_f1s,
        "test acc": test_acces,
        "test f1": test_f1s,
        "adv acc": avg_acces,
        "adv f1": avg_f1s,
        "adv distance": avg_distances,
        "clusters": clusters,
        #    "depth" : d,
        "bias": bias.item(),
    }
    results = pl.concat([results, pl.from_dict(res)])
    # d += 1

    return results


# def audit_svc(
#         X,
#         y,
#         SEED : int =42,
#         C: float = .3,
#         stopping : int = 5,
#         n_splits : int = 5,
#         top_k : int = 5
#         ):
#     acc_queue = deque(maxlen=stopping)
#     acc_queue.extend([0,.1])
#     d = 1
#     results = pl.DataFrame()
#     # for d in max_depths:

#     while np.std(acc_queue).item() >=.001:
#         model = SVC(
#             random_state=SEED, C=C, kernel="rbf")
#         skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
#         train_acces = []
#         train_f1s = []
#         test_acces = []
#         test_f1s = []
#         avg_acces = []
#         avg_f1s = []
#         avg_distances = []


#         for train_idx, test_idx in skf.split(X, y):
#             X_train = X[train_idx]
#             y_train = y[train_idx]
#             X_test = X[test_idx]
#             y_test = y[test_idx]
#             model.fit(X_train,y_train)

#             classifier = ScikitlearnSVC(model)

#             y_pred, test_acc, test_f1 = eval_model(X_test, y_test.reshape(y_test.shape[0],1), classifier)
#             y_train_pred, train_acc, train_f1 = eval_model(X_train, y_train.reshape(y_train.shape[0],1), classifier)


#             attack = FastGradientMethod(estimator=classifier, eps=.07)
#             try:
#                 x_attack_adv = attack.generate(x=X_test)
#             except:
#                 break


#             y_adv_pred, acc_adv, f1_adv = eval_model(x_attack_adv, y_test.reshape(y_test.shape[0],1), classifier)
#             distances = []
#             for idx, attack_data in enumerate(x_attack_adv):
#                 true_label = y_test[idx]
#                 attack_class_points = X_test[y_test != true_label]
#                 distance_vector = np.sqrt(np.sum((attack_data - attack_class_points)**2, axis=1))
#                 indices = np.argsort(distance_vector)[:top_k]
#                 knn_distance = distance_vector[indices]
#                 distances.append(np.mean(knn_distance).item())


#             adv_distance = np.mean(distances)
#             train_acces.append(train_acc)
#             train_f1s.append(train_f1)
#             test_acces.append(test_acc)
#             test_f1s.append(test_f1)
#             avg_acces.append(acc_adv)
#             avg_f1s.append(f1_adv)
#             avg_distances.append(adv_distance.tolist())

#         acc_queue.append(np.mean(test_acces).item())

#         res = {"train acc": train_acces,
#                "train f1" : train_f1s,
#                "test acc": test_acces,
#                "test f1" : test_f1s,
#                "adv acc": avg_acces,
#                "adv f1": avg_f1s,
#                "adv distance" : avg_distances,
#                "depth" : d
#                }
#         results = pl.concat([results, pl.from_dict(res)])
#         d += 1

#     return results


def eval_model(X, y, classifier):
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, np.argmax(y_pred, axis=1))
    f1 = f1_score(y, np.argmax(y_pred, axis=1), average="macro")
    return y_pred, acc, f1
