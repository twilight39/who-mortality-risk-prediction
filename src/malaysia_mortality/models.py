"""Model training factories with project-level defaults."""

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from malaysia_mortality.config import N_ESTIMATORS, RANDOM_STATE


def train_decision_tree(X_train, y_train, random_state: int = RANDOM_STATE):
    """Fit a Decision Tree Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Fitted ``DecisionTreeClassifier``.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest_classifier(
    X_train,
    y_train,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
):
    """Fit a Random Forest Classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 uses all cores).

    Returns:
        Fitted ``RandomForestClassifier``.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_regressor(
    X_train,
    y_train,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
):
    """Fit a Random Forest Regressor.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of trees in the forest.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs (-1 uses all cores).

    Returns:
        Fitted ``RandomForestRegressor``.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def run_kmeans_elbow(
    scaled_data,
    k_range: range = range(1, 11),
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
) -> list[float]:
    """Compute K-Means inertia across a range of *k* values (Elbow Method).

    Args:
        scaled_data: Standardised feature matrix.
        k_range: Range of cluster counts to evaluate.
        random_state: Seed for reproducibility.
        n_init: Number of initialisations for each *k*.

    Returns:
        List of inertia values in the same order as *k_range*.
    """
    inertias: list[float] = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        kmeans.fit(scaled_data)
        inertias.append(float(kmeans.inertia_))
    return inertias


def train_kmeans(
    scaled_data,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
):
    """Fit a K-Means model and return cluster assignments.

    Args:
        scaled_data: Standardised feature matrix.
        n_clusters: Desired number of clusters.
        random_state: Seed for reproducibility.
        n_init: Number of initialisations.

    Returns:
        Tuple of ``(kmeans_model, cluster_labels)``.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(scaled_data)
    return kmeans, labels


def run_hierarchical_clustering(scaled_data, method: str = "ward"):
    """Compute hierarchical linkage matrix.

    Args:
        scaled_data: Standardised feature matrix.
        method: Linkage algorithm (default ``"ward"``).

    Returns:
        Linkage matrix suitable for ``scipy.cluster.hierarchy.dendrogram``.
    """
    from scipy.cluster.hierarchy import linkage

    return linkage(scaled_data, method=method)
