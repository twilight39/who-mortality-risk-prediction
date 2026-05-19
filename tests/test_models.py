"""Smoke tests for model factories."""

import numpy as np
import pandas as pd

from malaysia_mortality.models import (
    run_hierarchical_clustering,
    run_kmeans_elbow,
    train_decision_tree,
    train_kmeans,
    train_random_forest_classifier,
    train_random_forest_regressor,
)


def test_train_decision_tree():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series(["Low", "Low", "High", "High"])
    model = train_decision_tree(X, y)
    assert hasattr(model, "predict")
    assert model.score(X, y) == 1.0


def test_train_random_forest_classifier():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series(["Low", "Low", "High", "High"])
    model = train_random_forest_classifier(X, y)
    assert hasattr(model, "predict")


def test_train_random_forest_regressor():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40])
    model = train_random_forest_regressor(X, y)
    assert hasattr(model, "predict")


def test_run_kmeans_elbow():
    data = np.random.default_rng(42).random((20, 3))
    inertias = run_kmeans_elbow(data, k_range=range(1, 6))
    assert len(inertias) == 5
    # Inertia should decrease as k increases
    assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))


def test_train_kmeans():
    data = np.random.default_rng(42).random((20, 3))
    model, labels = train_kmeans(data, n_clusters=3)
    assert len(labels) == 20
    assert set(labels).issubset({0, 1, 2})


def test_run_hierarchical_clustering():
    data = np.random.default_rng(42).random((10, 3))
    linked = run_hierarchical_clustering(data)
    # linkage matrix for n samples has shape (n-1, 4)
    assert linked.shape == (9, 4)
