"""Tests for evaluation utilities."""

import pandas as pd

from malaysia_mortality.evaluation import (
    evaluate_classification,
    evaluate_regression,
    get_feature_importances,
)
from malaysia_mortality.models import (
    train_random_forest_classifier,
    train_random_forest_regressor,
)


def test_evaluate_classification():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series(["Low", "Low", "High", "High"])
    model = train_random_forest_classifier(X, y)
    result = evaluate_classification(model, X, y)
    assert "accuracy" in result
    assert "report" in result
    assert 0 <= result["accuracy"] <= 1


def test_evaluate_regression():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40])
    model = train_random_forest_regressor(X, y)
    result = evaluate_regression(model, X, y)
    assert "r2" in result
    assert "mae" in result
    assert result["r2"] > 0.9


def test_get_feature_importances():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40])
    model = train_random_forest_regressor(X, y)
    importances = get_feature_importances(model, X.columns)
    assert len(importances) == 2
    assert importances.index[0] in {"a", "b"}
