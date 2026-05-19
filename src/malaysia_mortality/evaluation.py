"""Model evaluation helpers and feature-importance reporting."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)


def evaluate_classification(model, X_test, y_test) -> dict[str, float | str]:
    """Evaluate a classification model.

    Args:
        model: Fitted scikit-learn classifier.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary with ``accuracy`` and ``report`` keys.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred),
    }


def evaluate_regression(model, X_test, y_test) -> dict[str, float]:
    """Evaluate a regression model.

    Args:
        model: Fitted scikit-learn regressor.
        X_test: Test features.
        y_test: True test targets.

    Returns:
        Dictionary with ``r2`` and ``mae`` keys.
    """
    y_pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }


def get_feature_importances(model, feature_names) -> pd.Series:
    """Extract and sort feature importances from a tree-based model.

    Args:
        model: Fitted estimator with a ``feature_importances_`` attribute.
        feature_names: List or Index of feature names.

    Returns:
        Sorted Series (descending) of importance scores.
    """
    return pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)
