"""Feature engineering and data-splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split

from malaysia_mortality.config import RANDOM_STATE, RISK_LEVEL_HIGH, RISK_LEVEL_LOW, TEST_SIZE


def create_risk_levels(mortality_count: int | float) -> str:
    """Bucket a raw mortality count into Low / Medium / High risk.

    Args:
        mortality_count: Absolute mortality count for a single record.

    Returns:
        One of ``"Low"``, ``"Medium"``, or ``"High"``.
    """
    if mortality_count < RISK_LEVEL_LOW:
        return "Low"
    if mortality_count < RISK_LEVEL_HIGH:
        return "Medium"
    return "High"


def encode_features_for_classification(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector for classification.

    One-hot encodes Year, Age Group, Disease_L2, and sex indicators.

    Args:
        df: The model-ready DataFrame (must contain ``Risk_Level``).

    Returns:
        ``(X, y)`` where *X* is the dummy-encoded feature matrix and *y* is
        the ``Risk_Level`` target.
    """
    feature_cols = ["Year", "Age Group", "Disease_L2", "Sex_Females", "Sex_Males"]
    X = pd.get_dummies(df[feature_cols])
    y = df["Risk_Level"]
    return X, y


def encode_features_for_regression(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector for regression.

    One-hot encodes ``Disease_L2``; all other columns (except the target) are
    kept as-is.  Non-numeric helper columns such as ``Risk_Level`` are dropped
    automatically.

    Args:
        df: The model-ready DataFrame.

    Returns:
        ``(X, y)`` where *y* is ``Mortality Count``.
    """
    y = df["Mortality Count"]
    X = df.drop("Mortality Count", axis=1)
    # Drop any string helper columns that may have been added downstream
    if "Risk_Level" in X.columns:
        X = X.drop("Risk_Level", axis=1)
    X = pd.get_dummies(X, columns=["Disease_L2"], prefix="Disease")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Wrapper around ``train_test_split`` with project defaults.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
