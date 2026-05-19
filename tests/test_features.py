"""Tests for feature engineering utilities."""

import pandas as pd

from malaysia_mortality.features import (
    create_risk_levels,
    encode_features_for_classification,
    encode_features_for_regression,
    split_data,
)


def test_create_risk_levels_low():
    assert create_risk_levels(100) == "Low"
    assert create_risk_levels(499) == "Low"


def test_create_risk_levels_medium():
    assert create_risk_levels(500) == "Medium"
    assert create_risk_levels(1999) == "Medium"


def test_create_risk_levels_high():
    assert create_risk_levels(2000) == "High"
    assert create_risk_levels(5000) == "High"


def test_encode_features_for_classification():
    df = pd.DataFrame(
        {
            "Year": [2000, 2020],
            "Age Group": [0, 6],
            "Disease_L2": ["A", "B"],
            "Sex_Females": [0, 1],
            "Sex_Males": [1, 0],
            "Risk_Level": ["Low", "High"],
        }
    )
    X, y = encode_features_for_classification(df)
    assert "Disease_L2_A" in X.columns
    assert "Disease_L2_B" in X.columns
    assert list(y) == ["Low", "High"]


def test_encode_features_for_regression():
    df = pd.DataFrame(
        {
            "Year": [2000, 2020],
            "Age Group": [0, 6],
            "Disease_L2": ["A", "B"],
            "Mortality Count": [100, 5000],
        }
    )
    X, y = encode_features_for_regression(df)
    assert "Disease_A" in X.columns
    assert "Disease_B" in X.columns
    assert "Mortality Count" not in X.columns
    assert list(y) == [100, 5000]


def test_split_data_shape():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0] * 50 + [1] * 50)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 30
    assert len(X_train) == 70
    assert len(y_test) == 30
    assert len(y_train) == 70
