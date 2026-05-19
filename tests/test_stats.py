"""Tests for statistical testing utilities."""

import pandas as pd

from malaysia_mortality.stats import chi_squared_analysis, correlation_analysis


def test_correlation_analysis():
    # Perfect positive correlation
    data = {
        "Disease A": [10, 20, 30],
        "Disease B": [5, 5, 5],
    }
    pivot = pd.DataFrame(data, index=[2000, 2020, 2021])
    timeline_map = {2000: 0, 2020: 1, 2021: 2}
    result = correlation_analysis(pivot, timeline_map=timeline_map, threshold=0.5)
    assert "Disease A" in result["Disease"].values
    assert "Disease B" not in result["Disease"].values
    disease_a_row = result[result["Disease"] == "Disease A"].iloc[0]
    assert disease_a_row["Correlation (r)"] > 0.99


def test_chi_squared_analysis():
    table = pd.DataFrame(
        {
            2000: [100, 200],
            2020: [150, 250],
            2021: [300, 400],
        },
        index=["Disease A", "Disease B"],
    )
    result = chi_squared_analysis(table)
    assert "chi2" in result
    assert "p_value" in result
    assert "dof" in result
    assert "cramers_v" in result
    assert 0 <= result["cramers_v"] <= 1
