"""Statistical testing utilities."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr

from malaysia_mortality.config import TIMELINE_MAP


def correlation_analysis(
    pivot_df: pd.DataFrame,
    timeline_map: dict[int, int] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute Pearson correlation between mortality trends and timeline.

    Args:
        pivot_df: DataFrame with Years as index and diseases as columns.
        timeline_map: Mapping from year to ordinal value.
        threshold: Minimum absolute correlation to include in results.

    Returns:
        DataFrame of diseases with ``|r| > threshold``, sorted descending.
    """
    if timeline_map is None:
        timeline_map = TIMELINE_MAP

    timeline = pivot_df.index.map(timeline_map)
    results: list[dict[str, float]] = []

    for disease in pivot_df.columns:
        if pivot_df[disease].std() > 0:
            corr, p_val = pearsonr(timeline, pivot_df[disease])
            results.append(
                {
                    "Disease": disease,
                    "Correlation (r)": float(corr),
                    "P-value": float(p_val),
                    "R-squared": float(corr**2),
                }
            )

    corr_df = pd.DataFrame(results)
    strong = corr_df[abs(corr_df["Correlation (r)"]) > threshold].sort_values(
        by="Correlation (r)", ascending=False
    )
    return strong.reset_index(drop=True)


def chi_squared_analysis(contingency_table: pd.DataFrame) -> dict[str, float]:
    """Run Chi-squared test of independence and compute Cramér's V.

    Args:
        contingency_table: Table of observed frequencies.

    Returns:
        Dictionary with ``chi2``, ``p_value``, ``dof``, and ``cramers_v``.
    """
    chi2, p_value, dof, _ = chi2_contingency(contingency_table)
    n = float(contingency_table.sum().sum())
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
    }
