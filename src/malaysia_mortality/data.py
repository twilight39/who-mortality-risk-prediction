"""Data-loading helpers for processed artefacts."""

from pathlib import Path

import pandas as pd

from malaysia_mortality.config import PROCESSED_DATA_DIR


def load_eda_data(processed_dir: Path | str = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load the comprehensive EDA-ready dataset.

    Args:
        processed_dir: Directory containing processed CSV files.

    Returns:
        DataFrame with all years, age groups, sexes, and disease levels.
    """
    path = Path(processed_dir) / "malaysia_mortality_data_eda.csv"
    return pd.read_csv(path)


def load_model_data(processed_dir: Path | str = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load the model-ready dataset (L2 disease categories only).

    Args:
        processed_dir: Directory containing processed CSV files.

    Returns:
        DataFrame prepared for classification and regression tasks.
    """
    path = Path(processed_dir) / "malaysia_mortality_data_model.csv"
    return pd.read_csv(path)
