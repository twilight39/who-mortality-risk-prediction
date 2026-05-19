"""Tests for WHO CSV parsing and preprocessing."""

import pandas as pd
import pytest

from malaysia_mortality.preprocessing import (
    create_model_ready_df,
    extract_info_from_filename,
)


def test_extract_info_from_filename_valid():
    assert extract_info_from_filename("2020_0-4.csv") == ("2020", "0-4")
    assert extract_info_from_filename("2021_70+.csv") == ("2021", "70+")


def test_extract_info_from_filename_invalid():
    with pytest.raises(ValueError):
        extract_info_from_filename("bad_name.txt")


def test_create_model_ready_df():
    eda_df = pd.DataFrame(
        {
            "Year": ["2000", "2000", "2000"],
            "Age Group": ["0-4", "5-14", "0-4"],
            "Sex": ["Persons", "Persons", "Males"],
            "Mortality Count": [100, 200, 150],
            "Disease_L1": ["A", "A", "A"],
            "Disease_L2": ["B", "", "B"],
            "Disease_L3": ["", "", ""],
        }
    )
    model_df = create_model_ready_df(eda_df)
    # Only rows with L2 populated and L3 empty should remain
    assert len(model_df) == 2
    assert "Sex_Persons" in model_df.columns
    assert "Sex_Males" in model_df.columns
    assert "Sex" not in model_df.columns
    assert "Age Group" in model_df.columns
