"""WHO CSV parsing and preprocessing logic."""

import re
from pathlib import Path

import pandas as pd

from malaysia_mortality.config import AGE_GROUP_MAPPING, COUNTRY, PROCESSED_DATA_DIR


def extract_info_from_filename(filename: str) -> tuple[str, str]:
    """Extract year and age group from a WHO CSV filename.

    Args:
        filename: Expected format ``YYYY_age-group.csv`` (e.g. ``2020_0-4.csv``).

    Returns:
        Tuple of (year, age_group).

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    pattern = r"(\d{4})_(.+)\.csv"
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Filename {filename} does not match expected pattern.")


def process_csv_file(file_path: str | Path, country: str = COUNTRY) -> pd.DataFrame:
    """Parse a single WHO mortality CSV and extract Malaysian data.

    The WHO files contain 7 header rows of metadata.  This function skips those,
    locates the column for the requested country, and walks the 4-level disease
    hierarchy (L1→L4) to build a flat record per row.

    Mortality counts are stored per-1000-population in the raw files; they are
    multiplied by 1000 to obtain absolute counts.

    Args:
        file_path: Path to a single WHO CSV.
        country: ISO-3 country code to extract (default ``MYS``).

    Returns:
        DataFrame with columns: Year, Age Group, Disease_L1..L4, Sex, Mortality Count.
    """
    file_path = Path(file_path)
    year, age_group = extract_info_from_filename(file_path.name)

    df = pd.read_csv(file_path, skiprows=7, low_memory=False)

    malaysia_col_idx = next(
        (idx for idx, col_name in enumerate(df.columns) if col_name == country),
        -1,
    )
    if malaysia_col_idx == -1:
        raise ValueError(f"Country '{country}' not found in columns of {file_path}")

    result_data: list[dict[str, str | int]] = []
    disease_l1 = disease_l2 = disease_l3 = disease_l4 = ""

    for _, row in df.iterrows():
        sex = str(row.iloc[0])

        # Walk the disease hierarchy
        if pd.notna(row.iloc[3]) and len(str(row.iloc[3])) > 3:
            disease_l1 = str(row.iloc[3])
            disease_l2 = disease_l3 = disease_l4 = ""
        elif pd.notna(row.iloc[4]) and len(str(row.iloc[4])) > 3:
            disease_l2 = str(row.iloc[4])
            disease_l3 = disease_l4 = ""
        elif pd.notna(row.iloc[5]) and len(str(row.iloc[5])) > 3:
            disease_l3 = str(row.iloc[5])
            disease_l4 = ""
        elif pd.notna(row.iloc[6]) and len(str(row.iloc[6])) > 3:
            disease_l4 = str(row.iloc[6])

        if disease_l1.startswith(" Population") or disease_l1 == "":
            continue

        try:
            raw_value = row.iloc[malaysia_col_idx]
            mortality_count = int(float(raw_value if pd.notna(raw_value) else "0") * 1000)
        except (ValueError, TypeError):
            mortality_count = 0

        result_data.append(
            {
                "Year": year,
                "Age Group": age_group,
                "Disease_L1": disease_l1,
                "Disease_L2": disease_l2,
                "Disease_L3": disease_l3,
                "Disease_L4": disease_l4,
                "Sex": sex,
                "Mortality Count": mortality_count,
            }
        )

    return pd.DataFrame(result_data)


def build_mortality_dataframe(
    raw_dir: Path | str, country: str = COUNTRY
) -> pd.DataFrame:
    """Aggregate all WHO CSVs in *raw_dir* into a single EDA DataFrame.

    Args:
        raw_dir: Directory containing ``YYYY_age-group.csv`` files.
        country: ISO-3 country code.

    Returns:
        Concatenated DataFrame from all parsed files.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {raw_dir}")

    frames = [process_csv_file(f, country=country) for f in files]
    return pd.concat(frames, ignore_index=True)


def create_model_ready_df(
    eda_df: pd.DataFrame,
    target_level: str = "Disease_L2",
    next_level: str = "Disease_L3",
) -> pd.DataFrame:
    """Derive the model-ready dataset from the EDA DataFrame.

    Keeps only rows where *target_level* is populated but *next_level* is empty,
    effectively isolating L2 disease categories.  Age group is ordinally encoded
    and sex is one-hot encoded.

    Args:
        eda_df: The comprehensive EDA DataFrame.
        target_level: Column to use as the classification target.
        next_level: Column that must be empty to confirm a leaf-L2 row.

    Returns:
        DataFrame with encoded features and the target column.
    """
    features = ["Year", "Age Group", "Sex", "Mortality Count"]
    model_df = eda_df[
        (eda_df[target_level].str.strip() != "")
        & (eda_df[next_level].str.strip() == "")
    ].copy()

    features_df = model_df[features].copy()
    features_df["Age Group"] = features_df["Age Group"].map(AGE_GROUP_MAPPING)

    sex_dummies = pd.get_dummies(features_df["Sex"], prefix="Sex")
    features_df = pd.concat([features_df, sex_dummies], axis=1).drop("Sex", axis=1)

    target_series = model_df[target_level].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)

    return pd.concat([features_df, target_series], axis=1)


def save_processed_data(
    eda_df: pd.DataFrame,
    model_df: pd.DataFrame,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
) -> None:
    """Persist EDA and model-ready DataFrames to CSV.

    Args:
        eda_df: Comprehensive EDA DataFrame.
        model_df: Model-ready DataFrame.
        processed_dir: Destination directory.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    eda_df.to_csv(processed_dir / "malaysia_mortality_data_eda.csv", index=False)
    model_df.to_csv(processed_dir / "malaysia_mortality_data_model.csv", index=False)
