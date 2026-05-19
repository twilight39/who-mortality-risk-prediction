"""Project configuration and constants."""

from pathlib import Path

# Path resolution — works regardless of where the package is imported from
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# WHO country code for Malaysia
COUNTRY: str = "MYS"

# Reproducibility
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.3
N_ESTIMATORS: int = 100

# Feature mappings
AGE_GROUP_MAPPING: dict[str, int] = {
    "0-4": 0,
    "5-14": 1,
    "15-29": 2,
    "30-49": 3,
    "50-59": 4,
    "60-69": 5,
    "70+": 6,
}

TIMELINE_MAP: dict[int, int] = {2000: 0, 2020: 1, 2021: 2}

# Risk-level thresholds for classification
RISK_LEVEL_LOW: int = 500
RISK_LEVEL_HIGH: int = 2000
