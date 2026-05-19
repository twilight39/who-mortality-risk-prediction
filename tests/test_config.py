"""Smoke tests for configuration constants."""

from malaysia_mortality.config import (
    AGE_GROUP_MAPPING,
    COUNTRY,
    N_ESTIMATORS,
    RANDOM_STATE,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    TEST_SIZE,
    TIMELINE_MAP,
)


def test_country_code():
    assert COUNTRY == "MYS"


def test_random_state():
    assert RANDOM_STATE == 42


def test_test_size():
    assert 0 < TEST_SIZE < 1


def test_n_estimators():
    assert N_ESTIMATORS > 0


def test_age_group_mapping_order():
    assert AGE_GROUP_MAPPING["0-4"] == 0
    assert AGE_GROUP_MAPPING["70+"] == 6


def test_timeline_map():
    assert TIMELINE_MAP[2000] == 0
    assert TIMELINE_MAP[2020] == 1
    assert TIMELINE_MAP[2021] == 2


def test_risk_thresholds():
    assert RISK_LEVEL_LOW < RISK_LEVEL_HIGH
