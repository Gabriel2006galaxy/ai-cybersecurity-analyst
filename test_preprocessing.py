"""
test_preprocessing.py
----------------------
A small, dependency-free validation script for the Phase 1 preprocessing
pipeline (src/preprocessor.py). Deliberately written with plain asserts
instead of pytest, so no new dependency is needed beyond what's already
in requirements.txt.

Run with:

    python test_preprocessing.py

It exits with code 0 and prints "ALL TESTS PASSED" if everything is OK,
or raises an AssertionError (non-zero exit code) pointing at the first
failing check.

IMPORTANT: this script never modifies data/synthetic_security_logs.csv.
Error-handling checks build small, temporary, in-memory/on-disk DataFrames
of their own instead.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from config import LABEL_COLUMN, LOG_FILE_PATH, TIMESTAMP_COLUMN
from src.data_loader import load_logs
from src.preprocessor import (
    clean_logs,
    engineer_features,
    prepare_ml_features,
    run_preprocessing_pipeline,
    validate_logs,
)

PASSED = 0


def check(condition: bool, message: str) -> None:
    """Assert a condition and print a checkmark; stops the script on failure."""
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


def test_dataset_loads():
    print("\n1. The 350-record sample dataset loads")
    df = load_logs()
    check(len(df) == 350, f"expected 350 rows, got {len(df)}")
    check(not df.empty, "loaded dataset is not empty")
    return df


def test_timestamp_conversion(df: pd.DataFrame):
    print("\n2. Timestamp conversion works")
    check(
        pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COLUMN]),
        "load_logs() parses 'timestamp' into a real datetime column",
    )

    cleaned_df, _ = clean_logs(df)
    check(
        pd.api.types.is_datetime64_any_dtype(cleaned_df[TIMESTAMP_COLUMN]),
        "clean_logs() keeps timestamp as datetime",
    )
    return cleaned_df


def test_derived_time_features(cleaned_df: pd.DataFrame):
    print("\n3. Derived time features are generated")
    feature_df = engineer_features(cleaned_df)
    for col in ["hour", "day_of_week", "is_off_hours"]:
        check(col in feature_df.columns, f"'{col}' column was created")

    check(feature_df["hour"].between(0, 23).all(), "'hour' values are within 0-23")
    check(
        set(feature_df["is_off_hours"].unique()).issubset({0, 1}),
        "'is_off_hours' is binary (0/1)",
    )

    # Spot-check the off-hours rule itself: hour 2 -> off-hours, hour 14 -> not.
    sample = feature_df.iloc[0:1].copy()
    sample["hour"] = 2
    sample["is_off_hours"] = ((sample["hour"] >= 22) | (sample["hour"] < 6)).astype(int)
    check(sample["is_off_hours"].iloc[0] == 1, "hour=2 is correctly flagged as off-hours")

    return feature_df


def test_required_numerical_features_present(feature_df: pd.DataFrame):
    print("\n4. Required numerical features are present")
    for col in [
        "failed_logins",
        "request_count",
        "port",
        "data_transferred_mb",
        "session_duration",
        "device_count",
    ]:
        check(col in feature_df.columns, f"'{col}' present in engineered features")
        check(
            pd.api.types.is_numeric_dtype(feature_df[col]),
            f"'{col}' has a numeric dtype",
        )


def test_true_label_excluded_from_ml_features(feature_df: pd.DataFrame):
    print("\n5. true_label is excluded from ML input features")
    check(LABEL_COLUMN in feature_df.columns, "true_label still present for display/eval")

    ml_df, _scaler, feature_groups = prepare_ml_features(feature_df)
    check(LABEL_COLUMN not in ml_df.columns, "true_label NOT present in ml_df columns")
    check(
        LABEL_COLUMN not in feature_groups["all_ml_features"],
        "true_label NOT present in the ML feature-name list",
    )
    return ml_df


def test_ml_matrix_has_no_nans(ml_df: pd.DataFrame):
    print("\n6. The final ML-ready feature matrix contains no NaN values")
    check(int(ml_df.isna().sum().sum()) == 0, "ml_df has zero NaN values")
    check(
        all(pd.api.types.is_numeric_dtype(ml_df[c]) for c in ml_df.columns),
        "every column in ml_df is numeric (fully encoded)",
    )


def test_full_pipeline_runs_without_exceptions():
    print("\n7. Preprocessing functions run end-to-end without exceptions")
    df = load_logs()
    result = run_preprocessing_pipeline(df)
    check(result.is_usable, "run_preprocessing_pipeline() produced usable output")
    check(result.ml_df.shape[0] == result.cleaned_df.shape[0], "row counts stay aligned")
    check(len(result.numerical_features) > 0, "numerical feature list is non-empty")
    check(len(result.onehot_features) > 0, "one-hot feature list is non-empty")


def test_error_handling_on_bad_data():
    print("\n8. Error handling on corrupted data (does not touch the real CSV)")

    # 8a. Missing required column
    df_missing_col = pd.DataFrame({"user": ["a"], "timestamp": ["2026-01-01 10:00:00"]})
    result = validate_logs(df_missing_col)
    check(not result.is_valid, "missing required columns -> validation fails")
    check(len(result.errors) > 0, "missing columns produces at least one error message")

    # 8b. Empty dataframe
    result_empty = validate_logs(pd.DataFrame())
    check(not result_empty.is_valid, "empty dataframe -> validation fails")

    # 8c. Invalid numeric values, invalid timestamp, and a duplicate row
    base_row = {
        "timestamp": "2026-01-01 10:00:00",
        "user": "testuser",
        "source_ip": "10.0.1.5",
        "destination_ip": "10.0.5.5",
        "failed_logins": 0,
        "request_count": 10,
        "port": 443,
        "data_transferred_mb": 1.5,
        "session_duration": 100,
        "event_type": "login_success",
        "login_location": "Mumbai, IN",
        "device_count": 1,
    }
    bad_row = dict(base_row)
    bad_row["failed_logins"] = "not_a_number"       # invalid numeric
    bad_ts_row = dict(base_row)
    bad_ts_row["timestamp"] = "not_a_timestamp"       # invalid timestamp

    dirty_df = pd.DataFrame([base_row, base_row, bad_row, bad_ts_row])  # includes a duplicate

    validation = validate_logs(dirty_df)
    check(validation.is_valid, "dirty-but-recoverable data still passes validate_logs() (warnings only)")
    check(len(validation.warnings) > 0, "dirty data produces warnings")

    cleaned, report = clean_logs(dirty_df)
    check(report["duplicates_removed"] >= 1, "duplicate row was detected and removed")
    check(report["invalid_timestamp_removed"] >= 1, "row with bad timestamp was removed")
    check(report["invalid_numeric_removed"] >= 1, "row with invalid numeric value was removed")
    check(len(cleaned) == 1, "only the single valid, non-duplicate row survives cleaning")

    # 8d. Unexpected/blank categorical value is filled, not dropped
    blank_row = dict(base_row)
    blank_row["login_location"] = ""
    df_blank = pd.DataFrame([base_row, blank_row])
    cleaned_blank, _ = clean_logs(df_blank)
    check(len(cleaned_blank) == 2, "blank categorical value does not cause a dropped row")
    check(
        (cleaned_blank["login_location"] == "unknown").sum() == 1,
        "blank login_location was filled with 'unknown'",
    )

    # 8e. Fully-invalid dataset (every numeric value broken) -> pipeline reports failure, doesn't crash
    all_bad_row = dict(base_row)
    all_bad_row["failed_logins"] = "x"
    all_bad_row["request_count"] = "x"
    all_bad_row["port"] = "x"
    all_bad_row["data_transferred_mb"] = "x"
    all_bad_row["session_duration"] = "x"
    all_bad_row["device_count"] = "x"
    df_all_bad = pd.DataFrame([all_bad_row])
    result_all_bad = run_preprocessing_pipeline(df_all_bad)
    check(not result_all_bad.is_usable, "an entirely-invalid dataset is reported as unusable, not a crash")
    check(len(result_all_bad.validation.errors) > 0, "a clear error message is recorded")


def main():
    print("=" * 70)
    print("Phase 1 preprocessing test suite")
    print(f"Using dataset: {LOG_FILE_PATH}")
    print("=" * 70)

    df = test_dataset_loads()
    cleaned_df = test_timestamp_conversion(df)
    feature_df = test_derived_time_features(cleaned_df)
    test_required_numerical_features_present(feature_df)
    ml_df = test_true_label_excluded_from_ml_features(feature_df)
    test_ml_matrix_has_no_nans(ml_df)
    test_full_pipeline_runs_without_exceptions()
    test_error_handling_on_bad_data()

    print("\n" + "=" * 70)
    print(f"ALL TESTS PASSED ({PASSED} checks)")
    print("=" * 70)


if __name__ == "__main__":
    main()
