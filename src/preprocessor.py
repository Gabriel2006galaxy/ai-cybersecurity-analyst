"""
preprocessor.py
----------------
Phase 1 - Data preprocessing & feature engineering.

Pipeline implemented in this module:

    Raw Logs -> Validation -> Cleaning -> Feature Engineering
             -> Encoding / Scaling -> ML-Ready Features

Every stage is a small, independent function so later phases (Phase 2
anomaly detection, etc.) can import just the piece they need instead of
duplicating logic. The Streamlit app calls `run_preprocessing_pipeline()`,
which chains all of the stages together.

IMPORTANT: `true_label` (config.LABEL_COLUMN) is ground truth added to the
synthetic dataset only so later phases can be evaluated. It is never read
as an input feature anywhere in this module - `prepare_ml_features()`
builds the ML matrix from an explicit allow-list of engineered columns
that simply does not include it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    IDENTIFIER_COLUMNS,
    LABEL_COLUMN,
    LOW_CARDINALITY_CATEGORICAL_COLUMNS,
    OFF_HOURS_END_HOUR,
    OFF_HOURS_START_HOUR,
    REQUIRED_CATEGORICAL_COLUMNS,
    REQUIRED_NUMERICAL_COLUMNS,
    TIMESTAMP_COLUMN,
)


# =======================================================================
# Result containers
# =======================================================================
@dataclass
class ValidationResult:
    """Outcome of validate_logs(). Never raises - always inspectable."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid and not self.warnings:
            return "Dataset passed validation with no issues."
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts) if parts else "OK"


@dataclass
class PreprocessingResult:
    """
    Bundles the output of every pipeline stage so the Streamlit app and
    tests can inspect (or display) any of them without re-running work.

    Distinguishes the three data "levels" required by Phase 1:
        A. raw_df / cleaned_df -> display/original data
        B. feature_df          -> derived features (human-readable)
        C. ml_df                -> ML-ready features (encoded + scaled)
    """

    raw_df: pd.DataFrame
    validation: ValidationResult
    cleaned_df: pd.DataFrame
    cleaning_report: Dict[str, int]
    feature_df: pd.DataFrame
    ml_df: pd.DataFrame
    numerical_features: List[str]
    derived_features: List[str]
    onehot_features: List[str]
    excluded_columns: List[str]
    scaler: Optional[StandardScaler]

    @property
    def is_usable(self) -> bool:
        """True if the pipeline produced a non-empty ML-ready matrix."""
        return self.validation.is_valid and not self.ml_df.empty


# =======================================================================
# 1. Validation
# =======================================================================
def validate_logs(df: pd.DataFrame) -> ValidationResult:
    """
    Check that a raw log DataFrame is usable before cleaning / feature
    engineering. This function never raises: it always returns a
    ValidationResult so callers (Streamlit app, tests) can decide how to
    react and show a friendly message instead of crashing.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        errors.append("The dataset is empty - no log records were found.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    required_columns = (
        [TIMESTAMP_COLUMN] + REQUIRED_NUMERICAL_COLUMNS + REQUIRED_CATEGORICAL_COLUMNS
    )
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
        # Can't safely inspect columns that don't exist - stop here.
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # --- Timestamp must be parseable ---
    parsed_ts = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    bad_ts = int(parsed_ts.isna().sum())
    if bad_ts:
        warnings.append(
            f"{bad_ts} row(s) have an unparseable '{TIMESTAMP_COLUMN}' and will be dropped."
        )

    # --- Numeric columns must contain valid numbers ---
    for col in REQUIRED_NUMERICAL_COLUMNS:
        original_missing = int(df[col].isna().sum())
        coerced = pd.to_numeric(df[col], errors="coerce")
        newly_invalid = int(coerced.isna().sum()) - original_missing
        if newly_invalid > 0:
            warnings.append(
                f"Column '{col}' has {newly_invalid} non-numeric value(s) that will be dropped."
            )
        if original_missing:
            warnings.append(f"Column '{col}' has {original_missing} missing value(s).")

    # --- Categorical columns must have usable (non-blank) values ---
    for col in REQUIRED_CATEGORICAL_COLUMNS:
        blank = int(
            df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        )
        if blank:
            warnings.append(
                f"Column '{col}' has {blank} missing/blank value(s) "
                "(will be filled with 'unknown')."
            )

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} exact duplicate row(s) found and will be removed.")

    return ValidationResult(is_valid=True, errors=errors, warnings=warnings)


# =======================================================================
# 2. Cleaning
# =======================================================================
def clean_logs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Produce a clean copy of the raw logs:
      - drop exact duplicate rows
      - drop rows with an unparseable timestamp
      - coerce numeric columns to numeric, dropping rows that still fail
      - fill missing/blank categorical values with 'unknown' (kept, not dropped)

    Returns (cleaned_df, cleaning_report). The report records exactly how
    many rows were removed and why, for transparency on the dashboard.
    """
    original_rows = len(df)
    working = df.copy()

    # --- Duplicates ---
    before = len(working)
    working = working.drop_duplicates()
    duplicates_removed = before - len(working)

    # --- Timestamp ---
    working[TIMESTAMP_COLUMN] = pd.to_datetime(working[TIMESTAMP_COLUMN], errors="coerce")
    before = len(working)
    working = working[working[TIMESTAMP_COLUMN].notna()]
    invalid_timestamp_removed = before - len(working)

    # --- Numeric columns ---
    before = len(working)
    for col in REQUIRED_NUMERICAL_COLUMNS:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=REQUIRED_NUMERICAL_COLUMNS)
    invalid_numeric_removed = before - len(working)

    # --- Categorical columns: fill rather than drop ---
    for col in REQUIRED_CATEGORICAL_COLUMNS:
        working[col] = working[col].astype(str).str.strip()
        working.loc[working[col].isin(["", "nan", "None", "NaN"]), col] = "unknown"

    working = working.reset_index(drop=True)

    report = {
        "original_rows": original_rows,
        "cleaned_rows": len(working),
        "rows_removed": original_rows - len(working),
        "duplicates_removed": int(duplicates_removed),
        "invalid_timestamp_removed": int(invalid_timestamp_removed),
        "invalid_numeric_removed": int(invalid_numeric_removed),
    }
    return working, report


# =======================================================================
# 3 & 4. Timestamp processing + feature engineering
# =======================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add human-readable derived features to an already-cleaned log
    DataFrame. Never reads or writes `true_label`. Output is still meant
    for DISPLAY (level B) - `prepare_ml_features()` turns a subset of it
    into the final numeric matrix.

    Derived features added:
      - hour              : hour of day (0-23) the event occurred, from timestamp
      - day_of_week       : day name (Monday..Sunday), from timestamp
      - is_off_hours      : 1 if hour falls in 22:00-23:59 or 00:00-05:59, else 0
      - request_rate      : request_count / session_duration (requests per second)
      - failed_login_ratio: failed_logins / (failed_logins + request_count)
      - is_external_source_ip : 1 if source_ip is outside the internal 10.x.x.x range
      - location_mismatch : 1 if this row's login_location differs from the
                             user's own most common login_location in the dataset
                             (a simple, label-free stand-in for "impossible travel")
    """
    out = df.copy()

    # --- Timestamp-based features ---
    out["hour"] = out[TIMESTAMP_COLUMN].dt.hour
    out["day_of_week"] = out[TIMESTAMP_COLUMN].dt.day_name()
    out["is_off_hours"] = (
        (out["hour"] >= OFF_HOURS_START_HOUR) | (out["hour"] < OFF_HOURS_END_HOUR)
    ).astype(int)

    # --- Safe ratio features (guarded against division by zero) ---
    out["request_rate"] = np.where(
        out["session_duration"] > 0,
        out["request_count"] / out["session_duration"],
        0.0,
    )

    total_attempts = out["failed_logins"] + out["request_count"]
    out["failed_login_ratio"] = np.where(
        total_attempts > 0, out["failed_logins"] / total_attempts, 0.0
    )

    # --- Simple identifier-derived signals (replace raw IP / location) ---
    out["is_external_source_ip"] = (~out["source_ip"].astype(str).str.startswith("10.")).astype(int)

    def _usual_location(series: pd.Series) -> pd.Series:
        mode = series.mode()
        fallback = mode.iloc[0] if not mode.empty else series.iloc[0]
        return pd.Series(fallback, index=series.index)

    usual_location = out.groupby("user")["login_location"].transform(_usual_location)
    out["location_mismatch"] = (out["login_location"] != usual_location).astype(int)

    return out


# =======================================================================
# 5 & 6. Categorical encoding + numerical scaling -> ML-ready features
# =======================================================================
def prepare_ml_features(
    feature_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler, Dict[str, List[str]]]:
    """
    Build the final numeric, ML-ready feature matrix (level C) from the
    engineered feature DataFrame (level B).

    Design choices (kept deliberately simple):
      - `true_label` is never touched - it's simply not in the allow-list
        of columns used to build this matrix.
      - Identifier-style columns (user, source_ip, destination_ip,
        login_location) and the raw timestamp are NOT one-hot encoded -
        their useful signal was already captured as small numeric features
        in engineer_features() (is_external_source_ip, location_mismatch,
        hour, is_off_hours). One-hot encoding raw IPs/usernames directly
        would blow up the feature space for little benefit.
      - `event_type` is the one remaining categorical column and has low
        cardinality, so it IS one-hot encoded.
      - All numeric columns (continuous + the 0/1 engineered flags) are
        scaled together with a single StandardScaler for simplicity.
        Tree-based models (e.g. Isolation Forest, planned for Phase 2)
        are insensitive to scaling anyway; this also keeps the pipeline
        ready for distance-based methods such as a SOM.

    Returns
    -------
    ml_df : pd.DataFrame
        Fully numeric, scaled + encoded feature matrix, one row per
        surviving log record, index-aligned with feature_df.
    scaler : StandardScaler
        The fitted scaler (kept so the same transform could be reapplied
        to new data in a later phase).
    feature_groups : dict
        Names of the columns in each stage, for display/testing.
    """
    df = feature_df.copy()

    numerical_features = list(REQUIRED_NUMERICAL_COLUMNS) + [
        "hour",
        "request_rate",
        "failed_login_ratio",
    ]
    binary_features = ["is_off_hours", "is_external_source_ip", "location_mismatch"]

    onehot_df = pd.get_dummies(
        df[LOW_CARDINALITY_CATEGORICAL_COLUMNS],
        prefix=LOW_CARDINALITY_CATEGORICAL_COLUMNS,
    ).astype(int)
    onehot_features = list(onehot_df.columns)

    numeric_block = df[numerical_features + binary_features].astype(float)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_block)
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_block.columns, index=df.index)

    ml_df = pd.concat([scaled_df, onehot_df], axis=1)

    feature_groups = {
        "numerical_features": numerical_features,
        "binary_features": binary_features,
        "onehot_features": onehot_features,
        "all_ml_features": list(ml_df.columns),
    }
    return ml_df, scaler, feature_groups


# =======================================================================
# Orchestration - the one function most callers need
# =======================================================================
def run_preprocessing_pipeline(df: pd.DataFrame) -> PreprocessingResult:
    """
    Run the full Phase 1 pipeline end-to-end:

        raw -> validate_logs -> clean_logs -> engineer_features
            -> prepare_ml_features

    Designed to never raise for "bad but non-catastrophic" data (missing
    values, duplicates, bad timestamps, etc.) - those are handled and
    reported. It only returns an unusable (empty) result, with the reason
    recorded in `.validation.errors`, if the data is fundamentally unusable
    (empty dataset or missing required columns), so the Streamlit app can
    show a clear message instead of crashing.
    """
    validation = validate_logs(df)

    if not validation.is_valid:
        empty = pd.DataFrame()
        return PreprocessingResult(
            raw_df=df if df is not None else empty,
            validation=validation,
            cleaned_df=empty,
            cleaning_report={},
            feature_df=empty,
            ml_df=empty,
            numerical_features=[],
            derived_features=[],
            onehot_features=[],
            excluded_columns=[],
            scaler=None,
        )

    cleaned_df, cleaning_report = clean_logs(df)

    if cleaned_df.empty:
        validation.errors.append(
            "Every row was removed during cleaning (all data was invalid) - "
            "nothing left to build features from."
        )
        validation.is_valid = False
        empty = pd.DataFrame()
        return PreprocessingResult(
            raw_df=df,
            validation=validation,
            cleaned_df=cleaned_df,
            cleaning_report=cleaning_report,
            feature_df=empty,
            ml_df=empty,
            numerical_features=[],
            derived_features=[],
            onehot_features=[],
            excluded_columns=[],
            scaler=None,
        )

    feature_df = engineer_features(cleaned_df)
    ml_df, scaler, feature_groups = prepare_ml_features(feature_df)

    derived_features = [
        "hour",
        "day_of_week",
        "is_off_hours",
        "request_rate",
        "failed_login_ratio",
        "is_external_source_ip",
        "location_mismatch",
    ]
    excluded_columns = [LABEL_COLUMN, TIMESTAMP_COLUMN] + IDENTIFIER_COLUMNS

    return PreprocessingResult(
        raw_df=df,
        validation=validation,
        cleaned_df=cleaned_df,
        cleaning_report=cleaning_report,
        feature_df=feature_df,
        ml_df=ml_df,
        numerical_features=feature_groups["numerical_features"] + feature_groups["binary_features"],
        derived_features=derived_features,
        onehot_features=feature_groups["onehot_features"],
        excluded_columns=excluded_columns,
        scaler=scaler,
    )
