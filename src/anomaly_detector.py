"""
anomaly_detector.py
--------------------
Phase 2 - AI-based anomaly detection.

    Cybersecurity Logs
            |
    Phase 1 Preprocessing
            |
      ML-Ready Features (ml_df)
            |
      Isolation Forest              <- PRIMARY, required algorithm
            |
      Anomaly Score
            |
      Normal / Anomalous
            |
    Optional Post-Hoc Evaluation (true_label)

This module trains scikit-learn's IsolationForest on the ML-ready feature
matrix produced by `src.preprocessor.run_preprocessing_pipeline()`, turns
its output into analyst-friendly predictions, and re-attaches those
predictions to the original (human-readable) event columns.

IMPORTANT - true_label:
    `true_label` (config.LABEL_COLUMN) is NEVER passed to the model. It is
    only used, after predictions already exist, by
    `evaluate_against_ground_truth()` to measure how well the unsupervised
    model agrees with the ground truth baked into the synthetic dataset.
    `run_anomaly_detection()` asserts this explicitly (see the guard right
    before `model.fit(...)`).

Optional Self-Organizing Map (SOM):
    `run_som_analysis()` is a small, self-contained, EXPLORATORY clustering
    visualization built on the same ml_df. It never produces anomaly
    predictions and never replaces Isolation Forest. If the optional
    `minisom` package isn't installed, `is_som_available()` reports that
    up front so the caller (the Streamlit page) can skip the section with
    a clear message instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE,
    LABEL_COLUMN,
    REQUIRED_CATEGORICAL_COLUMNS,
    REQUIRED_NUMERICAL_COLUMNS,
    SOM_GRID_SIZE,
    SOM_NUM_ITERATIONS,
    TIMESTAMP_COLUMN,
)
from src.preprocessor import PreprocessingResult

# Columns shown to the analyst alongside the model's predictions, so they
# can inspect WHY an event was flagged. Anything not present is skipped
# quietly (keeps this robust if the input schema ever changes slightly).
DISPLAY_COLUMNS: List[str] = (
    [TIMESTAMP_COLUMN] + ["user"] + REQUIRED_CATEGORICAL_COLUMNS[1:] + REQUIRED_NUMERICAL_COLUMNS
)
# ^ built from config lists to avoid hard-coding the same column names twice;
#   evaluates to: timestamp, user, source_ip, destination_ip, event_type,
#   login_location, failed_logins, request_count, port, data_transferred_mb,
#   session_duration, device_count


# =======================================================================
# Result containers
# =======================================================================
@dataclass
class AnomalyDetectionResult:
    """Bundles everything the Streamlit page needs after one detection run."""

    results_df: pd.DataFrame        # display columns + true_label (if any) + predictions
    model: IsolationForest
    ml_features_used: List[str]
    n_estimators: int
    contamination: float
    random_state: int
    total_events: int
    normal_events: int
    anomalous_events: int
    anomaly_percentage: float


@dataclass
class SomResult:
    """Result of the OPTIONAL, exploratory SOM clustering visualization."""

    grid_x: int
    grid_y: int
    winner_coords: pd.DataFrame     # columns: som_x, som_y - index-aligned with ml_df
    quantization_error: float


# =======================================================================
# 1. Training
# =======================================================================
def train_isolation_forest(
    ml_df: pd.DataFrame,
    n_estimators: Optional[int] = None,
    contamination: Optional[float] = None,
    random_state: Optional[int] = None,
) -> IsolationForest:
    """
    Fit an IsolationForest on an ML-ready feature matrix.

    Parameters
    ----------
    ml_df : pd.DataFrame
        Output of `src.preprocessor.prepare_ml_features()` /
        `run_preprocessing_pipeline().ml_df`. Must be fully numeric and
        must NOT contain `true_label`.
    n_estimators, contamination, random_state : optional overrides.
        Fall back to config.ISOLATION_FOREST_* when not given.

    Raises
    ------
    ValueError
        For an empty feature matrix, or clearly invalid parameters
        (validated here with a friendly message before sklearn's own,
        less friendly, validation would kick in).
    """
    if ml_df is None or ml_df.empty:
        raise ValueError("Cannot train Isolation Forest on an empty feature matrix.")

    if LABEL_COLUMN in ml_df.columns:
        # Defensive guard - should never happen given how prepare_ml_features()
        # is built (allow-list, not a drop-list), but fail loudly if it ever does.
        raise ValueError(
            f"'{LABEL_COLUMN}' must never be part of the model input features."
        )

    n_estimators = n_estimators if n_estimators is not None else ISOLATION_FOREST_N_ESTIMATORS
    contamination = (
        contamination if contamination is not None else ISOLATION_FOREST_CONTAMINATION
    )
    random_state = random_state if random_state is not None else ISOLATION_FOREST_RANDOM_STATE

    if not isinstance(n_estimators, (int, np.integer)) or n_estimators <= 0:
        raise ValueError(f"n_estimators must be a positive integer, got {n_estimators!r}.")
    if not isinstance(contamination, (int, float)) or not (0.0 < contamination <= 0.5):
        raise ValueError(
            f"contamination must be a number in the range (0, 0.5], got {contamination!r}."
        )

    model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=float(contamination),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(ml_df)
    return model


# =======================================================================
# 2. Prediction + anomaly score
# =======================================================================
def predict_anomalies(model: IsolationForest, ml_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a fitted IsolationForest on ml_df and package the outputs.

    Columns produced (index-aligned with ml_df):
      - anomaly_prediction : sklearn's raw output, -1 = anomaly, 1 = normal
      - anomaly_score_raw  : sklearn's decision_function(X) - HIGHER means
                             MORE NORMAL. This is NOT a probability; it
                             reflects how many random splits it took to
                             isolate the point (shifted so 0 sits at the
                             contamination-based cutoff).
      - anomaly_score      : `-anomaly_score_raw`, i.e. the same score
                             flipped so HIGHER means MORE ANOMALOUS - purely
                             for analyst-friendly reading/sorting ("higher
                             score = more suspicious"). The raw score is
                             kept alongside it, unmodified, for transparency.
      - anomaly_status     : "ANOMALOUS" if anomaly_prediction == -1, else "NORMAL"
    """
    if ml_df is None or ml_df.empty:
        raise ValueError("Cannot generate predictions for an empty feature matrix.")

    predictions = model.predict(ml_df)                 # -1 / 1
    raw_scores = model.decision_function(ml_df)         # higher = more normal
    anomaly_score = -raw_scores                          # higher = more anomalous (documented UI transform)
    status = np.where(predictions == -1, "ANOMALOUS", "NORMAL")

    return pd.DataFrame(
        {
            "anomaly_prediction": predictions,
            "anomaly_score_raw": raw_scores,
            "anomaly_score": anomaly_score,
            "anomaly_status": status,
        },
        index=ml_df.index,
    )


# =======================================================================
# 3. Attach predictions back onto the original/display event columns
# =======================================================================
def attach_predictions_to_events(
    display_df: pd.DataFrame, prediction_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join model predictions onto the human-readable event columns, aligned
    by (shared) index. Neither input is modified in place.
    """
    return display_df.join(prediction_df, how="left")


# =======================================================================
# Orchestration - the one function the Streamlit page calls
# =======================================================================
def run_anomaly_detection(
    preprocessing_result: PreprocessingResult,
    n_estimators: Optional[int] = None,
    contamination: Optional[float] = None,
    random_state: Optional[int] = None,
) -> AnomalyDetectionResult:
    """
    Full Phase 2 pipeline: train Isolation Forest on `ml_df`, predict, and
    reattach results to the original event columns for display.

    `true_label` is deliberately NEVER read in this function - it is only
    ever touched (later, separately) by `evaluate_against_ground_truth()`.
    """
    ml_df = preprocessing_result.ml_df
    model = train_isolation_forest(ml_df, n_estimators, contamination, random_state)
    prediction_df = predict_anomalies(model, ml_df)

    cleaned_df = preprocessing_result.cleaned_df
    display_cols = [c for c in DISPLAY_COLUMNS if c in cleaned_df.columns]
    display_df = cleaned_df[display_cols].copy()

    # true_label is carried along ONLY for later post-hoc evaluation -
    # it plays no role in anything above this line.
    if LABEL_COLUMN in cleaned_df.columns:
        display_df[LABEL_COLUMN] = cleaned_df[LABEL_COLUMN]

    results_df = attach_predictions_to_events(display_df, prediction_df)

    total = len(results_df)
    anomalous = int((results_df["anomaly_status"] == "ANOMALOUS").sum())
    normal = total - anomalous

    return AnomalyDetectionResult(
        results_df=results_df,
        model=model,
        ml_features_used=list(ml_df.columns),
        n_estimators=model.n_estimators,
        contamination=model.contamination,
        random_state=model.random_state,
        total_events=total,
        normal_events=normal,
        anomalous_events=anomalous,
        anomaly_percentage=round(100 * anomalous / total, 2) if total else 0.0,
    )


# =======================================================================
# 4. Optional post-hoc evaluation against true_label
# =======================================================================
def evaluate_against_ground_truth(
    results_df: pd.DataFrame, label_column: str = LABEL_COLUMN
) -> Optional[Dict]:
    """
    POST-HOC EVALUATION ONLY - called after `run_anomaly_detection()` has
    already produced predictions. Never used during training.

    Treats `true_label == "suspicious"` and `anomaly_status == "ANOMALOUS"`
    as the positive class ("1"), so the metrics below answer: "of the
    events the model flagged, how many were really suspicious, and vice
    versa?"

    Returns None if `label_column` isn't present (e.g. on real-world logs
    that have no ground truth), so callers can skip this section cleanly.
    """
    if label_column not in results_df.columns:
        return None
    if results_df.empty:
        return None

    y_true = (results_df[label_column] == "suspicious").astype(int)
    y_pred = (results_df["anomaly_status"] == "ANOMALOUS").astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["normal", "suspicious"],
        "support_normal": int((y_true == 0).sum()),
        "support_suspicious": int((y_true == 1).sum()),
    }


# =======================================================================
# 5. Optional Self-Organizing Map (exploratory clustering visualization)
# =======================================================================
def is_som_available() -> bool:
    """True if the optional `minisom` package is installed and importable."""
    try:
        import minisom  # noqa: F401
    except ImportError:
        return False
    return True


def run_som_analysis(
    ml_df: pd.DataFrame,
    grid_size: Optional[int] = None,
    num_iterations: Optional[int] = None,
    random_state: Optional[int] = None,
) -> SomResult:
    """
    OPTIONAL, exploratory only. Trains a small Self-Organizing Map on the
    same ML-ready features Isolation Forest uses, purely to visualise how
    events cluster in feature space (item 10 of the Phase 2 spec). It does
    NOT produce anomaly predictions of its own and never replaces
    Isolation Forest.

    Raises
    ------
    ImportError
        If `minisom` is not installed - callers should check
        `is_som_available()` first, and/or catch this, to skip the section
        with a clear message instead of crashing the app.
    ValueError
        For an empty feature matrix.
    """
    from minisom import MiniSom  # local import: keeps minisom fully optional

    if ml_df is None or ml_df.empty:
        raise ValueError("Cannot run SOM analysis on an empty feature matrix.")

    random_state = random_state if random_state is not None else ISOLATION_FOREST_RANDOM_STATE
    num_iterations = num_iterations if num_iterations is not None else SOM_NUM_ITERATIONS

    n_samples, n_features = ml_df.shape
    grid = grid_size or SOM_GRID_SIZE or max(3, int(np.ceil(np.sqrt(5 * np.sqrt(n_samples)))))

    data = ml_df.to_numpy(dtype=float)

    som = MiniSom(
        grid,
        grid,
        n_features,
        sigma=1.0,
        learning_rate=0.5,
        random_seed=random_state,
    )
    som.random_weights_init(data)
    som.train_random(data, num_iterations)

    winners = [som.winner(row) for row in data]
    winner_df = pd.DataFrame(winners, columns=["som_x", "som_y"], index=ml_df.index)
    q_error = float(som.quantization_error(data))

    return SomResult(grid_x=grid, grid_y=grid, winner_coords=winner_df, quantization_error=q_error)
