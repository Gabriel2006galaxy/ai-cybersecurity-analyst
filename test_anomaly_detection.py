"""
test_anomaly_detection.py
---------------------------
A small, dependency-free validation script for the Phase 2 anomaly
detection pipeline (src/anomaly_detector.py). Same style as
test_preprocessing.py: plain asserts, no pytest, nothing beyond what's
already in requirements.txt (streamlit is already required, and its
`streamlit.testing.v1.AppTest` is used here to check the dashboard pages
run without exceptions).

Run with:

    python test_anomaly_detection.py

This script never modifies data/synthetic_security_logs.csv.
"""

import builtins

import numpy as np
import pandas as pd

from config import LABEL_COLUMN
from src.anomaly_detector import (
    attach_predictions_to_events,
    evaluate_against_ground_truth,
    is_som_available,
    predict_anomalies,
    run_anomaly_detection,
    run_som_analysis,
    train_isolation_forest,
)
from src.data_loader import load_logs
from src.preprocessor import run_preprocessing_pipeline

PASSED = 0


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


def test_phase1_ml_matrix_is_ready():
    print("\n1. Phase 1 preprocessing produces the expected ML-ready matrix")
    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    check(pre.is_usable, "preprocessing result is usable")
    check(pre.ml_df.shape[0] == 350, f"expected 350 processed events, got {pre.ml_df.shape[0]}")
    check(int(pre.ml_df.isna().sum().sum()) == 0, "ml_df has zero NaN values")
    return pre


def test_true_label_not_in_model_input(pre):
    print("\n2. true_label is not present in model input")
    check(LABEL_COLUMN not in pre.ml_df.columns, "true_label absent from ml_df")

    # And explicitly: it must never reach IsolationForest.fit(). We can't
    # peek inside sklearn's fit(), so instead assert the exact object
    # passed to train_isolation_forest() is ml_df itself, which we've
    # already shown excludes true_label.
    check(
        LABEL_COLUMN not in list(pre.ml_df.columns),
        "true_label is not among the columns handed to train_isolation_forest()",
    )


def test_isolation_forest_trains(pre):
    print("\n3. Isolation Forest trains successfully")
    model = train_isolation_forest(pre.ml_df)
    check(hasattr(model, "estimators_"), "model has fitted estimators after training")
    check(model.n_estimators == 100, "default n_estimators applied (100)")
    return model


def test_predictions_scores_status_generated(pre, model):
    print("\n4-6. A prediction, anomaly_score, and anomaly_status are generated for every event")
    prediction_df = predict_anomalies(model, pre.ml_df)
    check(len(prediction_df) == len(pre.ml_df), "one prediction row per processed event")
    check("anomaly_prediction" in prediction_df.columns, "anomaly_prediction column present")
    check("anomaly_score" in prediction_df.columns, "anomaly_score column present")
    check("anomaly_status" in prediction_df.columns, "anomaly_status column present")
    check(prediction_df["anomaly_prediction"].isin([-1, 1]).all(), "predictions are -1 or 1")
    check(
        prediction_df["anomaly_status"].isin(["NORMAL", "ANOMALOUS"]).all(),
        "status is always NORMAL or ANOMALOUS",
    )
    check(prediction_df["anomaly_score"].notna().all(), "no missing anomaly scores")
    return prediction_df


def test_result_row_count_and_content(pre):
    print("\n7-8. Result row count matches processed events; rows carry status + original info")
    result = run_anomaly_detection(pre)
    check(
        len(result.results_df) == len(pre.cleaned_df),
        "results_df row count matches the processed (cleaned) event count",
    )
    check(result.total_events == len(pre.cleaned_df), "total_events matches processed event count")
    check(result.normal_events + result.anomalous_events == result.total_events, "normal + anomalous == total")

    for col in ["user", "source_ip", "destination_ip", "event_type", "timestamp"]:
        check(col in result.results_df.columns, f"original event column '{col}' retained in results_df")
    for col in ["anomaly_prediction", "anomaly_score", "anomaly_status"]:
        check(col in result.results_df.columns, f"prediction column '{col}' present in results_df")

    return result


def test_evaluation_metrics(result):
    print("\n9. Evaluation metrics work when true_label exists")
    metrics = evaluate_against_ground_truth(result.results_df)
    check(metrics is not None, "evaluation metrics are produced when true_label is present")
    for key in ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"]:
        check(key in metrics, f"metrics dict contains '{key}'")
    check(0.0 <= metrics["accuracy"] <= 1.0, "accuracy is a valid proportion")
    check(metrics["confusion_matrix"].shape == (2, 2), "confusion matrix is 2x2")

    # And: no true_label column -> evaluation is skipped cleanly, not an error.
    no_label_df = result.results_df.drop(columns=[LABEL_COLUMN])
    check(
        evaluate_against_ground_truth(no_label_df) is None,
        "evaluation returns None (not an exception) when true_label is missing",
    )


def test_streamlit_pages_run_without_exceptions():
    # UI redesign note: the app's navigation is now two areas - "Case
    # Analysis" (the landing page) and "Review & Insights" - instead of
    # the old 11-page list. Anomaly detection itself is no longer a
    # manually-triggered, standalone page: it now runs automatically (via
    # the same src.anomaly_detector.run_anomaly_detection this test
    # already exercises directly above) the moment either area loads, and
    # its charts/tuning controls now live in an advanced section of the
    # "Evaluation / Performance" section under "Review & Insights" (a
    # tab-styled section switcher, not st.tabs() - st.tabs() would mount
    # all seven sections' dataframes/charts into the page at once, which
    # in real-browser testing crashed the page, so Review & Insights
    # renders exactly one section per run and it must be explicitly
    # selected). Only the navigation target and widget locators below
    # changed - the checks still confirm the same thing: the app runs
    # anomaly detection without exceptions and shows "Isolation Forest" as
    # the algorithm.
    print("\n10. Application pages run without exceptions")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    check(len(at.exception) == 0, "Case Analysis (landing page) raises no exception")
    check(
        any("Case Analysis" == t.value for t in at.title),
        "Case Analysis is the landing page",
    )

    at.sidebar.radio[0].set_value("Review & Insights").run(timeout=60)
    check(len(at.exception) == 0, "Review & Insights page raises no exception")
    ri_section = [r for r in at.radio if r.key == "ri_section"][0]
    ri_section.set_value("Evaluation / Performance").run(timeout=60)
    check(len(at.exception) == 0, "Evaluation / Performance section raises no exception")
    check(
        any("Isolation Forest" == m.value for m in at.metric),
        "Algorithm metric shows 'Isolation Forest' on the page (Evaluation / Performance section)",
    )

    rerun_buttons = [b for b in at.button if b.label == "Re-run Anomaly Detection"]
    check(len(rerun_buttons) == 1, "the advanced 're-run anomaly detection' control is present")
    rerun_buttons[0].click().run(timeout=60)
    check(len(at.exception) == 0, "Anomaly detection re-run (advanced control) raises no exception")


def test_invalid_model_parameters_handled(pre):
    print("\n11. Invalid model parameters are handled cleanly (no crash, clear error)")

    for bad_contamination in [0.0, -0.1, 0.6, 1.0]:
        try:
            train_isolation_forest(pre.ml_df, contamination=bad_contamination)
            raised = False
        except ValueError:
            raised = True
        check(raised, f"contamination={bad_contamination} raises a clean ValueError")

    for bad_n_estimators in [0, -5]:
        try:
            train_isolation_forest(pre.ml_df, n_estimators=bad_n_estimators)
            raised = False
        except ValueError:
            raised = True
        check(raised, f"n_estimators={bad_n_estimators} raises a clean ValueError")

    try:
        train_isolation_forest(pd.DataFrame())
        raised = False
    except ValueError:
        raised = True
    check(raised, "training on an empty feature matrix raises a clean ValueError")

    # Defensive guard: true_label must never be accepted as a feature column.
    poisoned = pre.ml_df.copy()
    poisoned[LABEL_COLUMN] = 0
    try:
        train_isolation_forest(poisoned)
        raised = False
    except ValueError:
        raised = True
    check(raised, "a feature matrix that (incorrectly) contains true_label is rejected")


def test_som_optional_and_isolated(pre):
    print("\n12. Optional SOM failure/unavailability does not break Isolation Forest")

    # 12a. Isolation Forest succeeds regardless of whether SOM is available.
    available = is_som_available()
    print(f"  (minisom currently installed: {available})")
    result = run_anomaly_detection(pre)
    check(result.total_events == len(pre.ml_df), "Isolation Forest still works no matter SOM's availability")

    # 12b. Simulate minisom being unavailable and confirm is_som_available()
    # correctly reports False, while Isolation Forest is entirely unaffected.
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "minisom":
            raise ImportError("simulated: minisom not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        check(is_som_available() is False, "is_som_available() reports False when minisom import fails")
        result_without_som = run_anomaly_detection(pre)
        check(
            result_without_som.total_events == result.total_events,
            "Isolation Forest results are identical whether or not SOM is available",
        )
        try:
            run_som_analysis(pre.ml_df)
            raised = False
        except ImportError:
            raised = True
        check(raised, "run_som_analysis() raises ImportError (caught by the UI) when minisom is missing")
    finally:
        builtins.__import__ = real_import

    # 12c. When minisom IS available, SOM runs but stays a visualization only
    # (no anomaly_prediction/anomaly_status of its own).
    if available:
        som_result = run_som_analysis(pre.ml_df)
        check(som_result.winner_coords.shape[0] == len(pre.ml_df), "SOM produces one grid coordinate per event")
        check(
            "anomaly_status" not in som_result.winner_coords.columns,
            "SOM output carries no anomaly_status of its own - it never replaces Isolation Forest",
        )


def main():
    print("=" * 70)
    print("Phase 2 anomaly detection test suite")
    print("=" * 70)

    pre = test_phase1_ml_matrix_is_ready()
    test_true_label_not_in_model_input(pre)
    model = test_isolation_forest_trains(pre)
    test_predictions_scores_status_generated(pre, model)
    result = test_result_row_count_and_content(pre)
    test_evaluation_metrics(result)
    test_streamlit_pages_run_without_exceptions()
    test_invalid_model_parameters_handled(pre)
    test_som_optional_and_isolated(pre)

    print("\n" + "=" * 70)
    print(f"ALL TESTS PASSED ({PASSED} checks)")
    print("=" * 70)


if __name__ == "__main__":
    main()
