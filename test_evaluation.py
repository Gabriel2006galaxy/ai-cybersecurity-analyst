"""
test_evaluation.py
-------------------
Phase 11 tests for `src/evaluation.py` ONLY.

Same style as test_explainability.py / test_audit_trail.py: plain
asserts, no pytest, and this file deliberately does NOT chain into the
Phase 1-10 regression cascade (unlike `test_retriever.py`, which re-runs
`test_feedback.main()` and so on down the chain). Run it directly:

    python test_evaluation.py

A full project regression (`python test_retriever.py`, 1,196+ checks) can
still be run separately whenever a complete check is actually needed -
this file is intentionally scoped to Phase 11's evaluation module and its
Streamlit-page integration only, per the Phase 11 instruction not to
repeatedly run the entire previous regression suite during this task.

Every test that needs a feedback/audit database uses its OWN temporary
SQLite file (never `config.FEEDBACK_DB_PATH`/`config.AUDIT_DB_PATH`, the
application's real databases), so running this suite never pollutes real
analyst feedback or the real audit trail.

Never modifies data/synthetic_security_logs.csv.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

import config
import src.audit_trail as audit_trail_module
import src.evaluation as evaluation_module
import src.feedback as feedback_module
from config import LABEL_COLUMN
from src.anomaly_detector import evaluate_against_ground_truth, run_anomaly_detection
from src.audit_trail import initialize_audit_database, write_audit_event
from src.confidence import run_confidence_scoring
from src.data_loader import load_logs
from src.evaluation import (
    build_evaluation_report,
    evaluate_anomaly_detection,
    evaluate_audit,
    evaluate_classification,
    evaluate_confidence,
    evaluate_dataset_overview,
    evaluate_feedback,
    evaluate_genai,
    evaluate_rag,
    measure_anomaly_detection_timing,
    measure_rag_retrieval_timing,
)
from src.feedback import (
    FeedbackRecord,
    build_incident_id,
    generate_feedback_id,
    initialize_feedback_database,
    plan_to_dict,
    save_feedback,
)
from src.preprocessor import run_preprocessing_pipeline
from src.response_planner import generate_response_plan
from src.rule_engine import run_rule_engine

PASSED = 0
_TEMP_DIRS = []


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1


def fresh_db_path(suffix: str) -> str:
    directory = tempfile.mkdtemp()
    _TEMP_DIRS.append(directory)
    return os.path.join(directory, f"{suffix}.db")


def cleanup_temp_dirs() -> None:
    import shutil

    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _TEMP_DIRS.clear()


@contextmanager
def isolated_databases():
    """Patches `src.feedback.FEEDBACK_DB_PATH` and
    `src.audit_trail.AUDIT_DB_PATH` to fresh temp files - never touches
    the real `data/analyst_feedback.db` or `data/audit_trail.db`."""
    original_feedback_path = feedback_module.FEEDBACK_DB_PATH
    original_audit_path = audit_trail_module.AUDIT_DB_PATH
    feedback_module.FEEDBACK_DB_PATH = fresh_db_path("feedback_test")
    audit_trail_module.AUDIT_DB_PATH = fresh_db_path("audit_test")
    try:
        initialize_feedback_database()
        initialize_audit_database()
        yield
    finally:
        feedback_module.FEEDBACK_DB_PATH = original_feedback_path
        audit_trail_module.AUDIT_DB_PATH = original_audit_path


# =======================================================================
# Shared real-pipeline fixture (computed once, reused by every test that
# needs the real dataset - avoids retraining Isolation Forest repeatedly)
# =======================================================================
_PIPELINE_CACHE = {}


def _pipeline():
    if not _PIPELINE_CACHE:
        df = load_logs()
        pre = run_preprocessing_pipeline(df)
        anomaly_result = run_anomaly_detection(pre)
        rule_result = run_rule_engine(pre, anomaly_result)
        confidence_result = run_confidence_scoring(rule_result)
        _PIPELINE_CACHE["pre"] = pre
        _PIPELINE_CACHE["anomaly_result"] = anomaly_result
        _PIPELINE_CACHE["rule_result"] = rule_result
        _PIPELINE_CACHE["confidence_result"] = confidence_result
    return (
        _PIPELINE_CACHE["pre"],
        _PIPELINE_CACHE["anomaly_result"],
        _PIPELINE_CACHE["rule_result"],
        _PIPELINE_CACHE["confidence_result"],
    )


# =======================================================================
# 1. Dataset overview counts
# =======================================================================
def test_dataset_overview_counts():
    _, anomaly_result, _, _ = _pipeline()
    overview = evaluate_dataset_overview(anomaly_result)

    check(overview.total_events == anomaly_result.total_events, "total_events matches AnomalyDetectionResult")
    check(overview.predicted_normal_count == anomaly_result.normal_events, "predicted normal count matches")
    check(overview.predicted_anomalous_count == anomaly_result.anomalous_events, "predicted anomalous count matches")
    check(
        overview.predicted_normal_count + overview.predicted_anomalous_count == overview.total_events,
        "predicted normal + anomalous == total",
    )
    check(overview.ground_truth_available is True, "ground truth is available on the real dataset")
    check(
        overview.ground_truth_normal_count + overview.ground_truth_anomalous_count == overview.total_events,
        "ground truth normal + anomalous == total",
    )
    # Cross-check against a hand count straight off the dataframe.
    df = anomaly_result.results_df
    hand_anom = int((df[LABEL_COLUMN] == "suspicious").sum())
    check(overview.ground_truth_anomalous_count == hand_anom, "ground truth anomalous count matches a hand count")


# =======================================================================
# 2-3. Anomaly metrics, confusion matrix, precision/recall/F1
# =======================================================================
def test_anomaly_metrics_match_reused_function():
    _, anomaly_result, _, _ = _pipeline()
    metrics = evaluate_anomaly_detection(anomaly_result)
    direct = evaluate_against_ground_truth(anomaly_result.results_df)

    check(metrics.measured is True, "anomaly metrics are measured when true_label is present")
    check(abs(metrics.accuracy - direct["accuracy"]) < 1e-9, "accuracy matches the reused evaluate_against_ground_truth()")
    check(abs(metrics.precision - direct["precision"]) < 1e-9, "precision matches")
    check(abs(metrics.recall - direct["recall"]) < 1e-9, "recall matches")
    check(abs(metrics.f1_score - direct["f1_score"]) < 1e-9, "f1_score matches")
    check(0.0 <= metrics.accuracy <= 1.0, "accuracy is a valid rate")
    check(0.0 <= metrics.precision <= 1.0, "precision is a valid rate")
    check(0.0 <= metrics.recall <= 1.0, "recall is a valid rate")
    check(0.0 <= metrics.f1_score <= 1.0, "f1_score is a valid rate")


def test_confusion_matrix_values():
    _, anomaly_result, _, _ = _pipeline()
    metrics = evaluate_anomaly_detection(anomaly_result)
    cm = metrics.confusion_matrix

    check(len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2, "confusion matrix is 2x2")
    total_from_cm = sum(sum(row) for row in cm)
    check(total_from_cm == metrics.total_events, "confusion matrix cells sum to total_events")
    check(metrics.confusion_matrix_labels == ["normal", "suspicious"], "confusion matrix labels are normal/suspicious")
    check(
        cm[0][0] + cm[0][1] == metrics.ground_truth_normal,
        "confusion matrix row 0 sums to ground-truth normal support",
    )
    check(
        cm[1][0] + cm[1][1] == metrics.ground_truth_anomalous,
        "confusion matrix row 1 sums to ground-truth suspicious support",
    )


def test_anomaly_metrics_no_ground_truth_handled_safely():
    _, anomaly_result, _, _ = _pipeline()
    stripped_df = anomaly_result.results_df.drop(columns=[LABEL_COLUMN])
    fake_result = evaluation_module.AnomalyDetectionResult(
        results_df=stripped_df,
        model=anomaly_result.model,
        ml_features_used=anomaly_result.ml_features_used,
        n_estimators=anomaly_result.n_estimators,
        contamination=anomaly_result.contamination,
        random_state=anomaly_result.random_state,
        total_events=anomaly_result.total_events,
        normal_events=anomaly_result.normal_events,
        anomalous_events=anomaly_result.anomalous_events,
        anomaly_percentage=anomaly_result.anomaly_percentage,
    )
    metrics = evaluate_anomaly_detection(fake_result)

    check(metrics.measured is False, "measured is False with no true_label column")
    check(metrics.accuracy is None, "accuracy is None (not fabricated) with no ground truth")
    check(metrics.precision is None, "precision is None (not fabricated) with no ground truth")
    check(metrics.confusion_matrix is None, "confusion_matrix is None (not fabricated) with no ground truth")
    check(bool(metrics.note), "a clear note explains why nothing was measured")
    # Prediction counts never require ground truth - still reported.
    check(metrics.predicted_anomalous == anomaly_result.anomalous_events, "prediction counts still shown without ground truth")


# =======================================================================
# 4-6. Classification distribution, rule counts, multi-rule events
# =======================================================================
def test_classification_metrics():
    _, _, rule_result, _ = _pipeline()
    metrics = evaluate_classification(rule_result)
    df = rule_result.results_df

    check(metrics.total_events == rule_result.total_events, "total_events matches RuleEngineResult")
    check(sum(metrics.incident_type_counts.values()) == metrics.total_events, "incident type counts sum to total")
    check(
        sum(metrics.severity_counts.values()) == metrics.total_events,
        "severity counts sum to total (N/A included for Normal Activity)",
    )
    # Cross-check incident_type_counts against a hand count.
    hand_counts = df["incident_type"].value_counts().to_dict()
    for incident_type, count in hand_counts.items():
        check(metrics.incident_type_counts.get(str(incident_type)) == int(count), f"incident_type_counts correct for {incident_type}")

    # Cross-check rule firing counts against a hand count over triggered_rule_ids.
    hand_rule_counts = {}
    hand_multi = 0
    for ids in df["triggered_rule_ids"]:
        ids = ids or []
        if len(ids) >= 2:
            hand_multi += 1
        for rid in ids:
            hand_rule_counts[rid] = hand_rule_counts.get(rid, 0) + 1
    check(metrics.rule_firing_counts == hand_rule_counts, "rule_firing_counts matches a hand count")
    check(metrics.multi_rule_event_count == hand_multi, "multi_rule_event_count matches a hand count")
    check(metrics.no_rule_anomaly_count >= 0, "no_rule_anomaly_count is non-negative")


def test_classification_metrics_empty_dataframe():
    empty_result = evaluation_module.RuleEngineResult(
        results_df=pd.DataFrame(), rules=[], total_events=0, normal_count=0, low_count=0, medium_count=0, high_count=0,
    )
    metrics = evaluate_classification(empty_result)
    check(metrics.total_events == 0, "empty rule result -> total_events == 0")
    check(metrics.incident_type_counts == {}, "empty rule result -> no incident type counts")
    check(metrics.rule_firing_counts == {}, "empty rule result -> no rule firing counts")


# =======================================================================
# 7-8. Confidence distribution, human-review counts
# =======================================================================
def test_confidence_metrics():
    _, _, _, confidence_result = _pipeline()
    metrics = evaluate_confidence(confidence_result)

    check(metrics.total_events == confidence_result.total_events, "total_events pass-through matches")
    check(metrics.high_count == confidence_result.high_confidence_count, "high_count pass-through matches")
    check(metrics.medium_count == confidence_result.medium_confidence_count, "medium_count pass-through matches")
    check(metrics.low_count == confidence_result.low_confidence_count, "low_count pass-through matches")
    check(metrics.human_review_count == confidence_result.human_review_count, "human_review_count pass-through matches")
    check(
        metrics.high_count + metrics.medium_count + metrics.low_count == metrics.total_events,
        "HIGH + MEDIUM + LOW == total_events",
    )
    # Cross-check against a hand count on the underlying dataframe.
    df = confidence_result.results_df
    hand_high = int((df["confidence_level"] == "HIGH").sum())
    check(metrics.high_count == hand_high, "high_count matches a hand count on confidence_level")


# =======================================================================
# 9. RAG metrics
# =======================================================================
def test_rag_metrics():
    _, _, _, confidence_result = _pipeline()
    metrics = evaluate_rag(confidence_result)

    check(metrics.knowledge_base_size > 0, "knowledge base is non-empty for the real project")
    anomalous_count = int((confidence_result.results_df["incident_type"] != "Normal Activity").sum())
    check(metrics.retrieval_attempted_count == anomalous_count, "one retrieval attempted per anomalous event")
    check(
        metrics.retrieval_hit_count + metrics.retrieval_empty_count == metrics.retrieval_attempted_count,
        "hit count + empty count == attempted count",
    )
    check(metrics.controlled_eligible_count <= metrics.retrieval_hit_count, "controlled-eligible count never exceeds hits")
    if metrics.controlled_success_rate is not None:
        check(0.0 <= metrics.controlled_success_rate <= 1.0, "controlled success rate is a valid rate")
        check(
            metrics.controlled_success_count <= metrics.controlled_eligible_count,
            "controlled success count never exceeds eligible count",
        )
    check(sum(metrics.top_match_counts_by_incident_type.values()) == metrics.retrieval_hit_count, "top-match counts sum to hit count")


def test_rag_metrics_deterministic():
    _, _, _, confidence_result = _pipeline()
    first = evaluate_rag(confidence_result)
    second = evaluate_rag(confidence_result)
    check(first.retrieval_attempted_count == second.retrieval_attempted_count, "RAG evaluation is deterministic (attempted)")
    check(first.retrieval_hit_count == second.retrieval_hit_count, "RAG evaluation is deterministic (hits)")
    check(
        first.controlled_success_count == second.controlled_success_count,
        "RAG evaluation is deterministic (controlled success)",
    )


def test_rag_metrics_no_anomalous_events():
    empty_df = pd.DataFrame({"incident_type": ["Normal Activity", "Normal Activity"]})
    fake_confidence = evaluation_module.ConfidenceEngineResult(
        results_df=empty_df, total_events=2, high_confidence_count=2, medium_confidence_count=0,
        low_confidence_count=0, human_review_count=0,
    )
    metrics = evaluate_rag(fake_confidence)
    check(metrics.retrieval_attempted_count == 0, "no anomalous events -> nothing attempted")
    check(bool(metrics.note), "a clear note explains why nothing was attempted")


# =======================================================================
# 10. GenAI metrics when measurable
# =======================================================================
def test_genai_metrics_no_audit_events():
    with isolated_databases():
        metrics = evaluate_genai()
        check(metrics.measured is False, "no GENAI_ANALYSIS audit events -> measured is False")
        check(metrics.generations_attempted == 0, "0 attempted (not fabricated) with an empty audit trail")
        check(metrics.successful_generations == 0, "0 successful (not fabricated) with an empty audit trail")
        check(metrics.fallback_generations == 0, "0 fallback (not fabricated) with an empty audit trail")
        check(metrics.model_name == config.OLLAMA_MODEL_NAME, "model_name reflects config.OLLAMA_MODEL_NAME")
        check(bool(metrics.note), "a clear note explains why nothing is measured")


def test_genai_metrics_parses_real_audit_events():
    with isolated_databases():
        write_audit_event(
            "INC-0000", "GENAI_ANALYSIS", "GENAI_ANALYSIS_RESULT",
            ai_model_name="qwen3:1.7b", descriptive_details="source=ollama, used_fallback=False.",
        )
        write_audit_event(
            "INC-0001", "GENAI_ANALYSIS", "GENAI_ANALYSIS_RESULT",
            ai_model_name="qwen3:1.7b", descriptive_details="source=fallback, used_fallback=True.",
        )
        write_audit_event(
            "INC-0002", "GENAI_ANALYSIS", "GENAI_ANALYSIS_RESULT",
            ai_model_name="qwen3:1.7b", descriptive_details="source=fallback, used_fallback=True.",
        )
        metrics = evaluate_genai()

        check(metrics.measured is True, "measured is True once GENAI_ANALYSIS events exist")
        check(metrics.generations_attempted == 3, "generations_attempted counts every GENAI_ANALYSIS event")
        check(metrics.successful_generations == 1, "successful_generations counts used_fallback=False events")
        check(metrics.fallback_generations == 2, "fallback_generations counts used_fallback=True events")
        check(metrics.source_breakdown.get("ollama") == 1, "source_breakdown counts the ollama source")
        check(metrics.source_breakdown.get("fallback") == 2, "source_breakdown counts the fallback source")
        check(
            metrics.successful_generations + metrics.fallback_generations == metrics.generations_attempted,
            "successful + fallback == attempted",
        )


# =======================================================================
# 11. Feedback metrics
# =======================================================================
def _real_pair():
    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly_result = run_anomaly_detection(pre)
    rule_result = run_rule_engine(pre, anomaly_result)
    confidence_result = run_confidence_scoring(rule_result)
    df2 = confidence_result.results_df
    anomalous_rows = df2[df2["incident_type"] != "Normal Activity"]
    row = anomalous_rows.iloc[0]
    return row["_inference"], row["_confidence"]


def test_feedback_metrics_empty():
    with isolated_databases():
        metrics = evaluate_feedback()
        check(metrics.total_reviews == 0, "no feedback saved -> total_reviews == 0")
        check(metrics.approval_rate == 0.0, "no feedback saved -> approval_rate == 0.0 (not fabricated)")


def test_feedback_metrics_with_records():
    with isolated_databases():
        inf, conf = _real_pair()
        plan = generate_response_plan(inf, conf)
        incident_id = build_incident_id(0)

        for decision in ("APPROVE", "APPROVE", "REJECT"):
            record = FeedbackRecord(
                feedback_id=generate_feedback_id(incident_id),
                incident_id=incident_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                incident_type=inf.incident_type,
                severity=inf.severity,
                confidence_level=conf.confidence_level,
                confidence_score=conf.confidence_score,
                human_review_recommended=conf.human_review_recommended,
                analyst_decision=decision,
                review_status="APPROVED" if decision == "APPROVE" else "REJECTED",
                original_plan=plan_to_dict(plan),
                analyst_notes="Test review.",
                triggered_rule_ids=[tr.rule_id for tr in inf.triggered_rules],
                supporting_indicators=list(conf.supporting_indicators),
            )
            success, errors = save_feedback(record)
            check(success, f"seed feedback record saved successfully ({errors})")

        metrics = evaluate_feedback()
        check(metrics.total_reviews == 3, "total_reviews matches the 3 seeded records")
        check(metrics.approved_count == 2, "approved_count matches the seeded data")
        check(metrics.rejected_count == 1, "rejected_count matches the seeded data")
        check(abs(metrics.approval_rate - round(2 / 3, 4)) < 1e-9, "approval_rate computed correctly")


# =======================================================================
# 12. Audit metrics
# =======================================================================
def test_audit_metrics_empty():
    with isolated_databases():
        metrics = evaluate_audit()
        check(metrics.total_events == 0, "empty audit trail -> total_events == 0")
        check(
            all(v == 0 for v in metrics.events_per_stage.values()),
            "empty audit trail -> every stage count is 0 (not fabricated)",
        )
        check(set(metrics.events_per_stage.keys()) == set(config.AUDIT_STAGES), "every configured stage is represented")


def test_audit_metrics_grouped_by_stage():
    with isolated_databases():
        write_audit_event("INC-0000", "ANOMALY_DETECTION", "ANOMALY_DETECTION_RESULT")
        write_audit_event("INC-0000", "RULE_CLASSIFICATION", "RULE_CLASSIFICATION_RESULT")
        write_audit_event("INC-0001", "ANOMALY_DETECTION", "ANOMALY_DETECTION_RESULT")

        metrics = evaluate_audit()
        check(metrics.total_events == 3, "total_events counts every written event")
        check(metrics.events_per_stage["ANOMALY_DETECTION"] == 2, "events_per_stage groups ANOMALY_DETECTION correctly")
        check(metrics.events_per_stage["RULE_CLASSIFICATION"] == 1, "events_per_stage groups RULE_CLASSIFICATION correctly")
        check(metrics.events_per_stage["CONFIDENCE"] == 0, "an unwritten stage stays at 0 (not fabricated)")
        check(sum(metrics.events_per_stage.values()) == metrics.total_events, "per-stage counts sum to total_events")


# =======================================================================
# 13. No fabricated metrics / true_label boundary
# =======================================================================
def test_true_label_only_used_inside_evaluation_module():
    """Structural proof: none of the earlier-phase decision/compute
    functions this module could have imported read true_label, and
    LABEL_COLUMN is not imported into any earlier-phase module's
    namespace."""
    import src.audit_trail as audit_mod
    import src.confidence as confidence_mod
    import src.explainability as explainability_mod
    import src.feedback as feedback_mod
    import src.genai_analyzer as genai_mod
    import src.response_planner as response_mod
    import src.retriever as retriever_mod
    import src.rule_engine as rule_mod

    for name, mod in [
        ("rule_engine", rule_mod), ("confidence", confidence_mod), ("retriever", retriever_mod),
        ("genai_analyzer", genai_mod), ("response_planner", response_mod), ("feedback", feedback_mod),
        ("audit_trail", audit_mod), ("explainability", explainability_mod),
    ]:
        check("LABEL_COLUMN" not in vars(mod), f"LABEL_COLUMN is not imported into src.{name}")

    # evaluation.py itself IS allowed to read it - confirm it does, deliberately.
    check("LABEL_COLUMN" in vars(evaluation_module), "src.evaluation explicitly imports LABEL_COLUMN for offline evaluation")


def test_true_label_does_not_enter_earlier_pipeline_stages():
    """Behavioral proof: flipping true_label upstream never changes any
    earlier-phase decision - only the evaluation metrics that are
    supposed to read it change."""
    df = load_logs().copy()
    flipped = df.copy()
    flipped[LABEL_COLUMN] = flipped[LABEL_COLUMN].map(
        lambda v: "normal" if v == "suspicious" else "suspicious"
    )

    pre_a = run_preprocessing_pipeline(df)
    pre_b = run_preprocessing_pipeline(flipped)
    anomaly_a = run_anomaly_detection(pre_a)
    anomaly_b = run_anomaly_detection(pre_b)

    check(
        anomaly_a.results_df["anomaly_status"].tolist() == anomaly_b.results_df["anomaly_status"].tolist(),
        "flipping true_label never changes anomaly predictions",
    )
    rule_a = run_rule_engine(pre_a, anomaly_a)
    rule_b = run_rule_engine(pre_b, anomaly_b)
    check(
        rule_a.results_df["incident_type"].tolist() == rule_b.results_df["incident_type"].tolist(),
        "flipping true_label never changes rule-engine classification",
    )

    # But the evaluation module's own metrics DO change, because they are
    # explicitly allowed (and expected) to read true_label.
    metrics_a = evaluate_anomaly_detection(anomaly_a)
    metrics_b = evaluate_anomaly_detection(anomaly_b)
    check(
        metrics_a.ground_truth_anomalous != metrics_b.ground_truth_anomalous,
        "evaluation metrics DO change when true_label is flipped (as expected/intended)",
    )


def test_no_fabricated_metrics_end_to_end():
    """A metric that cannot be computed is always None/0/measured=False
    with an explanatory note, never a made-up number."""
    with isolated_databases():
        genai_metrics = evaluate_genai()
        feedback_metrics = evaluate_feedback()
        audit_metrics = evaluate_audit()

        check(genai_metrics.generations_attempted == 0 and not genai_metrics.measured, "GenAI: unmeasured, not fabricated")
        check(feedback_metrics.total_reviews == 0, "Feedback: unmeasured, not fabricated")
        check(audit_metrics.total_events == 0, "Audit: unmeasured, not fabricated")


# =======================================================================
# 14. Empty/missing data handling (build_evaluation_report end-to-end)
# =======================================================================
def test_build_evaluation_report_end_to_end():
    with isolated_databases():
        pre, anomaly_result, rule_result, confidence_result = _pipeline()
        report = build_evaluation_report(anomaly_result, rule_result, confidence_result)

        check(report.dataset_overview.total_events == anomaly_result.total_events, "report bundles dataset overview")
        check(report.anomaly_metrics.measured is True, "report bundles anomaly metrics")
        check(report.classification_metrics.total_events == rule_result.total_events, "report bundles classification metrics")
        check(report.confidence_metrics.total_events == confidence_result.total_events, "report bundles confidence metrics")
        check(report.rag_metrics.knowledge_base_size > 0, "report bundles RAG metrics")
        check(report.genai_metrics.model_name == config.OLLAMA_MODEL_NAME, "report bundles GenAI metrics")
        check(report.feedback_metrics.total_reviews == 0, "report bundles feedback metrics (empty isolated DB)")
        check(report.audit_metrics.total_events == 0, "report bundles audit metrics (empty isolated DB)")


def test_timing_measurements_do_not_crash_on_empty_input():
    empty_pre = evaluation_module  # placeholder attribute holder, real check below
    del empty_pre

    class _FakePre:
        ml_df = pd.DataFrame()

    timing = measure_anomaly_detection_timing(_FakePre())
    check(timing.measured is False, "anomaly timing on an empty feature matrix reports measured=False safely")

    empty_df = pd.DataFrame({"incident_type": []})
    fake_confidence = evaluation_module.ConfidenceEngineResult(
        results_df=empty_df, total_events=0, high_confidence_count=0, medium_confidence_count=0,
        low_confidence_count=0, human_review_count=0,
    )
    rag_timing = measure_rag_retrieval_timing(fake_confidence)
    check(rag_timing.measured is False, "RAG timing on an empty dataframe reports measured=False safely")


def test_timing_measurement_real_run():
    pre, _, _, confidence_result = _pipeline()
    timing = measure_anomaly_detection_timing(pre)
    check(timing.measured is True, "anomaly timing measures successfully on the real dataset")
    check(timing.elapsed_seconds is not None and timing.elapsed_seconds >= 0, "elapsed_seconds is a valid non-negative number")
    check(timing.sample_size == len(pre.ml_df), "sample_size matches the real feature matrix size")
    check("not a production benchmark" in timing.note.lower(), "timing note is clearly labeled as a local sample")

    rag_timing = measure_rag_retrieval_timing(confidence_result)
    check(rag_timing.measured is True, "RAG timing measures successfully on the real dataset")
    check(rag_timing.elapsed_seconds is not None and rag_timing.elapsed_seconds >= 0, "RAG elapsed_seconds is valid")


# =======================================================================
# 15. Streamlit Evaluation page loads / charts render without exceptions
# =======================================================================
def test_streamlit_evaluation_page_renders():
    # UI redesign note: "Evaluation & Performance" is no longer a
    # standalone page - it is the "Evaluation / Performance" section under
    # "Review & Insights" (a tab-styled section switcher, not st.tabs() -
    # st.tabs() would mount all seven sections' dataframes/charts into the
    # page at once, which in real-browser testing crashed the page, so
    # Review & Insights renders exactly one section per run and it must be
    # explicitly selected), and its A-H section headings are one level
    # below that section's own subheader (st.markdown("#### ..."), not
    # st.subheader) - the same "check markdown, not subheader" adaptation
    # already established in test_continuous_improvement.py.
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        check(len(at.exception) == 0, "the app loads on the Case Analysis (landing) page with no exception")

        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "the Review & Insights page renders with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Evaluation / Performance").run(timeout=180)
        check(len(at.exception) == 0, "the Evaluation / Performance section renders with no exception")
        check(any(s.value == "Evaluation / Performance" for s in at.subheader), "the Evaluation / Performance section is present")

        headings = [m.value for m in at.markdown]
        expected = [
            "#### A. Dataset Overview", "#### B. Anomaly Detection Performance", "#### C. Incident Analysis",
            "#### D. Confidence & Human Review", "#### E. RAG Performance", "#### F. Generative AI Performance",
            "#### G. Analyst Feedback", "#### H. Audit Activity", "#### Local Performance Timing",
        ]
        for section in expected:
            check(section in headings, f"section '{section.lstrip('# ')}' renders on the Evaluation / Performance tab")

        # Charts/tables render without exception (dataframes + plotly charts
        # both surface as elements when AppTest walks the page tree).
        check(len(at.dataframe) >= 1, "at least one table (e.g. the confusion matrix) renders")


def test_streamlit_evaluation_page_timing_button_works():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Evaluation / Performance").run(timeout=180)
        check(len(at.exception) == 0, "page renders before the timing button is used")

        buttons = [b for b in at.button if "timing sample" in (b.label or "").lower()]
        check(bool(buttons), "the optional GenAI timing-sample button is present")
        buttons[0].click().run(timeout=180)
        check(len(at.exception) == 0, "clicking the GenAI timing-sample button raises no exception")


# =======================================================================
# Runner
# =======================================================================
def main():
    tests = [
        test_dataset_overview_counts,
        test_anomaly_metrics_match_reused_function,
        test_confusion_matrix_values,
        test_anomaly_metrics_no_ground_truth_handled_safely,
        test_classification_metrics,
        test_classification_metrics_empty_dataframe,
        test_confidence_metrics,
        test_rag_metrics,
        test_rag_metrics_deterministic,
        test_rag_metrics_no_anomalous_events,
        test_genai_metrics_no_audit_events,
        test_genai_metrics_parses_real_audit_events,
        test_feedback_metrics_empty,
        test_feedback_metrics_with_records,
        test_audit_metrics_empty,
        test_audit_metrics_grouped_by_stage,
        test_true_label_only_used_inside_evaluation_module,
        test_true_label_does_not_enter_earlier_pipeline_stages,
        test_no_fabricated_metrics_end_to_end,
        test_build_evaluation_report_end_to_end,
        test_timing_measurements_do_not_crash_on_empty_input,
        test_timing_measurement_real_run,
        test_streamlit_evaluation_page_renders,
        test_streamlit_evaluation_page_timing_button_works,
    ]
    try:
        for i, test in enumerate(tests, 1):
            test()
            print(f"{i}. {test.__doc__.strip().splitlines()[0] if test.__doc__ else test.__name__}")
        print()
        print("=" * 70)
        print(f"PHASE 11 EVALUATION CHECKS PASSED ({PASSED} checks)")
        print("=" * 70)
    finally:
        cleanup_temp_dirs()


if __name__ == "__main__":
    main()
