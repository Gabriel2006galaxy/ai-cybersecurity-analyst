"""
test_ui_polish.py
------------------
UI/UX polish tests for app.py ONLY.

These tests cover the presentation layer end to end: every page still
loads with no exception, the specific UI elements the interface promises
actually render (real status data, a consistent way to identify the case
being reviewed, the AI model status, the knowledge-base section, the full
Case Analysis decision workflow, Feedback / Continuous Improvement /
Explainability / Audit Trail / Evaluation in Review & Insights), empty/
error states stay exception-free, and the verified local baseline
(Qwen3:1.7B, think=False, the dataset, and the Phase 9/10/11 backend
outputs) is completely unaffected by any presentation change.

The app was redesigned around two top-level areas - "Case Analysis" (a
single case-centered workflow: select a case, see why it was flagged,
see what the AI used, see the recommended next steps, record a
decision) and "Review & Insights" (Case History, Analyst Decisions,
Feedback, Continuous Improvement, Explainability, Audit Trail, and
Evaluation / Performance, as tabs on one page) - replacing an earlier
eleven-separate-pages layout, including a dedicated "Overview" status
page. That redesign changed navigation and presentation only; it did not
change any detection, classification, confidence, RAG, GenAI,
response-planning, feedback, continuous-improvement, explainability,
audit, or evaluation LOGIC. Below, tests that used to target the retired
Overview page or the retired per-page layout are adapted to the same
real, non-fabricated data wherever it now lives, never to a placeholder.

Same style as test_evaluation.py / test_explainability.py: plain
asserts, no pytest, and this file deliberately does NOT chain into the
Phase 1-11 regression cascade. Run it directly:

    python test_ui_polish.py

Every test that needs a feedback/audit database uses its OWN temporary
SQLite file (never `config.FEEDBACK_DB_PATH`/`config.AUDIT_DB_PATH`, the
application's real databases), so running this suite never pollutes real
analyst feedback or the real audit trail.

Never modifies data/synthetic_security_logs.csv.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

import src.audit_trail as audit_trail_module
import src.feedback as feedback_module
from config import OLLAMA_MODEL_NAME
from src.anomaly_detector import run_anomaly_detection
from src.audit_trail import initialize_audit_database
from src.confidence import run_confidence_scoring
from src.continuous_improvement import generate_continuous_improvement_report
from src.data_loader import load_logs
from src.evaluation import build_evaluation_report
from src.explainability import build_incident_explanation
from src.feedback import (
    FeedbackRecord,
    build_incident_id,
    generate_feedback_id,
    initialize_feedback_database,
    plan_to_dict,
    save_feedback,
)
from src.genai_analyzer import get_ollama_status
from src.preprocessor import run_preprocessing_pipeline
from src.response_planner import generate_response_plan
from src.rule_engine import run_rule_engine

PASSED = 0
_TEMP_DIRS = []

DATASET_PATH = os.path.join("data", "synthetic_security_logs.csv")
EXPECTED_DATASET_CHECKSUM = "97d4aeff5a59bc2e472116e3c2b382c5"

NAV_AREAS = ["Case Analysis", "Review & Insights"]


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


def _first_anomalous_row(confidence_result):
    results_df = confidence_result.results_df
    anomalous = results_df[results_df["incident_type"] != "Normal Activity"]
    assert not anomalous.empty, "the real dataset has at least one anomalous event"
    idx = anomalous.index[0]
    return idx, results_df.loc[idx, "_inference"], results_df.loc[idx, "_confidence"]


# =======================================================================
# 1. Both top-level areas load with no exception (baseline, empty DBs)
# =======================================================================
def test_all_pages_load():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        check(len(at.exception) == 0, "the app loads on Case Analysis with no exception")

        for area in NAV_AREAS:
            at.sidebar.radio[0].set_value(area).run(timeout=180)
            check(len(at.exception) == 0, f"'{area}' renders with no exception")


# =======================================================================
# 2. Page titles/headers exist for both areas
# =======================================================================
def test_page_titles_exist():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        for area in NAV_AREAS:
            at.sidebar.radio[0].set_value(area).run(timeout=180)
            check(len(at.title) >= 1, f"'{area}' has a title")
            check(bool(at.title[0].value.strip()), f"'{area}' title is non-empty")
            check(at.title[0].value == area, f"'{area}' title matches its nav label")


# =======================================================================
# 3. Real, non-fabricated system status is visible (successor to the
#    retired Overview page's status dashboard - the same real data now
#    lives in Review & Insights -> Evaluation / Performance).
# =======================================================================
def test_review_insights_shows_real_system_status():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights renders with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Evaluation / Performance").run(timeout=180)
        check(len(at.exception) == 0, "Evaluation / Performance section renders with no exception")

        _, real_anomaly_result, _, _ = _pipeline()

        metric_map = {m.label: m for m in at.get("metric")}
        check("Total Events" in metric_map, "Evaluation / Performance shows a real dataset size")
        check(
            str(metric_map["Total Events"].value) == str(real_anomaly_result.total_events),
            "Total Events matches the real dataset, not a guessed number",
        )
        check("Model" in metric_map, "Evaluation / Performance shows the configured AI model")
        check(metric_map["Model"].value == OLLAMA_MODEL_NAME, "the model shown matches config.OLLAMA_MODEL_NAME")
        check("Ollama Available" in metric_map, "Evaluation / Performance shows real Ollama availability")
        check("Knowledge Base Size" in metric_map, "Evaluation / Performance shows the real knowledge-base size")

        # With no feedback recorded yet in this isolated database, this
        # must say so honestly rather than show a fabricated count.
        info_texts = [i.value for i in at.info]
        check(
            any("no analyst feedback is recorded yet" in (t or "").lower() for t in info_texts),
            "Analyst Feedback is honestly reported as not-yet-recorded, not a guessed number",
        )


# =======================================================================
# 4. The real anomaly-detection result computed on Case Analysis is the
#    exact same result Review & Insights shows (one shared, real pipeline
#    result per session - never two different or fabricated numbers).
# =======================================================================
def test_anomaly_results_consistent_across_areas():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        check(len(at.exception) == 0, "Case Analysis runs the real anomaly-detection pipeline with no exception")

        _, real_anomaly_result, _, _ = _pipeline()

        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights re-renders with no exception, reusing the cached pipeline result")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Evaluation / Performance").run(timeout=180)
        check(len(at.exception) == 0, "Evaluation / Performance section renders with no exception")

        predicted_anomalous_metrics = [m for m in at.get("metric") if m.label == "Predicted Anomalous"]
        check(bool(predicted_anomalous_metrics), "Evaluation / Performance shows a real Predicted Anomalous count")
        check(
            str(predicted_anomalous_metrics[0].value) == str(real_anomaly_result.anomalous_events),
            "the count matches the real, freshly-computed AnomalyDetectionResult - not a placeholder or stale value",
        )


# =======================================================================
# 5. The selected case's ID stays identifiable and consistent between
#    Case Analysis and the Review & Insights tabs (via the shared
#    last-selected-case session state).
# =======================================================================
def test_incident_id_shown_consistently():
    from streamlit.testing.v1 import AppTest

    # Two independent AppTest instances (each doing a single Review &
    # Insights section switch) rather than switching sections twice on one
    # instance - streamlit.testing.v1.AppTest does not reliably reconcile a
    # second, unrelated section's now-stale widget state (e.g. Case
    # History's own filters) once a page has swapped which single section
    # it mounts, which is a test-harness quirk, not an application bug.
    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        check(len(at.exception) == 0, "Case Analysis renders with no exception")

        case_selectors = [sb for sb in at.selectbox if sb.label == "Security case"]
        check(bool(case_selectors), "Case Analysis has a working case selector")
        selected_incident_id = case_selectors[0].value
        check(bool(selected_incident_id), "a case is selected by default")

        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights renders with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]

        ri_section.set_value("Explainability").run(timeout=180)
        check(len(at.exception) == 0, "the Explainability section renders with no exception")
        explain_selectors = [sb for sb in at.selectbox if sb.key == "explain_case_select"]
        check(bool(explain_selectors), "the Explainability section has a case selector")
        case_id_metrics = [m for m in at.get("metric") if m.label == "Case ID"]
        check(bool(case_id_metrics), "the Explainability section shows the case's ID as a 'Case ID' metric")
        check(
            case_id_metrics[0].value == selected_incident_id,
            "the case identified on Case Analysis stays the default-selected case in Explainability",
        )

        at2 = AppTest.from_file("app.py")
        at2.run(timeout=180)
        case_selectors2 = [sb for sb in at2.selectbox if sb.label == "Security case"]
        selected_incident_id_2 = case_selectors2[0].value
        check(
            selected_incident_id_2 == selected_incident_id,
            "the same case is deterministically selected by default across fresh sessions",
        )

        at2.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        ri_section2 = [r for r in at2.radio if r.key == "ri_section"][0]
        ri_section2.set_value("Audit Trail").run(timeout=180)
        check(len(at2.exception) == 0, "the Audit Trail section renders with no exception")
        audit_selectors = [sb for sb in at2.selectbox if sb.key == "audit_case_select"]
        check(bool(audit_selectors), "the Audit Trail section has a case selector")
        check(
            build_incident_id(audit_selectors[0].value) == selected_incident_id_2,
            "the Audit Trail section also defaults to the same case reviewed on Case Analysis",
        )


# =======================================================================
# 6. The AI model name and its real (never fabricated) availability
#    status are shown on Case Analysis.
# =======================================================================
def test_genai_model_status_shown_on_case_analysis():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        check(len(at.exception) == 0, "Case Analysis renders with no exception")

        expander_labels = [e.label for e in at.expander]
        check(
            any("AI model settings" in (lbl or "") for lbl in expander_labels),
            "Case Analysis shows an AI model settings section",
        )

        model_inputs = [t for t in at.text_input if "Local AI model name" in (t.label or "")]
        check(bool(model_inputs), "the configured AI model name is shown as a real, editable field")
        check(model_inputs[0].value == OLLAMA_MODEL_NAME, "the displayed model matches config.OLLAMA_MODEL_NAME (qwen3:1.7b)")

        real_status = get_ollama_status(OLLAMA_MODEL_NAME)
        captions = [c.value for c in at.caption]
        check(
            any(real_status.message in (c or "") for c in captions),
            "the real, live Ollama/model availability status is shown, not fabricated",
        )


# =======================================================================
# 7. The knowledge-base section the AI used renders on Case Analysis
# =======================================================================
def test_knowledge_base_section_renders():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        check(len(at.exception) == 0, "Case Analysis renders with no exception")

        subheaders = [s.value for s in at.subheader]
        check(
            "Knowledge Base Used by AI" in subheaders,
            "Case Analysis shows the real retrieved-knowledge section",
        )


# =======================================================================
# 8. The full Case Analysis decision workflow renders (case details,
#    why-flagged, knowledge base, next steps, decision UI)
# =======================================================================
def test_case_analysis_decision_workflow_renders():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        check(len(at.exception) == 0, "Case Analysis renders with no exception")

        subheaders = [s.value for s in at.subheader]
        for expected in [
            "Select Security Case", "AI Case Summary", "Why Was This Flagged?",
            "Knowledge Base Used by AI", "Recommended Next Steps", "Analyst Decision",
        ]:
            check(expected in subheaders, f"Case Analysis shows the '{expected}' section")

        check(
            any("final decision" in (i.value or "").lower() for i in at.info),
            "Case Analysis states the analyst's decision is final",
        )
        check(len(at.radio) >= 1, "the APPROVE / REJECT / MODIFY decision control is present")
        check(
            any("Submit Decision" in (b.label or "") for b in at.button),
            "the Submit Decision button is present",
        )


# =======================================================================
# 9-10. Feedback + Continuous Improvement render, clearly separated tabs
# =======================================================================
def test_feedback_history_and_continuous_improvement_render():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        # Seed one real, valid feedback record so the page has content to
        # render beyond its empty-state message (reuses the real Phase 1-6
        # pipeline + Phase 6's own plan/validate functions - never invents
        # a record by hand).
        _, anomaly_result, rule_result, confidence_result = _pipeline()
        idx, inf, conf = _first_anomalous_row(confidence_result)
        incident_id = build_incident_id(idx)
        plan = generate_response_plan(inf, conf)
        record = FeedbackRecord(
            feedback_id=generate_feedback_id(incident_id),
            incident_id=incident_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            incident_type=inf.incident_type,
            severity=inf.severity,
            confidence_level=conf.confidence_level,
            confidence_score=conf.confidence_score,
            human_review_recommended=conf.human_review_recommended,
            analyst_decision="APPROVE",
            review_status="APPROVED",
            original_plan=plan_to_dict(plan),
            modified_plan=None,
            analyst_notes="ui polish smoke test",
            triggered_rule_ids=[tr.rule_id for tr in inf.triggered_rules],
            supporting_indicators=list(conf.supporting_indicators),
            ai_model_name=None,
            ai_summary_source=None,
        )
        success, errors = save_feedback(record)
        check(success, f"seed feedback record saves cleanly ({errors})")

        # Two independent AppTest instances (see test_incident_id_shown_
        # consistently for why) - both read the same seeded feedback record
        # from the same isolated database.
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights renders with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]

        ri_section.set_value("Feedback").run(timeout=180)
        check(len(at.exception) == 0, "the Feedback section renders with no exception")
        subheaders = [s.value for s in at.subheader]
        check("Feedback" in subheaders, "a 'Feedback' section is present")

        at2 = AppTest.from_file("app.py")
        at2.run(timeout=180)
        at2.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        ri_section2 = [r for r in at2.radio if r.key == "ri_section"][0]
        ri_section2.set_value("Continuous Improvement").run(timeout=180)
        check(len(at2.exception) == 0, "the Continuous Improvement section renders with no exception")
        subheaders = [s.value for s in at2.subheader]
        check(
            "Continuous Improvement" in subheaders,
            "a 'Continuous Improvement' section is present, clearly separate from feedback",
        )
        check(
            any("do not automatically modify the system" in (w.value or "").lower() for w in at2.warning),
            "the required continuous-improvement disclaimer is shown verbatim",
        )


# =======================================================================
# 11. Explainability tab renders (the full evidence-chain framing)
# =======================================================================
def test_explainability_page_renders():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights renders with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Explainability").run(timeout=180)
        check(len(at.exception) == 0, "the Explainability section renders with no exception")

        subheaders = [s.value for s in at.subheader]
        check("Explainability" in subheaders, "the Explainability section is present")

        headings = [m.value for m in at.markdown]
        for expected in [
            "#### Incident Overview", "#### Anomaly Detection", "#### Rule / Knowledge-Based Explanation",
            "#### Severity + Confidence Explanation", "#### RAG Sources", "#### Generative AI Explanation",
            "#### Response Plan Explanation", "#### Human Analyst Review",
        ]:
            check(expected in headings, f"Explainability shows the '{expected}' section")


# =======================================================================
# 12. Audit timeline renders (with real recorded events)
# =======================================================================
def test_audit_timeline_renders_with_events():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        # Case Analysis is the default landing page. Generating a real AI
        # case summary there writes real audit events for the selected
        # incident (on top of the pipeline-stage events every render
        # already logs).
        gen_buttons = [b for b in at.button if "Generate AI Case Summary" in (b.label or "")]
        check(bool(gen_buttons), "the Generate AI Case Summary button is present")
        gen_buttons[0].click().run(timeout=180)
        check(len(at.exception) == 0, "generating a summary raises no exception")

        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights renders with no exception after real activity")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Audit Trail").run(timeout=180)
        check(len(at.exception) == 0, "the Audit Trail section renders with no exception")
        subheaders = [s.value for s in at.subheader]
        check("Audit Trail" in subheaders, "the Audit Trail section is present")
        captions = [c.value for c in at.caption]
        check(
            any("event(s) recorded" in c for c in captions),
            "the Audit Timeline reports real recorded event(s), not an empty placeholder",
        )


# =======================================================================
# 13. Evaluation / Performance tab renders (grouped sections intact)
# =======================================================================
def test_evaluation_page_renders():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights renders with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Evaluation / Performance").run(timeout=180)
        check(len(at.exception) == 0, "the Evaluation / Performance section renders with no exception")

        subheaders = [s.value for s in at.subheader]
        check("Evaluation / Performance" in subheaders, "the Evaluation / Performance section is present")

        headings = [m.value for m in at.markdown]
        expected = [
            "#### A. Dataset Overview", "#### B. Anomaly Detection Performance", "#### C. Incident Analysis",
            "#### D. Confidence & Human Review", "#### E. RAG Performance", "#### F. Generative AI Performance",
            "#### G. Analyst Feedback", "#### H. Audit Activity", "#### Local Performance Timing",
        ]
        for section in expected:
            check(section in headings, f"Evaluation section '{section}' still renders")
        check(
            any("Offline Evaluation" in (c.value or "") for c in at.caption)
            or any("Offline Evaluation" in (w.value or "") for w in at.warning),
            "ground-truth metrics are still clearly labeled 'Offline Evaluation'",
        )


# =======================================================================
# 14. Empty / error states never raise, and never surface a raw traceback
#     to the user for an ordinary "nothing here yet" situation
# =======================================================================
def test_empty_states_do_not_raise():
    from streamlit.testing.v1 import AppTest

    with isolated_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=180)

        # No AI case summary generated yet on Case Analysis.
        check(len(at.exception) == 0, "Case Analysis with nothing generated yet raises no exception")
        info_texts = [i.value for i in at.info]
        check(
            any("generate ai case summary" in (t or "").lower() for t in info_texts),
            "Case Analysis tells the analyst to click Generate, rather than showing nothing or an error",
        )

        # No feedback recorded yet.
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        check(len(at.exception) == 0, "Review & Insights with no records raises no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Feedback").run(timeout=180)
        check(len(at.exception) == 0, "the Feedback section with no records raises no exception")
        info_texts = [i.value for i in at.info]
        check(
            any("no feedback has been recorded yet" in (t or "").lower() for t in info_texts),
            "Feedback shows a clear, friendly empty-state message (no traceback)",
        )

        # No audit events recorded yet for a fresh incident, on the
        # Evaluation section's Audit Activity area (a fresh instance - see
        # test_incident_id_shown_consistently for why - covers this, and
        # test_review_insights_shows_real_system_status already exercises
        # this exact empty-audit-trail path end to end).
        at2 = AppTest.from_file("app.py")
        at2.run(timeout=180)
        at2.sidebar.radio[0].set_value("Review & Insights").run(timeout=180)
        ri_section2 = [r for r in at2.radio if r.key == "ri_section"][0]
        ri_section2.set_value("Evaluation / Performance").run(timeout=180)
        check(len(at2.exception) == 0, "the Evaluation / Performance section with an empty audit trail raises no exception")


# =======================================================================
# 15. Dataset checksum is unchanged by the UI work
# =======================================================================
def test_dataset_unchanged():
    with open(DATASET_PATH, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    check(digest == EXPECTED_DATASET_CHECKSUM, f"dataset checksum unchanged ({digest})")


# =======================================================================
# 16-17. Qwen3:1.7B + think=False remain configured
# =======================================================================
def test_qwen_model_still_configured():
    check(OLLAMA_MODEL_NAME == "qwen3:1.7b", "config.OLLAMA_MODEL_NAME is still 'qwen3:1.7b'")


def test_think_false_still_configured():
    with open(os.path.join("src", "genai_analyzer.py"), "r", encoding="utf-8") as f:
        source = f.read()
    check("think=False" in source, "src/genai_analyzer.py still passes think=False to ollama.chat()")
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = f.read()
    check("ollama==0.6.2" in requirements, "requirements.txt still pins ollama==0.6.2")


# =======================================================================
# 18. Phase 9/10/11 outputs remain accessible (structural regression guard)
# =======================================================================
def test_phase_9_10_11_outputs_remain_accessible():
    from src.feedback import load_feedback

    _, anomaly_result, rule_result, confidence_result = _pipeline()
    idx, inf, conf = _first_anomalous_row(confidence_result)
    incident_id = build_incident_id(idx)

    with isolated_databases():
        # Phase 9 - Continuous Improvement (works on whatever feedback
        # exists, including none - never raises). Uses its own isolated
        # database, never the real data/analyst_feedback.db.
        ci_report = generate_continuous_improvement_report(load_feedback())
        check(ci_report is not None, "Phase 9's generate_continuous_improvement_report() still returns a report")

    # Phase 10 - Explainability (pure explanation of already-computed
    # facts, never recomputes anything).
    explanation = build_incident_explanation(incident_id, inf, conf)
    check(explanation is not None, "Phase 10's build_incident_explanation() still returns a result")
    check(explanation.anomaly.explanation, "Phase 10 explanation still has real anomaly-detection content")

    # Phase 11 - Evaluation (reuses the same whole-dataset pipeline every
    # page already computed above).
    report = build_evaluation_report(anomaly_result, rule_result, confidence_result)
    check(report is not None, "Phase 11's build_evaluation_report() still returns a report")
    check(report.dataset_overview.total_events == anomaly_result.total_events, "Phase 11 evaluation still matches the real pipeline")


# =======================================================================
# Runner
# =======================================================================
def main():
    tests = [
        test_all_pages_load,
        test_page_titles_exist,
        test_review_insights_shows_real_system_status,
        test_anomaly_results_consistent_across_areas,
        test_incident_id_shown_consistently,
        test_genai_model_status_shown_on_case_analysis,
        test_knowledge_base_section_renders,
        test_case_analysis_decision_workflow_renders,
        test_feedback_history_and_continuous_improvement_render,
        test_explainability_page_renders,
        test_audit_timeline_renders_with_events,
        test_evaluation_page_renders,
        test_empty_states_do_not_raise,
        test_dataset_unchanged,
        test_qwen_model_still_configured,
        test_think_false_still_configured,
        test_phase_9_10_11_outputs_remain_accessible,
    ]
    try:
        for i, test in enumerate(tests, 1):
            test()
            print(f"{i}. {test.__doc__.strip().splitlines()[0] if test.__doc__ else test.__name__}")
        print()
        print("=" * 70)
        print(f"UI POLISH CHECKS PASSED ({PASSED} checks)")
        print("=" * 70)
    finally:
        cleanup_temp_dirs()


if __name__ == "__main__":
    main()
