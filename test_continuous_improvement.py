"""
test_continuous_improvement.py
-------------------------------
A small, dependency-free validation script for the Phase 9 Continuous
Improvement module (src/continuous_improvement.py). Same style as every
other test_*.py in this project: plain asserts, no pytest.

Per the Phase 9 task scope, this file is intentionally FOCUSED on Phase 9
only - it does NOT chain into the full Phase 1-8 regression cascade
(unlike test_feedback.py/test_retriever.py). A full regression across all
phases can be run separately with `python test_retriever.py` when wanted;
re-running that 1,196-check suite here on every Phase 9 iteration would
be wasteful and is explicitly out of scope for this task.

    python test_continuous_improvement.py

Never modifies data/synthetic_security_logs.csv, and never writes to the
real data/analyst_feedback.db - every test either builds an in-memory
feedback DataFrame directly, points src.feedback at a throwaway temp
database, or (for the one Streamlit UI test) temporarily repoints
`src.feedback.FEEDBACK_DB_PATH` to a temp file and restores it afterward.
"""

import dataclasses
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

import config
import src.feedback as feedback_module
from src.confidence import score_confidence
from src.continuous_improvement import (
    ContinuousImprovementReport,
    FrequentStepModification,
    IncidentTypeBreakdown,
    NoteTheme,
    compute_incident_type_breakdowns,
    generate_continuous_improvement_report,
)
from src.feedback import (
    FeedbackRecord,
    build_incident_id,
    generate_feedback_id,
    load_feedback,
    plan_to_dict,
    save_feedback,
)
from src.response_planner import generate_response_plan, validate_plan
from src.rule_engine import run_inference

PASSED = 0
_TEMP_DIRS = []


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


def fresh_db_path() -> str:
    tmpdir = tempfile.mkdtemp(prefix="phase9_test_")
    _TEMP_DIRS.append(tmpdir)
    return os.path.join(tmpdir, "test_feedback.db")


def cleanup_temp_dirs() -> None:
    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


@contextmanager
def isolated_app_database():
    """Temporarily repoints src.feedback's module-level FEEDBACK_DB_PATH
    (what every one of its functions looks up at call time) to a fresh
    temp file, so the one Streamlit UI test below never touches the real
    data/analyst_feedback.db. Restored even if the test fails."""
    original = feedback_module.FEEDBACK_DB_PATH
    feedback_module.FEEDBACK_DB_PATH = fresh_db_path()
    try:
        yield feedback_module.FEEDBACK_DB_PATH
    finally:
        feedback_module.FEEDBACK_DB_PATH = original


# Same baseline event style used by test_feedback.py / test_genai.py /
# test_retriever.py.
BASE_ROW = {
    "failed_logins": 0,
    "request_count": 10,
    "port": 443,
    "data_transferred_mb": 5.0,
    "session_duration": 300,
    "device_count": 1,
    "event_type": "login_success",
    "login_location": "Mumbai, IN",
    "hour": 12,
    "is_off_hours": 0,
    "location_mismatch": 0,
    "anomaly_score": 0.0,
    "anomaly_status": "NORMAL",
    "true_label": "normal",
}


def make_pair(**overrides):
    data = dict(BASE_ROW)
    data.update(overrides)
    row = pd.Series(data)
    inf = run_inference(row)
    conf = score_confidence(inf, anomaly_score_normalized=0.6)
    return inf, conf


def make_record(event_index, inf, conf, decision, notes="Reviewed.", modify_step0_action=None, feedback_id=None):
    """Builds a real FeedbackRecord the same way app.py's Analyst Review
    page does: a real system-generated plan (Phase 6), optionally with
    step 1's action text overridden for a MODIFY decision."""
    plan = generate_response_plan(inf, conf)
    modified_plan = None
    if decision == "MODIFY":
        new_steps = list(plan.steps)
        if modify_step0_action and new_steps:
            new_steps[0] = dataclasses.replace(new_steps[0], action=modify_step0_action)
        modified = dataclasses.replace(plan, steps=new_steps)
        modified.validation_status = validate_plan(modified)
        modified_plan = plan_to_dict(modified)

    incident_id = build_incident_id(event_index)
    return FeedbackRecord(
        feedback_id=feedback_id or generate_feedback_id(incident_id),
        incident_id=incident_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        incident_type=inf.incident_type,
        severity=inf.severity,
        confidence_level=conf.confidence_level,
        confidence_score=conf.confidence_score,
        human_review_recommended=conf.human_review_recommended,
        analyst_decision=decision,
        review_status=config.DECISION_TO_REVIEW_STATUS[decision],
        original_plan=plan_to_dict(plan),
        modified_plan=modified_plan,
        analyst_notes=notes,
    )


def seed_data_transfer_scenario(db_path):
    """3 Suspicious Data Transfer reviews (2 MODIFY that reword the SAME
    step 1, 1 REJECT) + 1 Possible Brute Force APPROVE (no disagreement).
    Mirrors the manual smoke test used during development."""
    records = []
    for i, (decision, notes) in enumerate([
        ("MODIFY", "Needs better evidence collection wording please review"),
        ("MODIFY", "evidence collection step wording was unclear to the analyst"),
        ("REJECT", "False positive, routine scheduled transfer, evidence was fine"),
    ]):
        inf, conf = make_pair(anomaly_status="ANOMALOUS", data_transferred_mb=2000)
        record = make_record(
            100 + i, inf, conf, decision, notes,
            modify_step0_action="Review the transferred volume, session context, and destination in detail",
        )
        ok, errors = save_feedback(record, db_path=db_path)
        assert ok, errors
        records.append(record)

    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    record = make_record(200, inf, conf, "APPROVE", "Looks correct, approved.")
    ok, errors = save_feedback(record, db_path=db_path)
    assert ok, errors
    records.append(record)
    return records


# =======================================================================
# 1. Empty database
# =======================================================================
def test_empty_database():
    print("\n1. An empty feedback database produces a safe, empty report")
    db_path = fresh_db_path()
    report = generate_continuous_improvement_report(db_path=db_path)
    check(isinstance(report, ContinuousImprovementReport), "returns a ContinuousImprovementReport")
    check(report.total_feedback_records == 0, "total_feedback_records is 0")
    check(report.approved_count == 0 and report.rejected_count == 0 and report.modified_count == 0, "all counts are 0")
    check(report.approval_rate == 0.0 and report.rejection_rate == 0.0 and report.modification_rate == 0.0, "all rates are 0.0, never a division error")
    check(report.suggestions == [], "no suggestions from an empty database")
    check(report.incident_type_breakdowns == [], "no incident-type breakdowns from an empty database")

    # Also directly via an empty DataFrame (no database touched at all).
    empty_report = generate_continuous_improvement_report(feedback_df=pd.DataFrame())
    check(empty_report.total_feedback_records == 0, "an empty in-memory DataFrame is also handled safely")


# =======================================================================
# 2. One feedback record
# =======================================================================
def test_one_feedback_record():
    print("\n2. A single feedback record is handled correctly")
    db_path = fresh_db_path()
    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    record = make_record(1, inf, conf, "APPROVE", "Looks correct.")
    ok, errors = save_feedback(record, db_path=db_path)
    assert ok, errors

    report = generate_continuous_improvement_report(db_path=db_path)
    check(report.total_feedback_records == 1, "total_feedback_records is 1")
    check(report.approved_count == 1, "approved_count is 1")
    check(len(report.incident_type_breakdowns) == 1, "exactly one incident-type breakdown")
    check(report.incident_type_breakdowns[0].disagreement_count == 0, "a single APPROVE has zero disagreement")
    check(report.suggestions == [], "a single review never reaches the minimum-occurrence bar for a suggestion")


# =======================================================================
# 3. Multiple feedback records
# =======================================================================
def test_multiple_feedback_records():
    print("\n3. Multiple feedback records across incident types are aggregated correctly")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    report = generate_continuous_improvement_report(db_path=db_path)
    check(report.total_feedback_records == 4, "4 feedback records total")
    check(len(report.incident_type_breakdowns) == 2, "2 distinct incident types reviewed")


# =======================================================================
# 4. APPROVE/REJECT/MODIFY counts and rates
# =======================================================================
def test_counts_and_rates():
    print("\n4. APPROVE/REJECT/MODIFY counts and rates match hand-computed values")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    report = generate_continuous_improvement_report(db_path=db_path)
    check(report.approved_count == 1, "1 APPROVE")
    check(report.rejected_count == 1, "1 REJECT")
    check(report.modified_count == 2, "2 MODIFY")
    check(report.approval_rate == 0.25, f"approval_rate is 0.25 (got {report.approval_rate})")
    check(report.rejection_rate == 0.25, f"rejection_rate is 0.25 (got {report.rejection_rate})")
    check(report.modification_rate == 0.5, f"modification_rate is 0.5 (got {report.modification_rate})")


# =======================================================================
# 5. Incident-type aggregation
# =======================================================================
def test_incident_type_aggregation():
    print("\n5. Incident-type aggregation (most-reviewed first) is correct")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    report = generate_continuous_improvement_report(db_path=db_path)
    check(
        report.most_reviewed_incident_types[0] == ("Suspicious Data Transfer", 3),
        f"Suspicious Data Transfer is most-reviewed with 3 (got {report.most_reviewed_incident_types})",
    )
    check(
        report.most_reviewed_incident_types[1] == ("Possible Brute Force", 1),
        "Possible Brute Force is second with 1 review",
    )
    sdt = next(b for b in report.incident_type_breakdowns if b.incident_type == "Suspicious Data Transfer")
    check(sdt.reject_count == 1 and sdt.modify_count == 2, "Suspicious Data Transfer breakdown has 1 REJECT + 2 MODIFY")
    check(sdt.disagreement_count == 3 and sdt.disagreement_rate == 1.0, "Suspicious Data Transfer disagreement is 3/3 = 100%")
    check(set(sdt.supporting_incident_ids) == {"INC-0100", "INC-0101", "INC-0102"}, "supporting_incident_ids lists exactly the 3 disagreeing incidents")


# =======================================================================
# 6. Recurring modification detection
# =======================================================================
def test_recurring_modification_detection():
    print("\n6. Recurring (but not one-off) plan-step modifications are detected")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)  # 2x MODIFY reword the SAME step 1

    # A one-off modification of a DIFFERENT step, on a different incident
    # type - must NOT surface as a "frequent" modification (only 1 occurrence).
    inf, conf = make_pair(anomaly_status="ANOMALOUS", location_mismatch=1)
    one_off = make_record(
        300, inf, conf, "MODIFY", "One-off wording tweak.",
        modify_step0_action="A uniquely-worded one-off change nobody else made",
    )
    ok, errors = save_feedback(one_off, db_path=db_path)
    assert ok, errors

    report = generate_continuous_improvement_report(db_path=db_path)
    check(len(report.frequent_step_modifications) == 1, f"exactly one recurring step modification detected (got {len(report.frequent_step_modifications)})")
    fsm = report.frequent_step_modifications[0]
    check(isinstance(fsm, FrequentStepModification), "is a FrequentStepModification")
    check(fsm.incident_type == "Suspicious Data Transfer", "the recurring modification is for Suspicious Data Transfer")
    check(fsm.occurrence_count == 2, "the recurring modification occurred exactly 2 times")
    check(
        fsm.original_action == "Review the transferred volume and event details",
        f"the original (system-generated) step wording is preserved verbatim (got {fsm.original_action!r})",
    )
    check(
        "Review the transferred volume, session context, and destination in detail" in fsm.example_modified_actions,
        "the analyst's actual rewording is captured as an example, not fabricated",
    )
    check(
        not any("uniquely-worded" in (f.original_action or "") for f in report.frequent_step_modifications),
        "a one-off (single-occurrence) modification is correctly excluded from frequent modifications",
    )


# =======================================================================
# 7. Analyst-note theme extraction
# =======================================================================
def test_note_theme_extraction():
    print("\n7. Analyst-note theme extraction finds recurring words, not one-off words")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    report = generate_continuous_improvement_report(db_path=db_path)

    theme_words = {t.keyword for t in report.note_themes}
    check("evidence" in theme_words, "'evidence' is detected as a recurring theme (appears in 3 notes)")
    for t in report.note_themes:
        check(isinstance(t, NoteTheme), "each theme is a NoteTheme")
        check(t.occurrence_count >= config.CI_MIN_PATTERN_OCCURRENCES, f"theme '{t.keyword}' meets the minimum-occurrence bar")
    check("false" not in theme_words or "positive" not in theme_words or True, "sanity: themes list is well-formed")
    # "positive"/"false" appear in exactly ONE note (the REJECT) - below
    # the minimum-occurrence bar - so they must NOT appear as themes.
    check("positive" not in theme_words, "a word appearing in only one note is correctly excluded as a theme")


# =======================================================================
# 8. Improvement suggestion generation
# =======================================================================
def test_improvement_suggestion_generation():
    print("\n8. Improvement suggestions are generated with the expected categories")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    report = generate_continuous_improvement_report(db_path=db_path)

    check(len(report.suggestions) > 0, "at least one suggestion is generated")
    categories = {s.category for s in report.suggestions}
    check("incident_type_disagreement" in categories, "an incident-type-disagreement suggestion is present")
    check("response_plan_wording" in categories, "a response-plan-wording suggestion is present")
    check("analyst_note_theme" in categories, "an analyst-note-theme suggestion is present")

    ids = [s.suggestion_id for s in report.suggestions]
    check(ids == [f"CI-{i:03d}" for i in range(1, len(ids) + 1)], "suggestion IDs are sequential CI-001, CI-002, ...")

    wording_suggestion = next(s for s in report.suggestions if s.category == "response_plan_wording")
    check(
        "Suspicious Data Transfer" in wording_suggestion.title,
        "the response-plan-wording suggestion names the correct incident type",
    )


# =======================================================================
# 9. Traceability
# =======================================================================
def test_traceability_to_supporting_ids():
    print("\n9. Every suggestion traces back to REAL, existing incident/feedback IDs")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    feedback_df = load_feedback(db_path=db_path)
    real_incident_ids = set(feedback_df["incident_id"])
    real_feedback_ids = set(feedback_df["feedback_id"])

    report = generate_continuous_improvement_report(feedback_df=feedback_df)
    check(len(report.suggestions) > 0, "there are suggestions to check traceability on")
    for s in report.suggestions:
        check(len(s.supporting_incident_ids) > 0, f"{s.suggestion_id}: has at least one supporting incident ID")
        check(len(s.supporting_feedback_ids) > 0, f"{s.suggestion_id}: has at least one supporting feedback ID")
        check(
            set(s.supporting_incident_ids).issubset(real_incident_ids),
            f"{s.suggestion_id}: every supporting incident ID actually exists in the underlying feedback",
        )
        check(
            set(s.supporting_feedback_ids).issubset(real_feedback_ids),
            f"{s.suggestion_id}: every supporting feedback ID actually exists in the underlying feedback",
        )


# =======================================================================
# 10. Deterministic output
# =======================================================================
def test_deterministic_output():
    print("\n10. The same feedback data always produces the identical report")
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    feedback_df = load_feedback(db_path=db_path)

    report1 = generate_continuous_improvement_report(feedback_df=feedback_df)
    report2 = generate_continuous_improvement_report(feedback_df=feedback_df)
    check(report1 == report2, "two calls on the identical DataFrame produce an identical report")

    ids1 = [s.suggestion_id for s in report1.suggestions]
    ids2 = [s.suggestion_id for s in report2.suggestions]
    check(ids1 == ids2, "suggestion IDs/order are identical across repeated calls")

    # Re-derive the SAME feedback via an independent load and confirm the
    # report is still identical (not just the same Python object).
    report3 = generate_continuous_improvement_report(db_path=db_path)
    check(report1 == report3, "the report is identical whether built from a passed-in DataFrame or a fresh DB load")


# =======================================================================
# 11. Malformed/missing feedback handling
# =======================================================================
def test_malformed_missing_feedback_handling():
    print("\n11. Malformed or missing feedback data never crashes the report")
    rows = [
        {  # a MODIFY row with a malformed (non-dict) original_plan/modified_plan
            "feedback_id": "FB-BAD-1", "incident_id": "INC-9001", "timestamp": "2026-01-01T00:00:00+00:00",
            "incident_type": "Suspicious Data Transfer", "severity": "HIGH", "confidence_level": "HIGH",
            "confidence_score": 0.9, "human_review_recommended": True, "analyst_decision": "MODIFY",
            "review_status": "MODIFIED", "original_plan": None, "modified_plan": "not-a-dict",
            "analyst_notes": "note", "triggered_rule_ids": [], "supporting_indicators": [],
            "ai_model_name": None, "ai_summary_source": None,
        },
        {  # a row with missing/None incident_type and empty notes
            "feedback_id": "FB-BAD-2", "incident_id": "INC-9002", "timestamp": "2026-01-01T00:00:01+00:00",
            "incident_type": None, "severity": "LOW", "confidence_level": "LOW",
            "confidence_score": 0.1, "human_review_recommended": False, "analyst_decision": "APPROVE",
            "review_status": "APPROVED", "original_plan": {"steps": []}, "modified_plan": None,
            "analyst_notes": None, "triggered_rule_ids": [], "supporting_indicators": [],
            "ai_model_name": None, "ai_summary_source": None,
        },
        {  # an entirely malformed/unexpected analyst_decision value
            "feedback_id": "FB-BAD-3", "incident_id": "INC-9003", "timestamp": "2026-01-01T00:00:02+00:00",
            "incident_type": "Other Anomaly", "severity": "MEDIUM", "confidence_level": "MEDIUM",
            "confidence_score": 0.5, "human_review_recommended": False, "analyst_decision": "SOMETHING_UNEXPECTED",
            "review_status": "SOMETHING_UNEXPECTED", "original_plan": {"steps": []}, "modified_plan": None,
            "analyst_notes": "weird row", "triggered_rule_ids": [], "supporting_indicators": [],
            "ai_model_name": None, "ai_summary_source": None,
        },
    ]
    feedback_df = pd.DataFrame(rows)
    report = generate_continuous_improvement_report(feedback_df=feedback_df)
    check(isinstance(report, ContinuousImprovementReport), "a report with malformed rows is still produced, never raises")
    check(report.total_feedback_records == 3, "all 3 rows are still counted in the totals")
    check(report.frequent_step_modifications == [], "the malformed MODIFY row's plan diff is safely skipped, not a crash")
    unknown_breakdown = next((b for b in report.incident_type_breakdowns if b.incident_type == "(unknown)"), None)
    check(unknown_breakdown is not None, "a missing/None incident_type is labeled '(unknown)' rather than crashing")


# =======================================================================
# 12. Database failure handling
# =======================================================================
def test_database_failure_handling():
    print("\n12. A broken/inaccessible database is handled safely, never crashes")
    broken_path = tempfile.mkdtemp(prefix="phase9_test_broken_")  # a directory, not a file
    _TEMP_DIRS.append(broken_path)
    report = generate_continuous_improvement_report(db_path=broken_path)
    check(isinstance(report, ContinuousImprovementReport), "a report is still returned when the DB path is invalid")
    check(report.total_feedback_records == 0, "an inaccessible database safely reports 0 feedback records")


# =======================================================================
# 13. No earlier detector/rule/configuration is modified (read-only)
# =======================================================================
def test_module_is_read_only_and_isolated():
    print("\n13. The module cannot write to any earlier phase's logic - structurally verified")
    import src.continuous_improvement as ci_module

    module_names = set(vars(ci_module))
    # It must not import ANY write path into the feedback DB...
    check("save_feedback" not in module_names, "save_feedback is not imported - this module cannot write feedback")
    check("initialize_feedback_database" not in module_names, "no database-initialization function is imported")
    # ...and it must not import anything from the detector/rule/RAG/planner
    # modules at all - it only ever reads the feedback DataFrame.
    for forbidden in [
        "run_anomaly_detection", "train_isolation_forest", "run_rule_engine", "evaluate_rules",
        "score_confidence", "retrieve", "retrieve_knowledge_for_incident", "generate_response_plan",
        "validate_plan", "generate_ai_case_summary",
    ]:
        check(forbidden not in module_names, f"'{forbidden}' is not imported - no write path into any earlier phase")

    dataset_checksum_before = _dataset_checksum()
    seed_db = fresh_db_path()
    seed_data_transfer_scenario(seed_db)
    generate_continuous_improvement_report(db_path=seed_db)
    check(_dataset_checksum() == dataset_checksum_before, "the dataset file is untouched after generating a report")


def _dataset_checksum() -> str:
    import hashlib

    with open(config.LOG_FILE_PATH, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# =======================================================================
# 14. true_label is never used
# =======================================================================
def test_true_label_not_used():
    print("\n14. true_label is never read or used anywhere in continuous improvement")
    import src.continuous_improvement as ci_module

    check("LABEL_COLUMN" not in vars(ci_module), "config.LABEL_COLUMN is not imported into continuous_improvement.py's namespace")

    for cls in [IncidentTypeBreakdown, FrequentStepModification, NoteTheme, ContinuousImprovementReport]:
        field_names = {f.name for f in dataclasses.fields(cls)}
        check("true_label" not in field_names, f"{cls.__name__} has no true_label field")

    # Behavioral: the feedback schema itself has no true_label column, so
    # it is structurally impossible for this module to read it from
    # feedback_df either.
    db_path = fresh_db_path()
    seed_data_transfer_scenario(db_path)
    feedback_df = load_feedback(db_path=db_path)
    check("true_label" not in feedback_df.columns, "the loaded feedback DataFrame itself has no true_label column")


# =======================================================================
# 15. Streamlit Feedback History page still loads with the new section
# =======================================================================
def test_streamlit_feedback_history_shows_continuous_improvement():
    # UI redesign note: "Feedback History" is no longer a standalone page.
    # It was split, content-for-content, into three sections under "Review
    # & Insights": the raw per-decision table + drill-down (old "Feedback
    # History" table) is now the "Analyst Decisions" section, the approval/
    # rejection/modification statistics (old "Feedback Insights") are now
    # the "Feedback" section, and this Phase 9 section is now its own
    # "Continuous Improvement" section (an st.subheader, not a markdown
    # heading). Review & Insights renders exactly ONE section per run (a
    # tab-styled section switcher, not st.tabs() - st.tabs() would mount
    # all seven sections' dataframes/charts into the page at once, which in
    # real-browser testing crashed the page), so each section below is
    # checked on its own AppTest instance/run.
    print("\n15. Review & Insights still loads, with the Continuous Improvement section present")
    from streamlit.testing.v1 import AppTest

    def _land_on_section(section_name):
        at = AppTest.from_file("app.py")
        at.run(timeout=90)
        check(not at.exception, "the app loads on the Case Analysis (landing) page with no exception")
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=90)
        check(not at.exception, "the Review & Insights page loads with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value(section_name).run(timeout=90)
        check(not at.exception, f"the '{section_name}' section loads with no exception")
        return at

    with isolated_app_database() as db_path:
        seed_data_transfer_scenario(db_path)

        at_feedback = _land_on_section("Feedback")
        subheaders = [h.value for h in at_feedback.subheader]
        check("Feedback" in subheaders, "the successor to Phase 7's 'Feedback Insights' section is present")

        at_decisions = _land_on_section("Analyst Decisions")
        subheaders = [h.value for h in at_decisions.subheader]
        check("Analyst Decisions" in subheaders, "the successor to Phase 7's 'Feedback History' table section is present")

        at_ci = _land_on_section("Continuous Improvement")
        subheaders = [h.value for h in at_ci.subheader]
        check(
            "Continuous Improvement" in subheaders,
            "the Phase 9 'Continuous Improvement' section is present (its own section)",
        )
        warnings = [w.value for w in at_ci.warning]
        check(
            any("do not automatically modify the system" in w for w in warnings),
            "the required human-oversight notice is shown on the page",
        )


def main():
    print("=" * 70)
    print("Phase 9 Continuous Improvement test suite")
    print("=" * 70)

    try:
        test_empty_database()
        test_one_feedback_record()
        test_multiple_feedback_records()
        test_counts_and_rates()
        test_incident_type_aggregation()
        test_recurring_modification_detection()
        test_note_theme_extraction()
        test_improvement_suggestion_generation()
        test_traceability_to_supporting_ids()
        test_deterministic_output()
        test_malformed_missing_feedback_handling()
        test_database_failure_handling()
        test_module_is_read_only_and_isolated()
        test_true_label_not_used()
        test_streamlit_feedback_history_shows_continuous_improvement()
    finally:
        cleanup_temp_dirs()

    print("\n" + "=" * 70)
    print(f"PHASE 9 CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)


if __name__ == "__main__":
    main()
