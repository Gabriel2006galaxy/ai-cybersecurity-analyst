"""
test_feedback.py
------------------
A small, dependency-free validation script for the Phase 7 human analyst
review & feedback loop (src/feedback.py). Same style as
test_preprocessing.py, test_anomaly_detection.py, test_rule_engine.py,
test_confidence.py, test_genai.py, and test_response_planner.py: plain
asserts, no pytest.

This file also folds in a full regression run of the Phase 1-6 test
suites (test_response_planner.py already re-runs Phase 1-5 itself), so a
single command verifies the whole pipeline still works after adding
Phase 7:

    python test_feedback.py

Every test that needs a database uses its OWN temporary SQLite file
(never `config.FEEDBACK_DB_PATH`, the application's real database), so
running this suite never pollutes real analyst feedback and every test
starts from a clean, isolated slate.

Never modifies data/synthetic_security_logs.csv.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

import src.feedback as feedback_module
from src.confidence import score_confidence
from src.feedback import (
    FeedbackRecord,
    build_incident_id,
    dict_to_plan,
    generate_feedback_id,
    get_feedback_by_incident,
    get_review_status_map,
    initialize_feedback_database,
    load_feedback,
    plan_to_dict,
    save_feedback,
    summarize_feedback,
    validate_feedback,
)
from src.response_planner import PlanStep, generate_response_plan, validate_plan
from src.rule_engine import run_inference

PASSED = 0
_TEMP_DIRS = []


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


def fresh_db_path() -> str:
    """A brand-new, not-yet-existing SQLite file path in an isolated temp
    directory - every test that touches the database gets its own, so
    tests never interfere with each other or with the real application
    database."""
    tmpdir = tempfile.mkdtemp(prefix="phase7_test_")
    _TEMP_DIRS.append(tmpdir)
    return os.path.join(tmpdir, "test_feedback.db")


def cleanup_temp_dirs() -> None:
    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


@contextmanager
def isolated_app_database():
    """
    The Streamlit UI tests below exercise the REAL app.py, whose pages
    call `save_feedback()`/`load_feedback()` with no explicit `db_path`
    (they default to `config.FEEDBACK_DB_PATH`, the actual application
    database). This context manager temporarily repoints that default -
    by patching the module-level name every one of `src/feedback.py`'s
    functions looks up at call time - to a fresh temp file, so running
    this test suite NEVER creates or pollutes the real
    data/analyst_feedback.db. The original value is always restored,
    even if a test fails.
    """
    original = feedback_module.FEEDBACK_DB_PATH
    feedback_module.FEEDBACK_DB_PATH = fresh_db_path()
    try:
        yield feedback_module.FEEDBACK_DB_PATH
    finally:
        feedback_module.FEEDBACK_DB_PATH = original


BASE_ROW = {
    "failed_logins": 23,
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
    "anomaly_status": "ANOMALOUS",  # Possible Brute Force via failed_logins=23
    "true_label": "suspicious",
}


def make_pair(**overrides):
    import pandas as pd

    data = dict(BASE_ROW)
    data.update(overrides)
    row = pd.Series(data)
    inf = run_inference(row)
    conf = score_confidence(inf, anomaly_score_normalized=0.6)
    return inf, conf


def make_record(incident_id, inf, conf, plan, decision, modified_plan=None, notes="Reviewed.", feedback_id=None):
    from config import DECISION_TO_REVIEW_STATUS

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
        review_status=DECISION_TO_REVIEW_STATUS[decision],
        original_plan=plan_to_dict(plan),
        modified_plan=plan_to_dict(modified_plan) if modified_plan is not None else None,
        analyst_notes=notes,
        triggered_rule_ids=[tr.rule_id for tr in inf.triggered_rules],
        supporting_indicators=list(conf.supporting_indicators),
    )


# =======================================================================
# 1-2. Database / table initialization
# =======================================================================
def test_database_initializes_successfully():
    print("\n1. Database initializes successfully")
    db_path = fresh_db_path()
    check(not os.path.exists(db_path), "sanity: the database file does not exist yet")
    ok = initialize_feedback_database(db_path)
    check(ok is True, "initialize_feedback_database() reports success")
    check(os.path.exists(db_path), "the database file was actually created on disk")


def test_feedback_table_created_automatically():
    print("\n2. The feedback table is created automatically, even without calling initialize first")
    db_path = fresh_db_path()
    # Deliberately skip initialize_feedback_database() - load_feedback()
    # itself must create the table on demand (Requirement #7).
    df = load_feedback(db_path)
    check(df.empty, "a brand-new database has no feedback rows yet")
    check(os.path.exists(db_path), "the database file was created as a side effect of the first call")

    import sqlite3

    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    check("feedback" in tables, "the 'feedback' table exists in the database")


# =======================================================================
# 3-6. APPROVE / REJECT / MODIFY saving
# =======================================================================
def test_approve_decision_saves_correctly():
    print("\n3. An APPROVE decision saves correctly")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(1)
    plan = generate_response_plan(inf, conf)
    record = make_record(incident_id, inf, conf, plan, "APPROVE")
    ok, errors = save_feedback(record, db_path)
    check(ok, f"APPROVE saves without error (got errors: {errors})")

    loaded = load_feedback(db_path)
    check(len(loaded) == 1, "exactly one feedback row exists after saving")
    check(loaded.iloc[0]["analyst_decision"] == "APPROVE", "the saved decision is APPROVE")
    check(loaded.iloc[0]["review_status"] == "APPROVED", "the saved review_status is APPROVED")
    check(loaded.iloc[0]["modified_plan"] is None, "an APPROVE record has no modified_plan")


def test_reject_decision_saves_correctly():
    print("\n4. A REJECT decision saves correctly")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(2)
    plan = generate_response_plan(inf, conf)
    record = make_record(incident_id, inf, conf, plan, "REJECT", notes="Does not look malicious.")
    ok, errors = save_feedback(record, db_path)
    check(ok, f"REJECT saves without error (got errors: {errors})")

    loaded = load_feedback(db_path)
    check(loaded.iloc[0]["analyst_decision"] == "REJECT", "the saved decision is REJECT")
    check(loaded.iloc[0]["review_status"] == "REJECTED", "the saved review_status is REJECTED")
    check(loaded.iloc[0]["analyst_notes"] == "Does not look malicious.", "the analyst notes were saved correctly")


def test_modify_decision_saves_correctly():
    print("\n5. A MODIFY decision saves correctly")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(3)
    plan = generate_response_plan(inf, conf)

    edited_steps = [PlanStep(**s) for s in plan_to_dict(plan)["steps"]]
    edited_steps[0].action = "Review failed-login activity (analyst-edited)"
    import dataclasses as dc

    modified_plan = dc.replace(plan, steps=edited_steps)
    modified_plan.validation_status = validate_plan(modified_plan)
    check(modified_plan.validation_status.is_valid, "sanity: the edited plan is still valid")

    record = make_record(incident_id, inf, conf, plan, "MODIFY", modified_plan=modified_plan)
    ok, errors = save_feedback(record, db_path)
    check(ok, f"MODIFY saves without error (got errors: {errors})")

    loaded = load_feedback(db_path)
    check(loaded.iloc[0]["review_status"] == "MODIFIED", "the saved review_status is MODIFIED")


def test_modified_plan_stored_separately_from_original():
    print("\n6. The modified plan is stored SEPARATELY from the original plan")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(4)
    plan = generate_response_plan(inf, conf)

    edited_steps = [PlanStep(**s) for s in plan_to_dict(plan)["steps"]]
    original_first_action = edited_steps[0].action
    edited_steps[0].action = "COMPLETELY DIFFERENT WORDING FOR THIS STEP"
    import dataclasses as dc

    modified_plan = dc.replace(plan, steps=edited_steps)
    modified_plan.validation_status = validate_plan(modified_plan)

    record = make_record(incident_id, inf, conf, plan, "MODIFY", modified_plan=modified_plan)
    ok, _ = save_feedback(record, db_path)
    check(ok, "sanity: the MODIFY record saved")

    loaded = load_feedback(db_path).iloc[0]
    check(loaded["original_plan"]["steps"][0]["action"] == original_first_action, "the ORIGINAL plan's first step is unchanged")
    check(
        loaded["modified_plan"]["steps"][0]["action"] == "COMPLETELY DIFFERENT WORDING FOR THIS STEP",
        "the MODIFIED plan's first step reflects the analyst's edit",
    )
    check(
        loaded["original_plan"]["steps"][0]["action"] != loaded["modified_plan"]["steps"][0]["action"],
        "the original and modified plans are genuinely different objects, not the same data twice",
    )


# =======================================================================
# 7-8. Invalid / malformed feedback
# =======================================================================
def test_invalid_decision_is_rejected():
    print("\n7. An invalid analyst decision is rejected, and nothing is saved")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(5)
    plan = generate_response_plan(inf, conf)
    record = FeedbackRecord(
        feedback_id=generate_feedback_id(incident_id), incident_id=incident_id,
        timestamp=datetime.now(timezone.utc).isoformat(), incident_type=inf.incident_type,
        severity=inf.severity, confidence_level=conf.confidence_level, confidence_score=conf.confidence_score,
        human_review_recommended=conf.human_review_recommended, analyst_decision="DELETE_EVERYTHING",
        review_status="APPROVED", original_plan=plan_to_dict(plan),
    )
    check_result = validate_feedback(record)
    check(not check_result.is_valid, "validate_feedback() flags the invalid decision")
    ok, errors = save_feedback(record, db_path)
    check(not ok, "save_feedback() refuses to save an invalid decision")
    check(any("not a valid analyst decision" in e for e in errors), "the error clearly names the problem")
    check(load_feedback(db_path).empty, "nothing was written to the database")


def test_empty_invalid_feedback_handled_safely():
    print("\n8. Empty/invalid feedback fields are handled safely (never crash)")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(6)
    plan = generate_response_plan(inf, conf)

    # Empty notes must be accepted (notes are optional, not a validity rule).
    record = make_record(incident_id, inf, conf, plan, "APPROVE", notes="")
    ok, errors = save_feedback(record, db_path)  # must not raise
    check(ok, f"an APPROVE with empty notes still saves successfully (errors: {errors})")

    # Missing incident_id must be rejected safely, not crash.
    bad_record = make_record("", inf, conf, plan, "APPROVE")
    ok2, errors2 = save_feedback(bad_record, db_path)  # must not raise
    check(not ok2, "a record with an empty incident_id is rejected")
    check(len(errors2) > 0, "a clear error is returned for the missing incident_id")

    # None notes must be coerced to a safe empty string, never crash the DB write.
    none_notes_record = make_record(build_incident_id(60), inf, conf, plan, "APPROVE")
    none_notes_record.analyst_notes = None
    ok3, errors3 = save_feedback(none_notes_record, db_path)  # must not raise
    check(ok3, f"a record with notes=None still saves safely (errors: {errors3})")


# =======================================================================
# 9-11. Persistence and retrieval
# =======================================================================
def test_feedback_survives_reopening_the_database():
    print("\n9. Feedback survives reopening the database (real file-based persistence)")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(7)
    plan = generate_response_plan(inf, conf)
    record = make_record(incident_id, inf, conf, plan, "APPROVE", notes="Persisted across a reopen.")
    ok, _ = save_feedback(record, db_path)
    check(ok, "sanity: the record saved")

    # Every module function opens and closes its own connection - calling
    # load_feedback() again is a genuine "reopen", not a cached in-memory read.
    reopened = load_feedback(db_path)
    check(len(reopened) == 1, "the record is still there after reopening the database")
    check(reopened.iloc[0]["analyst_notes"] == "Persisted across a reopen.", "the record's content survived the reopen intact")

    # Also confirm it's real file-based persistence, not something only
    # this module's own connection object could see.
    import sqlite3

    raw_conn = sqlite3.connect(db_path)
    raw_count = raw_conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    raw_conn.close()
    check(raw_count == 1, "a completely independent sqlite3 connection sees the same persisted row")


def test_multiple_feedback_records_can_be_stored():
    print("\n10. Multiple feedback records can be stored")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    plan = generate_response_plan(inf, conf)
    for i in range(5):
        record = make_record(build_incident_id(100 + i), inf, conf, plan, "APPROVE", notes=f"Review #{i}")
        ok, errors = save_feedback(record, db_path)
        check(ok, f"record {i} saves without error ({errors})")

    loaded = load_feedback(db_path)
    check(len(loaded) == 5, "all 5 records are present")
    check(loaded["incident_id"].nunique() == 5, "all 5 records have distinct incident_ids")


def test_feedback_retrieved_by_incident_id():
    print("\n11. Feedback can be retrieved by incident ID")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    plan = generate_response_plan(inf, conf)
    incident_a, incident_b = build_incident_id(200), build_incident_id(201)
    save_feedback(make_record(incident_a, inf, conf, plan, "APPROVE"), db_path)
    save_feedback(make_record(incident_a, inf, conf, plan, "MODIFY", modified_plan=plan), db_path)
    save_feedback(make_record(incident_b, inf, conf, plan, "REJECT"), db_path)

    only_a = get_feedback_by_incident(incident_a, db_path)
    check(len(only_a) == 2, "both of incident A's reviews are returned")
    check(set(only_a["incident_id"]) == {incident_a}, "no other incident's feedback leaks into the result")

    only_b = get_feedback_by_incident(incident_b, db_path)
    check(len(only_b) == 1, "incident B's single review is returned")

    none_found = get_feedback_by_incident(build_incident_id(9999), db_path)
    check(none_found.empty, "an incident with no feedback returns an empty result, not an error")


# =======================================================================
# 12. Review statuses
# =======================================================================
def test_review_statuses_work_correctly():
    print("\n12. PENDING/APPROVED/REJECTED/MODIFIED statuses work correctly")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    plan = generate_response_plan(inf, conf)

    reviewed_approve = build_incident_id(300)
    reviewed_reject = build_incident_id(301)
    reviewed_modify = build_incident_id(302)
    never_reviewed = build_incident_id(303)

    save_feedback(make_record(reviewed_approve, inf, conf, plan, "APPROVE"), db_path)
    save_feedback(make_record(reviewed_reject, inf, conf, plan, "REJECT"), db_path)
    save_feedback(make_record(reviewed_modify, inf, conf, plan, "MODIFY", modified_plan=plan), db_path)

    status_map = get_review_status_map(load_feedback(db_path))
    check(status_map[reviewed_approve] == "APPROVED", "an APPROVE decision maps to review_status APPROVED")
    check(status_map[reviewed_reject] == "REJECTED", "a REJECT decision maps to review_status REJECTED")
    check(status_map[reviewed_modify] == "MODIFIED", "a MODIFY decision maps to review_status MODIFIED")
    check(
        never_reviewed not in status_map,
        "an incident with no feedback record simply isn't in the map - callers treat this as PENDING",
    )


# =======================================================================
# 13-14. Modified-plan validation and prohibited actions
# =======================================================================
def test_modified_plans_must_pass_phase6_validation():
    print("\n13. Modified plans must pass Phase 6's own validation before they can be saved")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(8)
    plan = generate_response_plan(inf, conf)

    # A structurally broken modified plan: duplicate step numbers.
    broken_steps = [PlanStep(**s) for s in plan_to_dict(plan)["steps"]]
    broken_steps[1].step_number = broken_steps[0].step_number
    import dataclasses as dc

    broken_plan = dc.replace(plan, steps=broken_steps)
    broken_plan.validation_status = validate_plan(broken_plan)
    check(not broken_plan.validation_status.is_valid, "sanity: Phase 6's validator itself flags the duplicate step numbers")

    record = make_record(incident_id, inf, conf, plan, "MODIFY", modified_plan=broken_plan)
    ok, errors = save_feedback(record, db_path)
    check(not ok, "save_feedback() refuses a MODIFY whose plan fails Phase 6 validation")
    check(any("Modified plan:" in e for e in errors), "the error clearly attributes the problem to the modified plan")
    check(load_feedback(db_path).empty, "nothing was written to the database")


def test_prohibited_actions_cannot_be_saved():
    print("\n14. Prohibited (offensive/automated) actions cannot be saved, in either plan")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(9)
    plan = generate_response_plan(inf, conf)

    bad_steps = [PlanStep(**s) for s in plan_to_dict(plan)["steps"]]
    bad_steps[0].action = "Disable the account immediately"
    import dataclasses as dc

    bad_plan = dc.replace(plan, steps=bad_steps)
    bad_plan.validation_status = validate_plan(bad_plan)

    record = make_record(incident_id, inf, conf, plan, "MODIFY", modified_plan=bad_plan)
    ok, errors = save_feedback(record, db_path)
    check(not ok, "a MODIFY containing a prohibited action is rejected")
    check(
        any("prohibited action keyword" in e.lower() for e in errors),
        "the specific prohibited keyword is named in the rejection",
    )
    check(load_feedback(db_path).empty, "the prohibited-action plan was never persisted")


# =======================================================================
# 15. true_label
# =======================================================================
def test_true_label_not_stored_or_used():
    print("\n15. true_label is never stored in, or used by, the feedback module")
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(FeedbackRecord)}
    check("true_label" not in field_names, "FeedbackRecord has no true_label field at all")

    import src.feedback as feedback_module

    check("LABEL_COLUMN" not in vars(feedback_module), "config.LABEL_COLUMN is not imported into feedback.py's namespace")

    # Behavioral check: two otherwise-identical events differing only in
    # true_label produce identical feedback records once saved.
    db_path = fresh_db_path()
    inf_a, conf_a = make_pair(true_label="normal")
    inf_b, conf_b = make_pair(true_label="suspicious")
    plan_a = generate_response_plan(inf_a, conf_a)
    plan_b = generate_response_plan(inf_b, conf_b)
    check(
        plan_to_dict(plan_a) == plan_to_dict(plan_b),
        "flipping true_label upstream never changes the plan that would be reviewed/stored",
    )


# =======================================================================
# 16. Incident ID stability
# =======================================================================
def test_incident_ids_remain_stable():
    print("\n16. Incident IDs are stable and deterministic")
    check(build_incident_id(7) == build_incident_id(7) == "INC-0007", "the same index always produces the same ID")
    check(build_incident_id(0) == "INC-0000", "index 0 formats correctly")
    check(build_incident_id(123) == "INC-0123", "a multi-digit index formats correctly, zero-padded")

    # Cross-run determinism: two independent runs of the full pipeline
    # produce the same classification for the same row index, so the
    # incident ID keeps referring to the "same" incident across restarts.
    from src.anomaly_detector import run_anomaly_detection
    from src.data_loader import load_logs
    from src.preprocessor import run_preprocessing_pipeline
    from src.rule_engine import run_rule_engine

    def classify_row(idx):
        df = load_logs()
        pre = run_preprocessing_pipeline(df)
        anomaly = run_anomaly_detection(pre)
        rule_result = run_rule_engine(pre, anomaly)
        rec = rule_result.results_df.loc[idx, "_inference"]
        return rec.incident_type, rec.severity

    first_run = classify_row(11)
    second_run = classify_row(11)
    check(first_run == second_run, "the same event index classifies identically across two independent pipeline runs")


# =======================================================================
# 17. Analyst notes
# =======================================================================
def test_analyst_notes_stored_correctly():
    print("\n17. Analyst notes are stored and retrieved exactly as entered")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    incident_id = build_incident_id(10)
    plan = generate_response_plan(inf, conf)
    note_text = "Reviewed with authentication logs; activity appears expected.\nSecond line, with 'quotes' and a comma, here."
    record = make_record(incident_id, inf, conf, plan, "APPROVE", notes=note_text)
    ok, _ = save_feedback(record, db_path)
    check(ok, "sanity: the record with a multi-line note saved")

    loaded = load_feedback(db_path).iloc[0]
    check(loaded["analyst_notes"] == note_text, "the note round-trips through SQLite byte-for-byte, including punctuation and newlines")


# =======================================================================
# 18. Feedback summary metrics
# =======================================================================
def test_feedback_summary_metrics_calculated_correctly():
    print("\n18. Feedback summary metrics are calculated correctly")
    db_path = fresh_db_path()
    inf, conf = make_pair()
    plan = generate_response_plan(inf, conf)

    decisions = ["APPROVE", "APPROVE", "REJECT", "MODIFY"]
    for i, decision in enumerate(decisions):
        modified = plan if decision == "MODIFY" else None
        note = "policy review needed" if decision == "MODIFY" else "routine review routine"
        rec = make_record(build_incident_id(400 + i), inf, conf, plan, decision, modified_plan=modified, notes=note)
        ok, errors = save_feedback(rec, db_path)
        check(ok, f"decision {i} ({decision}) saved ({errors})")

    summary = summarize_feedback(load_feedback(db_path))
    check(summary.total_reviews == 4, "total_reviews counts all 4 saved records")
    check(summary.approved_count == 2, "approved_count is exactly 2")
    check(summary.rejected_count == 1, "rejected_count is exactly 1")
    check(summary.modified_count == 1, "modified_count is exactly 1")
    check(summary.approval_rate == 0.5, "approval_rate is 2/4 = 0.5")
    check(summary.rejection_rate == 0.25, "rejection_rate is 1/4 = 0.25")
    check(summary.modification_rate == 0.25, "modification_rate is 1/4 = 0.25")
    check("routine" in summary.note_keyword_counts, "a repeated, meaningful word from the notes is counted as a keyword")
    check(summary.note_keyword_counts.get("routine", 0) >= 2, "the keyword count reflects how many notes actually used that word")


def test_empty_feedback_database_handled_safely():
    print("\n19. An empty feedback database is handled safely everywhere")
    db_path = fresh_db_path()
    df = load_feedback(db_path)  # must not raise
    check(df.empty, "load_feedback() on a fresh database returns an empty DataFrame")

    summary = summarize_feedback(df)  # must not raise
    check(summary.total_reviews == 0, "summarize_feedback() on empty data reports 0 total reviews")
    check(summary.approval_rate == 0.0, "rates are 0.0, not a division-by-zero crash, when there is no data")

    status_map = get_review_status_map(df)  # must not raise
    check(status_map == {}, "get_review_status_map() on empty data returns an empty map")

    by_incident = get_feedback_by_incident(build_incident_id(1), db_path)  # must not raise
    check(by_incident.empty, "get_feedback_by_incident() on an empty database returns an empty result")


def test_database_errors_handled_safely():
    print("\n20. Database connection failures are handled safely, never crash the app")
    # A directory path can never be opened as a SQLite file - this
    # reliably exercises the failure path without relying on filesystem
    # permissions, which can vary in a sandboxed environment.
    bad_path = tempfile.mkdtemp(prefix="phase7_bad_db_")
    _TEMP_DIRS.append(bad_path)

    ok = initialize_feedback_database(bad_path)  # must not raise
    check(ok is False, "initialize_feedback_database() reports failure instead of raising")

    df = load_feedback(bad_path)  # must not raise
    check(df.empty, "load_feedback() returns an empty DataFrame instead of raising on a connection failure")

    inf, conf = make_pair()
    plan = generate_response_plan(inf, conf)
    record = make_record(build_incident_id(1), inf, conf, plan, "APPROVE")
    save_ok, errors = save_feedback(record, bad_path)  # must not raise
    check(save_ok is False, "save_feedback() reports failure instead of raising on a connection failure")
    check(len(errors) > 0, "a clear error message is returned")


# =======================================================================
# 27-32. Streamlit UI
# =======================================================================
def test_streamlit_analyst_review_page_loads():
    # UI redesign note: "Analyst Review" is no longer a standalone page -
    # its decision workflow (radio/notes/plan editor/submit, all reused
    # verbatim) is the "Analyst Decision" section on "Case Analysis" (the
    # app's landing page, so no navigation click is needed to reach it).
    # The old page's "Pending Review Dashboard" table now lives in the
    # "Case History" tab under "Review & Insights".
    print("\n27. Streamlit 'Case Analysis' page loads without exceptions (analyst decision)")
    from streamlit.testing.v1 import AppTest

    with isolated_app_database():
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        check(len(at.exception) == 0, "the page renders with no exception (auto-runs Phases 1-4 first)")
        check(any("Analyst Decision" == sh.value for sh in at.subheader), "the Analyst Decision section is rendered")
        check(
            any("final decision" in w.value.lower() for w in list(at.warning) + list(at.info)),
            "the advisory-only / final-decision statement is shown on the page",
        )

        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=60)
        check(len(at.exception) == 0, "Review & Insights renders with no exception")
        check(
            any("Total Cases" == m.label for m in at.metric),
            "the Case History dashboard (successor to the old Pending Review Dashboard) is rendered",
        )


def test_streamlit_feedback_history_page_loads():
    print("\n28. Streamlit 'Review & Insights' page loads without exceptions (feedback)")
    from streamlit.testing.v1 import AppTest

    with isolated_app_database():
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=60)
        check(len(at.exception) == 0, "the page renders with no exception, even with no feedback recorded yet")


def test_approve_interaction_works():
    print("\n29. The APPROVE interaction works end-to-end through the real UI")
    from streamlit.testing.v1 import AppTest

    with isolated_app_database():
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        [r for r in at.radio if r.label == "Decision"][0].set_value("APPROVE").run(timeout=60)
        [ta for ta in at.text_area if ta.label == "Analyst Notes"][0].set_value("Approving via test.").run(timeout=60)
        [b for b in at.button if "Submit Decision" in b.label][0].click().run(timeout=60)
        check(len(at.exception) == 0, "no exception during the APPROVE interaction")
        check(any("APPROVED" in s.value for s in at.success), "a success message confirming APPROVED is shown")


def test_reject_interaction_works():
    print("\n30. The REJECT interaction works end-to-end through the real UI")
    from streamlit.testing.v1 import AppTest

    with isolated_app_database():
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        [r for r in at.radio if r.label == "Decision"][0].set_value("REJECT").run(timeout=60)
        [ta for ta in at.text_area if ta.label == "Analyst Notes"][0].set_value("Rejecting via test.").run(timeout=60)
        [b for b in at.button if "Submit Decision" in b.label][0].click().run(timeout=60)
        check(len(at.exception) == 0, "no exception during the REJECT interaction")
        check(any("REJECTED" in s.value for s in at.success), "a success message confirming REJECTED is shown")


def test_modify_interaction_works():
    print("\n31. The MODIFY interaction (editing a real step) works end-to-end through the real UI")
    from streamlit.testing.v1 import AppTest

    with isolated_app_database():
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        [r for r in at.radio if r.label == "Decision"][0].set_value("MODIFY").run(timeout=60)
        check(len(at.exception) == 0, "no exception when the step editor renders")

        action_inputs = [ti for ti in at.text_input if ti.label == "Action"]
        check(len(action_inputs) > 0, "the step editor shows at least one editable Action field")
        action_inputs[0].set_value("Review the anomaly evidence (edited via test)").run(timeout=60)
        [ta for ta in at.text_area if ta.label == "Analyst Notes"][0].set_value("Edited wording via test.").run(timeout=60)
        [b for b in at.button if "Submit Decision" in b.label][0].click().run(timeout=60)
        check(len(at.exception) == 0, "no exception during the MODIFY interaction")
        check(any("MODIFIED" in s.value for s in at.success), "a success message confirming MODIFIED is shown")


def test_modified_plan_validation_works_in_ui():
    print("\n32. The UI blocks a MODIFY submission whose plan fails Phase 6 validation")
    from streamlit.testing.v1 import AppTest

    with isolated_app_database() as db_path:
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        [r for r in at.radio if r.label == "Decision"][0].set_value("MODIFY").run(timeout=60)

        action_inputs = [ti for ti in at.text_input if ti.label == "Action"]
        action_inputs[0].set_value("Block IP of the attacker immediately").run(timeout=60)
        check(len(at.exception) == 0, "no exception while entering a prohibited action")
        check(
            any("FAILS validation" in e.value for e in at.error),
            "the UI clearly shows that the modified plan fails validation",
        )
        submit_buttons = [b for b in at.button if "Submit Decision" in b.label]
        check(submit_buttons[0].disabled, "the Submit Decision button is disabled while the plan is invalid")

        before_count = len(load_feedback(db_path))
        submit_buttons[0].click().run(timeout=60)
        check(len(at.exception) == 0, "clicking the (disabled) button still never crashes the app")
        after_count = len(load_feedback(db_path))
        check(after_count == before_count, "no invalid record was written even if the click was processed")


def run_phase1_through_phase6_regression():
    print("\n" + "=" * 70)
    print("Regression: re-running the Phase 1, 2, 3, 4, 5, and 6 test suites")
    print("=" * 70)
    import test_response_planner  # its own main() already re-runs Phase 1-5

    test_response_planner.main()


def main():
    print("=" * 70)
    print("Phase 7 human analyst review & feedback loop test suite")
    print("=" * 70)

    try:
        test_database_initializes_successfully()
        test_feedback_table_created_automatically()
        test_approve_decision_saves_correctly()
        test_reject_decision_saves_correctly()
        test_modify_decision_saves_correctly()
        test_modified_plan_stored_separately_from_original()
        test_invalid_decision_is_rejected()
        test_empty_invalid_feedback_handled_safely()
        test_feedback_survives_reopening_the_database()
        test_multiple_feedback_records_can_be_stored()
        test_feedback_retrieved_by_incident_id()
        test_review_statuses_work_correctly()
        test_modified_plans_must_pass_phase6_validation()
        test_prohibited_actions_cannot_be_saved()
        test_true_label_not_stored_or_used()
        test_incident_ids_remain_stable()
        test_analyst_notes_stored_correctly()
        test_feedback_summary_metrics_calculated_correctly()
        test_empty_feedback_database_handled_safely()
        test_database_errors_handled_safely()
        test_streamlit_analyst_review_page_loads()
        test_streamlit_feedback_history_page_loads()
        test_approve_interaction_works()
        test_reject_interaction_works()
        test_modify_interaction_works()
        test_modified_plan_validation_works_in_ui()
    finally:
        cleanup_temp_dirs()

    print("\n" + "=" * 70)
    print(f"PHASE 7 CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)

    run_phase1_through_phase6_regression()

    print("\n" + "=" * 70)
    print("ALL PHASE 1 + 2 + 3 + 4 + 5 + 6 + 7 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
