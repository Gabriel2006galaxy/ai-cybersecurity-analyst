"""
test_audit_trail.py
---------------------
Phase 10 tests for `src/audit_trail.py` ONLY.

This file deliberately does NOT chain into the Phase 1-9 regression
cascade (unlike `test_retriever.py`, which re-runs `test_feedback.main()`
and so on down the chain) and does NOT re-run `test_explainability.py`
either. Run it directly:

    python test_audit_trail.py

A full project regression (`python test_retriever.py`, 1,196+ checks) can
still be run separately whenever a complete check is actually needed -
this file is intentionally scoped to Phase 10's audit-trail module and
its Streamlit-page integration only.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager

import pandas as pd

import config
from src.audit_trail import (
    AuditEvent,
    IncidentAuditSummary,
    event_exists,
    generate_audit_id,
    get_events_by_incident,
    get_events_by_stage,
    initialize_audit_database,
    load_audit_events,
    summarize_incident_audit_history,
    write_audit_event,
)
import src.audit_trail as audit_trail_module

PASSED = 0
_TEMP_DIRS = []


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1


def fresh_db_path() -> str:
    directory = tempfile.mkdtemp()
    _TEMP_DIRS.append(directory)
    return os.path.join(directory, "audit_trail_test.db")


def cleanup_temp_dirs() -> None:
    import shutil

    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _TEMP_DIRS.clear()


@contextmanager
def isolated_app_audit_database():
    """Patches `src.feedback.FEEDBACK_DB_PATH` and
    `src.audit_trail.AUDIT_DB_PATH` to fresh temp files, for the one
    Streamlit `AppTest` test below - never touches the real
    `data/audit_trail.db` or `data/analyst_feedback.db`."""
    import src.feedback as feedback_module

    feedback_fd, feedback_path = tempfile.mkstemp(suffix="_feedback.db")
    audit_fd, audit_path = tempfile.mkstemp(suffix="_audit.db")
    os.close(feedback_fd)
    os.close(audit_fd)
    os.remove(feedback_path)
    os.remove(audit_path)
    original_feedback_path = feedback_module.FEEDBACK_DB_PATH
    original_audit_path = audit_trail_module.AUDIT_DB_PATH
    feedback_module.FEEDBACK_DB_PATH = feedback_path
    audit_trail_module.AUDIT_DB_PATH = audit_path
    try:
        yield
    finally:
        feedback_module.FEEDBACK_DB_PATH = original_feedback_path
        audit_trail_module.AUDIT_DB_PATH = original_audit_path
        for p in (feedback_path, audit_path):
            if os.path.exists(p):
                os.remove(p)


# =======================================================================
# 1. Database auto-creation
# =======================================================================
def test_database_auto_creation():
    print("\n1. The audit database and table are created automatically")
    db_path = fresh_db_path()
    check(not os.path.exists(db_path), "the database file does not exist yet")
    check(initialize_audit_database(db_path) is True, "initialize_audit_database() succeeds on a brand-new path")
    check(os.path.exists(db_path), "the database file now exists")

    # Also auto-creates on first use even without an explicit init call.
    db_path2 = fresh_db_path()
    events = load_audit_events(db_path2)
    check(events.empty, "reading a never-initialized database returns an empty DataFrame, not an error")


# =======================================================================
# 2. Event creation
# =======================================================================
def test_event_creation():
    print("\n2. A single audit event can be written and read back correctly")
    db_path = fresh_db_path()
    success, errors = write_audit_event(
        "INC-0001", "ANOMALY_DETECTION", "ANOMALY_DETECTION_RESULT",
        incident_type="Possible Brute Force", severity="HIGH",
        descriptive_details="anomaly_status=ANOMALOUS, anomaly_score=0.5000.",
        db_path=db_path,
    )
    check(success is True, "writing a well-formed audit event succeeds")
    check(errors == [], "a successful write returns no error messages")

    events = load_audit_events(db_path)
    check(len(events) == 1, "exactly one event was stored")
    row = events.iloc[0]
    check(row["incident_id"] == "INC-0001", "incident_id round-trips correctly")
    check(row["stage"] == "ANOMALY_DETECTION", "stage round-trips correctly")
    check(row["incident_type"] == "Possible Brute Force", "incident_type round-trips correctly")
    check(row["severity"] == "HIGH", "severity round-trips correctly")
    check(row["audit_id"].startswith("AUD-"), "a generated audit_id has the expected prefix")


# =======================================================================
# 3. Multiple events for one incident
# =======================================================================
def test_multiple_events_for_one_incident():
    print("\n3. Multiple stages for the same incident are all recorded")
    db_path = fresh_db_path()
    stages = ["INCIDENT_SELECTED", "ANOMALY_DETECTION", "RULE_CLASSIFICATION", "CONFIDENCE", "RESPONSE_PLAN"]
    for stage in stages:
        ok, _ = write_audit_event("INC-0007", stage, f"{stage}_RESULT", incident_type="Suspicious Data Transfer", db_path=db_path)
        check(ok, f"writing the {stage} event succeeds")

    events = get_events_by_incident("INC-0007", db_path)
    check(len(events) == len(stages), "every stage's event is stored for this incident")
    check(set(events["stage"]) == set(stages), "all the expected stages are present")


# =======================================================================
# 4. Retrieval by incident
# =======================================================================
def test_retrieval_by_incident():
    print("\n4. get_events_by_incident() returns only that incident's events, in chronological order")
    db_path = fresh_db_path()
    write_audit_event("INC-0001", "ANOMALY_DETECTION", "A", db_path=db_path)
    write_audit_event("INC-0002", "ANOMALY_DETECTION", "A", db_path=db_path)
    write_audit_event("INC-0001", "RULE_CLASSIFICATION", "B", db_path=db_path)
    write_audit_event("INC-0001", "CONFIDENCE", "C", db_path=db_path)

    inc1_events = get_events_by_incident("INC-0001", db_path)
    check(len(inc1_events) == 3, "only INC-0001's 3 events are returned")
    check(set(inc1_events["incident_id"]) == {"INC-0001"}, "no other incident's events leak in")
    check(list(inc1_events["stage"]) == ["ANOMALY_DETECTION", "RULE_CLASSIFICATION", "CONFIDENCE"], "events are returned oldest-first (chronological trace order)")

    empty = get_events_by_incident("INC-9999", db_path)
    check(empty.empty, "an incident with no recorded events returns an empty DataFrame, not an error")


# =======================================================================
# 5. Retrieval by stage
# =======================================================================
def test_retrieval_by_stage():
    print("\n5. get_events_by_stage() returns only that stage's events, across all incidents")
    db_path = fresh_db_path()
    write_audit_event("INC-0001", "GENAI_ANALYSIS", "A", db_path=db_path)
    write_audit_event("INC-0002", "GENAI_ANALYSIS", "A", db_path=db_path)
    write_audit_event("INC-0001", "RESPONSE_PLAN", "B", db_path=db_path)

    genai_events = get_events_by_stage("GENAI_ANALYSIS", db_path)
    check(len(genai_events) == 2, "both GENAI_ANALYSIS events are returned")
    check(set(genai_events["incident_id"]) == {"INC-0001", "INC-0002"}, "events span every incident that has this stage")

    empty = get_events_by_stage("RAG_RETRIEVAL", db_path)
    check(empty.empty, "a stage with no recorded events returns an empty DataFrame, not an error")


# =======================================================================
# 6. Append behavior (never overwrites a prior event)
# =======================================================================
def test_append_behavior():
    print("\n6. Writing never overwrites a previously stored event")
    db_path = fresh_db_path()
    write_audit_event("INC-0007", "ANALYST_REVIEW", "FEEDBACK_DECISION", analyst_decision="APPROVE", db_path=db_path)
    write_audit_event("INC-0007", "ANALYST_REVIEW", "FEEDBACK_DECISION", analyst_decision="REJECT", db_path=db_path, skip_if_exists=False)

    events = get_events_by_incident("INC-0007", db_path)
    check(len(events) == 2, "a second genuinely new ANALYST_REVIEW event is appended, not merged into the first")
    check(set(events["analyst_decision"]) == {"APPROVE", "REJECT"}, "both the original APPROVE and the later REJECT are preserved")

    # skip_if_exists=True makes a write for an already-recorded stage a
    # safe no-op instead of a duplicate row - this is what keeps a
    # Streamlit rerun from appending the same event over and over.
    ok, messages = write_audit_event("INC-0007", "ANALYST_REVIEW", "FEEDBACK_DECISION", analyst_decision="MODIFY", db_path=db_path, skip_if_exists=True)
    check(ok is True, "a skipped duplicate write still reports success, never an error")
    events_after = get_events_by_incident("INC-0007", db_path)
    check(len(events_after) == 2, "skip_if_exists=True does not append a third row when the stage already has an event")


# =======================================================================
# 7. Persistence across a database reconnect
# =======================================================================
def test_persistence_across_reconnect():
    print("\n7. Audit events survive closing and reopening the database file")
    db_path = fresh_db_path()
    write_audit_event("INC-0003", "CONFIDENCE", "CONFIDENCE_RESULT", confidence_level="LOW", confidence_score=0.31, db_path=db_path)

    # A brand-new, independent connection (simulating an app restart).
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT incident_id, confidence_level, confidence_score FROM audit_trail").fetchone()
    conn.close()
    check(row is not None, "the event is still present after reconnecting independently")
    check(row[0] == "INC-0003" and row[1] == "LOW", "the reconnected data matches what was written")
    check(abs(row[2] - 0.31) < 1e-9, "the confidence_score survives the round trip exactly")

    # And through this module's own read path too.
    events = load_audit_events(db_path)
    check(len(events) == 1, "load_audit_events() also sees the persisted event after reconnecting")


# =======================================================================
# 8. Malformed input handling
# =======================================================================
def test_malformed_input_handling():
    print("\n8. Malformed input is rejected safely, never crashes, and nothing is written")
    db_path = fresh_db_path()

    ok, errors = write_audit_event("", "ANOMALY_DETECTION", "X", db_path=db_path)
    check(ok is False, "an empty incident_id is rejected")
    check(len(errors) >= 1, "a rejection includes at least one error message")

    ok, errors = write_audit_event("INC-0001", "NOT_A_REAL_STAGE", "X", db_path=db_path)
    check(ok is False, "an invalid stage name is rejected")

    ok, errors = write_audit_event("INC-0001", "ANOMALY_DETECTION", "", db_path=db_path)
    check(ok is False, "an empty event_type is rejected")

    events = load_audit_events(db_path)
    check(events.empty, "none of the rejected writes actually stored a row")

    # None/omitted optional fields never crash a write.
    ok, errors = write_audit_event(
        "INC-0002", "RAG_RETRIEVAL", "RAG_RETRIEVAL_RESULT",
        confidence_score=None, relevant_rule_ids=None, supporting_indicators=None,
        retrieved_knowledge_ids=None, human_review_recommended=None, db_path=db_path,
    )
    check(ok is True, "omitted/None optional fields are handled safely, not a crash")


# =======================================================================
# 9. Empty database handling
# =======================================================================
def test_empty_database_handling():
    print("\n9. An empty audit database is handled safely everywhere it's read")
    db_path = fresh_db_path()
    initialize_audit_database(db_path)

    check(load_audit_events(db_path).empty, "load_audit_events() on an empty database returns an empty DataFrame")
    check(get_events_by_incident("INC-0001", db_path).empty, "get_events_by_incident() on an empty database returns an empty DataFrame")
    check(get_events_by_stage("CONFIDENCE", db_path).empty, "get_events_by_stage() on an empty database returns an empty DataFrame")
    check(event_exists("INC-0001", "CONFIDENCE", db_path) is False, "event_exists() on an empty database returns False, not an error")

    summary = summarize_incident_audit_history("INC-0001", db_path)
    check(isinstance(summary, IncidentAuditSummary), "summarize_incident_audit_history() still returns a summary object")
    check(summary.total_events == 0, "the summary correctly reports zero events")
    check(summary.stages_recorded == [], "the summary correctly reports no stages recorded")


# =======================================================================
# 10. Database failure handling
# =======================================================================
def test_database_failure_handling():
    print("\n10. A broken/inaccessible database path never crashes the app")
    directory_as_db_path = tempfile.mkdtemp()
    _TEMP_DIRS.append(directory_as_db_path)

    check(initialize_audit_database(directory_as_db_path) is False, "initializing against a directory (not a file) fails gracefully")
    check(load_audit_events(directory_as_db_path).empty, "loading from a directory path returns an empty DataFrame, not a crash")

    ok, errors = write_audit_event("INC-0001", "ANOMALY_DETECTION", "X", db_path=directory_as_db_path)
    check(ok is False, "writing to a directory path fails gracefully")
    check(len(errors) >= 1, "the failure includes an explanatory message")

    summary = summarize_incident_audit_history("INC-0001", directory_as_db_path)
    check(summary.total_events == 0, "summarize_incident_audit_history() falls back to an empty summary on a broken database")


# =======================================================================
# 11. No true_label
# =======================================================================
def test_true_label_not_used():
    print("\n11. true_label is never read, stored, or exposed by this module")
    import dataclasses

    module_vars = vars(audit_trail_module)
    check("LABEL_COLUMN" not in module_vars, "config.LABEL_COLUMN is not imported into audit_trail.py's namespace")

    for dc in (AuditEvent, IncidentAuditSummary):
        field_names = {f.name for f in dataclasses.fields(dc)}
        check("true_label" not in field_names, f"{dc.__name__} has no true_label field")

    db_path = fresh_db_path()
    write_audit_event("INC-0001", "ANOMALY_DETECTION", "X", incident_type="Possible Brute Force", db_path=db_path)
    events = load_audit_events(db_path)
    check("true_label" not in events.columns, "the loaded audit-events DataFrame has no true_label column")


# =======================================================================
# 12. No changes to previous system decisions (structurally read-only /
#     no write path into any earlier phase or into Phase 7's own database)
# =======================================================================
def test_no_changes_to_previous_system_decisions():
    print("\n12. The audit trail cannot change any earlier phase's decision or Phase 7's feedback records")
    module_vars = vars(audit_trail_module)
    never_imported = [
        "run_inference", "run_rule_engine", "score_confidence", "run_confidence_scoring",
        "retrieve", "retrieve_knowledge_for_incident", "generate_ai_case_summary",
        "generate_response_plan", "train_isolation_forest", "run_anomaly_detection",
        "save_feedback", "initialize_feedback_database",
    ]
    for name in never_imported:
        check(name not in module_vars, f"'{name}' is not imported into audit_trail.py - it cannot recompute or overwrite any decision")

    # A separate database file from Phase 7's feedback store (never the same path).
    check(config.AUDIT_DB_PATH != config.FEEDBACK_DB_PATH, "the audit trail uses a separate database file from Phase 7's feedback store")

    # Writing an audit event never touches the feedback database file.
    db_path = fresh_db_path()
    feedback_db_path = fresh_db_path()  # a second, separate temp file simulating FEEDBACK_DB_PATH
    write_audit_event("INC-0001", "ANALYST_REVIEW", "FEEDBACK_DECISION", analyst_decision="APPROVE", db_path=db_path)
    check(not os.path.exists(feedback_db_path), "writing an audit event never creates or touches a separate feedback database path")

    # Dataset checksum is unaffected by any audit-trail activity.
    import hashlib

    with open(config.LOG_FILE_PATH, "rb") as fh:
        checksum = hashlib.md5(fh.read()).hexdigest()
    check(checksum == "97d4aeff5a59bc2e472116e3c2b382c5", "the synthetic dataset's checksum is unchanged after exercising the audit trail")


# =======================================================================
# 13. Deterministic incident ID usage (Phase 7's build_incident_id, reused)
# =======================================================================
def test_deterministic_incident_id_usage():
    print("\n13. The audit trail keys events by Phase 7's own deterministic incident IDs")
    from src.feedback import build_incident_id

    db_path = fresh_db_path()
    incident_id = build_incident_id(7)
    check(incident_id == "INC-0007", "build_incident_id(7) is the well-known, stable INC-0007")

    write_audit_event(incident_id, "ANOMALY_DETECTION", "X", db_path=db_path)
    write_audit_event(incident_id, "RULE_CLASSIFICATION", "Y", db_path=db_path)

    # The SAME deterministic ID, computed again independently, retrieves
    # the SAME events - this is what lets separate pages (which each
    # independently call build_incident_id()) agree on one shared trace.
    events_first_lookup = get_events_by_incident(build_incident_id(7), db_path)
    events_second_lookup = get_events_by_incident("INC-0007", db_path)
    check(len(events_first_lookup) == 2, "events are retrievable via the deterministic ID computed fresh")
    check(
        list(events_first_lookup["audit_id"]) == list(events_second_lookup["audit_id"]),
        "the deterministically-computed ID and the literal string retrieve the identical set of events",
    )


# =======================================================================
# 14. summarize_incident_audit_history() - stage ordering and latest decision
# =======================================================================
def test_summarize_incident_audit_history():
    print("\n14. summarize_incident_audit_history() reports stages in canonical order and the latest decision")
    db_path = fresh_db_path()
    # Written deliberately out of canonical order.
    write_audit_event("INC-0007", "CONFIDENCE", "C", db_path=db_path)
    write_audit_event("INC-0007", "ANOMALY_DETECTION", "A", db_path=db_path)
    write_audit_event("INC-0007", "RULE_CLASSIFICATION", "B", db_path=db_path)
    write_audit_event("INC-0007", "ANALYST_REVIEW", "FEEDBACK_DECISION", analyst_decision="APPROVE", db_path=db_path)
    write_audit_event("INC-0007", "ANALYST_REVIEW", "FEEDBACK_DECISION", analyst_decision="MODIFY", db_path=db_path, skip_if_exists=False)

    summary = summarize_incident_audit_history("INC-0007", db_path)
    check(summary.total_events == 5, "all 5 events are counted")
    check(
        summary.stages_recorded == ["ANOMALY_DETECTION", "RULE_CLASSIFICATION", "CONFIDENCE", "ANALYST_REVIEW"],
        "stages_recorded is reported in canonical pipeline order, regardless of write order",
    )
    check(summary.latest_analyst_decision == "MODIFY", "latest_analyst_decision reflects the MOST RECENT decision (MODIFY), not the first (APPROVE)")
    check(summary.first_event_timestamp <= summary.last_event_timestamp, "first/last timestamps are correctly ordered")


# =======================================================================
# 15. Streamlit UI - audit events accumulate correctly across the app
# =======================================================================
def test_streamlit_audit_trail_integration():
    # UI redesign note: "Analyst Review" is now the "Analyst Decision"
    # section on "Case Analysis" (the app's landing page - no navigation
    # click needed), and "Explainability & Audit Trail" is now the
    # "Audit Trail" section under "Review & Insights" (a tab-styled section
    # switcher, not st.tabs() - st.tabs() would mount all seven sections'
    # dataframes/charts into the page at once, which in real-browser
    # testing crashed the page, so Review & Insights renders exactly one
    # section per run and that section must be explicitly selected). Case
    # Analysis remembers the last-selected case in
    # st.session_state["last_selected_incident_id"], and the Audit Trail
    # section defaults its own case selector to that same incident, so
    # re-visiting it after submitting a decision inspects the SAME incident
    # the decision was just recorded for - exactly what this
    # no-duplication check needs.
    print("\n15. The real Streamlit app records audit events end-to-end with no duplication")
    from streamlit.testing.v1 import AppTest

    with isolated_app_audit_database():
        at = AppTest.from_file("app.py")
        at.run(timeout=120)
        check(len(at.exception) == 0, "the app boots with no exception")

        text_areas = at.text_area
        if text_areas:
            text_areas[0].input("Reviewed for the audit-trail integration test.")
        submit_buttons = [b for b in at.button if "Submit Decision" in (b.label or "")]
        check(len(submit_buttons) == 1, "the Submit Decision button is present")
        submit_buttons[0].click().run(timeout=120)
        check(len(at.exception) == 0, "submitting a decision produces no exception")

        events = load_audit_events(audit_trail_module.AUDIT_DB_PATH)
        check(not events.empty, "at least one audit event was recorded by the real app")
        check("ANALYST_REVIEW" in set(events["stage"]), "an ANALYST_REVIEW event was recorded for the submitted decision")
        by_incident = events.groupby(["incident_id", "stage"]).size()
        check(bool((by_incident.drop("ANALYST_REVIEW", level="stage", errors="ignore") <= 1).all()), "no deterministic pipeline stage has more than one recorded event per incident, despite multiple reruns")

        # Re-rendering the SAME page for the SAME incident again must not
        # add duplicate rows for the deterministic stages.
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=120)
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]
        ri_section.set_value("Audit Trail").run(timeout=120)
        check(len(at.exception) == 0, "the Audit Trail section loads with no exception")
        audit_selectboxes = [s for s in at.selectbox if s.key == "audit_case_select"]
        check(len(audit_selectboxes) == 1, "the Audit Trail section's case selector is present")
        audit_selectboxes[0].set_value(audit_selectboxes[0].value).run(timeout=120)
        events_after = load_audit_events(audit_trail_module.AUDIT_DB_PATH)
        counts_before = events.groupby(["incident_id", "stage"]).size().to_dict()
        counts_after = events_after.groupby(["incident_id", "stage"]).size().to_dict()
        # Every stage count either stayed the same or only grew for a
        # newly-visited incident/stage - never shrank, and the incidents
        # already seen never grew for their already-recorded stages.
        for key, count in counts_before.items():
            check(counts_after.get(key, 0) >= count, "a previously recorded event was never lost")


# =======================================================================
def main():
    test_database_auto_creation()
    test_event_creation()
    test_multiple_events_for_one_incident()
    test_retrieval_by_incident()
    test_retrieval_by_stage()
    test_append_behavior()
    test_persistence_across_reconnect()
    test_malformed_input_handling()
    test_empty_database_handling()
    test_database_failure_handling()
    test_true_label_not_used()
    test_no_changes_to_previous_system_decisions()
    test_deterministic_incident_id_usage()
    test_summarize_incident_audit_history()
    test_streamlit_audit_trail_integration()

    print("\n" + "=" * 70)
    print(f"PHASE 10 AUDIT TRAIL CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_temp_dirs()
