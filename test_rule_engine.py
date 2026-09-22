"""
test_rule_engine.py
---------------------
A small, dependency-free validation script for the Phase 3 knowledge-based
rule engine (src/rule_engine.py). Same style as test_preprocessing.py and
test_anomaly_detection.py: plain asserts, no pytest.

This file also folds in a full regression run of the Phase 1 and Phase 2
test suites (imported and executed directly), so a single command verifies
the whole pipeline still works after adding Phase 3:

    python test_rule_engine.py

Never modifies data/synthetic_security_logs.csv.
"""

import pandas as pd

from config import DATA_TRANSFER_THRESHOLD, SEVERE_DATA_TRANSFER_THRESHOLD
from src.anomaly_detector import run_anomaly_detection
from src.data_loader import load_logs
from src.preprocessor import run_preprocessing_pipeline
from src.rule_engine import (
    RULES,
    RULES_BY_ID,
    classify_incident,
    determine_severity,
    evaluate_rules,
    extract_facts,
    run_inference,
    run_rule_engine,
)

PASSED = 0


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


# A baseline "ordinary" event - every test below overrides only the fields
# that matter for that specific rule, so each test isolates one signal.
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


def make_row(**overrides) -> pd.Series:
    data = dict(BASE_ROW)
    data.update(overrides)
    return pd.Series(data)


def rule_ids(triggered) -> set:
    return {tr.rule_id for tr in triggered}


def test_normal_event():
    print("\n1. A normal (non-anomalous) event produces Normal Activity")
    record = run_inference(make_row())
    check(record.incident_type == "Normal Activity", "incident_type is 'Normal Activity'")
    check(record.severity == "N/A", "severity is 'N/A' for a non-anomalous event")
    check(record.triggered_rules == [], "no rules trigger on a non-anomalous event")


def test_brute_force_rule():
    print("\n2. High failed-login event triggers the brute-force rule (RULE-BF-01)")
    record = run_inference(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    check("RULE-BF-01" in rule_ids(record.triggered_rules), "RULE-BF-01 fires")
    check(record.incident_type == "Possible Brute Force", "incident_type is 'Possible Brute Force'")
    check("high_failed_logins" in record.derived_facts, "'high_failed_logins' fact derived")


def test_network_activity_rule():
    print("\n3. High request-count event triggers the network-anomaly rule (RULE-NET-01)")
    # port=443 (common) and event_type not a scan, so ONLY high_request_count
    # drives this - isolates RULE-NET-01 from RULE-SCAN-01.
    record = run_inference(
        make_row(anomaly_status="ANOMALOUS", request_count=500, port=443, event_type="login_success")
    )
    check("RULE-NET-01" in rule_ids(record.triggered_rules), "RULE-NET-01 fires")
    check("RULE-SCAN-01" not in rule_ids(record.triggered_rules), "RULE-SCAN-01 does NOT fire (port is common)")
    check(record.incident_type == "Unusual Network Activity", "incident_type is 'Unusual Network Activity'")


def test_data_transfer_rule():
    print("\n4. Large data transfer triggers the suspicious-data-transfer rule (RULE-DATA-01)")
    record = run_inference(
        make_row(anomaly_status="ANOMALOUS", data_transferred_mb=DATA_TRANSFER_THRESHOLD + 50)
    )
    check("RULE-DATA-01" in rule_ids(record.triggered_rules), "RULE-DATA-01 fires")
    check(record.incident_type == "Suspicious Data Transfer", "incident_type is 'Suspicious Data Transfer'")
    check(record.severity == "MEDIUM", "moderate transfer -> MEDIUM severity (below the 'severe' threshold)")

    severe_record = run_inference(
        make_row(anomaly_status="ANOMALOUS", data_transferred_mb=SEVERE_DATA_TRANSFER_THRESHOLD + 50)
    )
    check(severe_record.severity == "HIGH", "transfer above SEVERE_DATA_TRANSFER_THRESHOLD -> HIGH severity")


def test_location_mismatch_rule():
    print("\n5. Location mismatch triggers the suspicious-login rule (RULE-LOC-01)")
    record = run_inference(make_row(anomaly_status="ANOMALOUS", location_mismatch=1))
    check("RULE-LOC-01" in rule_ids(record.triggered_rules), "RULE-LOC-01 fires")
    check(record.incident_type == "Suspicious Login", "incident_type is 'Suspicious Login'")
    check(record.severity == "MEDIUM", "a single non-high-risk rule match -> MEDIUM severity")


def test_off_hours_privilege_rule():
    print("\n6. Off-hours privilege escalation triggers its rule (RULE-PRIV-01)")
    record = run_inference(
        make_row(anomaly_status="ANOMALOUS", is_off_hours=1, event_type="privilege_escalation_attempt")
    )
    check("RULE-PRIV-01" in rule_ids(record.triggered_rules), "RULE-PRIV-01 fires")
    check(record.incident_type == "Other Anomaly", "incident_type is 'Other Anomaly'")
    check(record.severity == "HIGH", "off-hours privilege escalation is always HIGH severity, even alone")

    # And: privilege escalation event_type WITHOUT off-hours should NOT fire it.
    daytime_record = run_inference(
        make_row(anomaly_status="ANOMALOUS", is_off_hours=0, event_type="privilege_escalation_attempt")
    )
    check(
        "RULE-PRIV-01" not in rule_ids(daytime_record.triggered_rules),
        "RULE-PRIV-01 does NOT fire without off_hours_activity (AND, not OR)",
    )


def test_multi_device_rule():
    print("\n7. Multiple-device activity triggers its rule (RULE-DEV-01)")
    record = run_inference(make_row(anomaly_status="ANOMALOUS", device_count=6))
    check("RULE-DEV-01" in rule_ids(record.triggered_rules), "RULE-DEV-01 fires")
    check(record.incident_type == "Other Anomaly", "incident_type is 'Other Anomaly'")


def test_port_scanning_rule():
    print("\n8. Port scanning behavior triggers the correct rule (RULE-SCAN-01)")
    scan_record = run_inference(make_row(anomaly_status="ANOMALOUS", event_type="network_scan_detected"))
    check("RULE-SCAN-01" in rule_ids(scan_record.triggered_rules), "RULE-SCAN-01 fires via network_scan_indicator")
    check(scan_record.incident_type == "Port Scanning Indicator", "incident_type is 'Port Scanning Indicator'")

    # Second OR-branch: unusual port AND high request count, without a scan event_type.
    port_record = run_inference(
        make_row(anomaly_status="ANOMALOUS", event_type="login_success", port=31337, request_count=500)
    )
    check(
        "RULE-SCAN-01" in rule_ids(port_record.triggered_rules),
        "RULE-SCAN-01 also fires via (unusual_port AND high_request_count)",
    )

    # Neither branch alone (just an unusual port, low request count) must NOT fire it.
    quiet_record = run_inference(
        make_row(anomaly_status="ANOMALOUS", event_type="login_success", port=31337, request_count=10)
    )
    check(
        "RULE-SCAN-01" not in rule_ids(quiet_record.triggered_rules),
        "an unusual port alone (without high request_count or a scan event_type) does NOT fire RULE-SCAN-01",
    )


def test_multiple_rules_same_event():
    print("\n9. Multiple rules can trigger on the same event")
    record = run_inference(
        make_row(anomaly_status="ANOMALOUS", failed_logins=23, data_transferred_mb=800)
    )
    check(len(record.triggered_rules) >= 2, "at least two rules fire on the same event")
    check(
        {"RULE-BF-01", "RULE-DATA-01"} <= rule_ids(record.triggered_rules),
        "both RULE-BF-01 and RULE-DATA-01 are present (neither is skipped once one matches)",
    )
    check(
        record.incident_type == "Possible Brute Force",
        "deterministic priority picks 'Possible Brute Force' over 'Suspicious Data Transfer'",
    )
    check(record.severity == "HIGH", "2+ triggered rules -> HIGH severity")


def test_classification_is_deterministic():
    print("\n10. Incident classification is deterministic")
    row = make_row(anomaly_status="ANOMALOUS", failed_logins=23, location_mismatch=1)
    results = {run_inference(row).incident_type for _ in range(20)}
    check(len(results) == 1, "20 repeated runs on the same input produce the same incident_type")

    facts = extract_facts(row)
    triggered = evaluate_rules(facts, row)
    check(
        classify_incident(True, triggered) == classify_incident(True, triggered),
        "classify_incident() is a pure function of its inputs",
    )


def test_severity_is_deterministic():
    print("\n11. Initial severity is deterministic")
    row = make_row(anomaly_status="ANOMALOUS", device_count=6)
    results = {run_inference(row).severity for _ in range(20)}
    check(len(results) == 1, "20 repeated runs on the same input produce the same severity")

    facts = extract_facts(row)
    triggered = evaluate_rules(facts, row)
    check(
        determine_severity(True, triggered, row) == determine_severity(True, triggered, row),
        "determine_severity() is a pure function of its inputs",
    )


def test_true_label_not_used():
    print("\n12. true_label is never required by (or influential on) the rule engine")

    row_no_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23)
    del row_no_label["true_label"]
    record = run_inference(row_no_label)  # must not raise
    check(record.incident_type == "Possible Brute Force", "rule engine works with no true_label column at all")

    # Behavioral proof, not just a missing-column check: flipping true_label
    # must not change the outcome.
    row_normal_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="normal")
    row_suspicious_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="suspicious")
    r1 = run_inference(row_normal_label)
    r2 = run_inference(row_suspicious_label)
    check(r1.incident_type == r2.incident_type, "changing true_label does not change incident_type")
    check(r1.severity == r2.severity, "changing true_label does not change severity")

    # Static check: config.LABEL_COLUMN is not bound anywhere in the
    # module's namespace (it is never `from config import LABEL_COLUMN`'d),
    # so it's structurally unavailable to use - not just "unused by
    # convention". Doc comments are free to mention it by name to explain
    # the exclusion; that's expected and desirable.
    import src.rule_engine as rule_engine_module

    check(
        "LABEL_COLUMN" not in vars(rule_engine_module),
        "config.LABEL_COLUMN is not imported into rule_engine.py's namespace",
    )

    import inspect

    source_lines = [
        line for line in inspect.getsource(rule_engine_module).splitlines()
        if not line.strip().startswith(("#", '"', "'"))
    ]
    code_only = "\n".join(source_lines)
    check(
        '"true_label"]' not in code_only and "'true_label']" not in code_only
        and '.get("true_label"' not in code_only and ".get('true_label'" not in code_only,
        "true_label is never accessed as a dict/Series key in any actual code line",
    )


def test_missing_optional_fields_handled_safely():
    print("\n13. Missing optional fields are handled safely")

    minimal = pd.Series({"anomaly_status": "ANOMALOUS", "failed_logins": 23})
    record = run_inference(minimal)  # must not raise, despite missing most columns
    check(record.incident_type == "Possible Brute Force", "still classifies correctly from a minimal row")

    no_score = make_row(anomaly_status="ANOMALOUS", failed_logins=23)
    del no_score["anomaly_score"]
    record2 = run_inference(no_score)  # explanation string formats anomaly_score - must not crash
    check(record2.incident_type == "Possible Brute Force", "missing anomaly_score does not break the explanation text")

    no_transfer = make_row(anomaly_status="ANOMALOUS", data_transferred_mb=DATA_TRANSFER_THRESHOLD + 10)
    del no_transfer["data_transferred_mb"]
    facts = extract_facts(no_transfer)
    check(facts["high_data_transfer"] is False, "a missing data_transferred_mb defaults to 'not high' rather than crashing")


def test_empty_input_handled_safely():
    print("\n14. Empty input is handled safely")

    empty_row = pd.Series(dtype=object)
    record = run_inference(empty_row)  # must not raise
    check(record.incident_type == "Normal Activity", "a completely empty row is treated as Normal Activity")
    check(record.severity == "N/A", "severity is 'N/A' for the empty row")

    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly = run_anomaly_detection(pre)

    class _EmptyFeatureResult:
        feature_df = pd.DataFrame()

    empty_result = run_rule_engine(_EmptyFeatureResult(), anomaly)
    check(empty_result.results_df.empty, "an empty feature_df produces an empty (not crashing) RuleEngineResult")
    check(empty_result.total_events == 0, "total_events is 0 for empty input")


def test_full_pipeline_and_rule_registry():
    print("\n(extra) Full pipeline integration + rule registry sanity")
    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly = run_anomaly_detection(pre)
    result = run_rule_engine(pre, anomaly)

    check(len(RULES) == 7, "exactly 7 rules are registered")
    check(len(RULES_BY_ID) == 7, "all 7 rule IDs are unique")
    check(result.total_events == 350, "all 350 processed events are classified")
    check(
        result.normal_count + (result.total_events - result.normal_count) == result.total_events,
        "normal + anomalous counts add up to the total",
    )
    check(
        result.low_count + result.medium_count + result.high_count
        == result.total_events - result.normal_count,
        "LOW + MEDIUM + HIGH severities account for exactly the anomalous events",
    )
    for col in ["incident_type", "severity", "triggered_rule_ids", "derived_facts", "reason", "evidence"]:
        check(col in result.results_df.columns, f"results_df contains '{col}'")
    # At least one real event exercises each rule on the actual dataset.
    fired_rule_ids = set()
    for ids in result.results_df["triggered_rule_ids"]:
        fired_rule_ids.update(ids)
    check(
        fired_rule_ids == set(RULES_BY_ID.keys()),
        f"every rule fires at least once on the real dataset (fired: {sorted(fired_rule_ids)})",
    )


def test_streamlit_page_runs():
    # UI redesign note: "Incident Classification" is no longer a
    # standalone page - rule-engine classification now happens
    # automatically for every case (Case Analysis and Review & Insights
    # both call the same src.rule_engine.run_rule_engine this test
    # exercises directly above), and its per-event inference detail now
    # lives in the "Explainability" tab under "Review & Insights" (the
    # dataset-wide "Total Events" summary lives in that same page's
    # "Evaluation / Performance" section. Review & Insights renders exactly
    # ONE section per run (a tab-styled section switcher, not st.tabs() -
    # st.tabs() would mount all seven sections' dataframes/charts into the
    # page at once, which in real-browser testing crashed the page), so
    # the two sections below (Evaluation / Performance, Explainability) are
    # each checked on their own AppTest instance.
    print("\n17. Streamlit 'Review & Insights' page runs without exceptions (rule engine)")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("Review & Insights").run(timeout=60)
    ri_section = [r for r in at.radio if r.key == "ri_section"][0]
    ri_section.set_value("Evaluation / Performance").run(timeout=60)
    check(len(at.exception) == 0, "page renders with no exception (auto-runs anomaly detection + rule engine first)")
    check(
        any("Total Events" == m.label for m in at.metric),
        "Summary metrics are shown",
    )

    at2 = AppTest.from_file("app.py")
    at2.run(timeout=30)
    at2.sidebar.radio[0].set_value("Review & Insights").run(timeout=60)
    ri_section2 = [r for r in at2.radio if r.key == "ri_section"][0]
    ri_section2.set_value("Explainability").run(timeout=60)
    check(len(at2.exception) == 0, "the Explainability section renders with no exception")
    event_select = [s for s in at2.selectbox if s.label == "Case" and s.key == "explain_case_select"][0]
    event_select.set_value(0).run(timeout=30)
    check(len(at2.exception) == 0, "selecting an event to inspect raises no exception")


def run_phase1_and_phase2_regression():
    print("\n" + "=" * 70)
    print("15-16. Regression: re-running the Phase 1 and Phase 2 test suites")
    print("=" * 70)
    import test_anomaly_detection
    import test_preprocessing

    test_preprocessing.main()
    test_anomaly_detection.main()


def main():
    print("=" * 70)
    print("Phase 3 rule engine test suite")
    print("=" * 70)

    test_normal_event()
    test_brute_force_rule()
    test_network_activity_rule()
    test_data_transfer_rule()
    test_location_mismatch_rule()
    test_off_hours_privilege_rule()
    test_multi_device_rule()
    test_port_scanning_rule()
    test_multiple_rules_same_event()
    test_classification_is_deterministic()
    test_severity_is_deterministic()
    test_true_label_not_used()
    test_missing_optional_fields_handled_safely()
    test_empty_input_handled_safely()
    test_full_pipeline_and_rule_registry()
    test_streamlit_page_runs()

    print("\n" + "=" * 70)
    print(f"PHASE 3 CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)

    run_phase1_and_phase2_regression()

    print("\n" + "=" * 70)
    print("ALL PHASE 1 + PHASE 2 + PHASE 3 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
