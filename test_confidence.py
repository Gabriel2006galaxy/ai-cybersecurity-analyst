"""
test_confidence.py
--------------------
A small, dependency-free validation script for the Phase 4 evidence-based
confidence model (src/confidence.py). Same style as test_preprocessing.py,
test_anomaly_detection.py, and test_rule_engine.py: plain asserts, no
pytest.

This file also folds in a full regression run of the Phase 1, Phase 2, and
Phase 3 test suites (test_rule_engine.py already re-runs Phase 1 + Phase 2
itself, so importing and running its main() cascades through all three),
so a single command verifies the whole pipeline still works after adding
Phase 4:

    python test_confidence.py

Never modifies data/synthetic_security_logs.csv.
"""

import pandas as pd

from config import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    RULE_EVIDENCE_MULTIPLE_RULES,
    RULE_EVIDENCE_NORMAL_BASELINE,
)
from src.anomaly_detector import run_anomaly_detection
from src.confidence import (
    compute_consistency,
    compute_rule_evidence,
    compute_supporting_evidence,
    determine_confidence_level,
    normalize_anomaly_scores,
    run_confidence_scoring,
    score_confidence,
)
from src.data_loader import load_logs
from src.preprocessor import run_preprocessing_pipeline
from src.rule_engine import run_inference, run_rule_engine

PASSED = 0


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


# Same baseline event used by test_rule_engine.py, so records built here
# are directly comparable with that suite's scenarios.
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


def test_normal_event_handled_correctly():
    print("\n1. A normal (non-anomalous) event produces a valid, non-absolute confidence")
    record = run_inference(make_row())
    check(record.incident_type == "Normal Activity", "sanity: Phase 3 classifies this as Normal Activity")

    conf = score_confidence(record, anomaly_score_normalized=0.1)  # clearly unremarkable within its batch
    check(0.0 <= conf.confidence_score <= 1.0, "confidence_score is within [0, 1]")
    check(conf.confidence_level in {"LOW", "MEDIUM", "HIGH"}, "confidence_level is one of the three valid levels")
    check(isinstance(conf.human_review_recommended, bool), "human_review_recommended is a plain bool")
    check(
        conf.confidence_score < 1.0,
        "confidence for a normal event never sounds like total certainty (score < 1.0)",
    )
    check(
        "not sound like certainty" not in conf.uncertainty_reason.lower(),  # sanity: no leaked internal text
        "uncertainty_reason is a clean, analyst-facing sentence",
    )


def test_anomalous_no_rules_lower_confidence():
    print("\n2. Anomalous + zero rules ('bare' anomaly) produces lower evidence/confidence")
    bare_record = run_inference(make_row(anomaly_status="ANOMALOUS"))
    check(bare_record.triggered_rules == [], "sanity: no Phase 3 rule matches this crafted event")

    one_rule_record = run_inference(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    check(len(one_rule_record.triggered_rules) == 1, "sanity: exactly one rule matches this crafted event")

    same_score = 0.5
    bare_conf = score_confidence(bare_record, anomaly_score_normalized=same_score)
    one_rule_conf = score_confidence(one_rule_record, anomaly_score_normalized=same_score)

    check(
        bare_conf.confidence_score < one_rule_conf.confidence_score,
        "a bare anomaly (0 rules) scores lower confidence than an otherwise-identical event with 1 rule",
    )
    check(bare_conf.human_review_recommended is True, "a bare anomaly is always flagged for human review")
    check(
        "insufficient" in bare_conf.uncertainty_reason.lower(),
        "the bare-anomaly uncertainty reason names the evidence as insufficient",
    )


def test_one_rule_event_valid_confidence():
    print("\n3. A one-rule event produces a valid confidence result")
    record = run_inference(make_row(anomaly_status="ANOMALOUS", location_mismatch=1))
    check(len(record.triggered_rules) == 1, "sanity: exactly one rule (RULE-LOC-01) matches")

    conf = score_confidence(record, anomaly_score_normalized=0.6)
    check(0.0 <= conf.confidence_score <= 1.0, "confidence_score is within [0, 1]")
    check(conf.confidence_level in {"LOW", "MEDIUM", "HIGH"}, "confidence_level is valid")
    check(conf.evidence_strength in {"WEAK", "MODERATE", "STRONG"}, "evidence_strength is one of the three valid tiers")
    check(
        any("1 rule" in reason for reason in conf.confidence_reasons),
        "confidence_reasons explicitly mentions the single triggered rule",
    )


def test_multi_rule_event_valid_confidence():
    print("\n4. A multi-rule event produces a valid, appropriately higher rule-evidence score")
    record = run_inference(make_row(anomaly_status="ANOMALOUS", failed_logins=23, data_transferred_mb=800))
    check(len(record.triggered_rules) >= 2, "sanity: at least two rules match this crafted event")

    conf = score_confidence(record, anomaly_score_normalized=0.6)
    check(0.0 <= conf.confidence_score <= 1.0, "confidence_score is within [0, 1]")
    check(
        conf.components["rule_evidence"].value == RULE_EVIDENCE_MULTIPLE_RULES,
        "the rule_evidence component uses the 'multiple rules' tier for 2+ triggered rules",
    )


def test_confidence_score_bounds():
    print("\n5. Confidence score always stays within [0, 1], across many scenarios and the real dataset")
    scenarios = [
        make_row(),
        make_row(anomaly_status="ANOMALOUS"),
        make_row(anomaly_status="ANOMALOUS", failed_logins=23),
        make_row(anomaly_status="ANOMALOUS", failed_logins=23, data_transferred_mb=800, device_count=6),
        make_row(anomaly_status="ANOMALOUS", is_off_hours=1, event_type="privilege_escalation_attempt"),
        make_row(anomaly_status="ANOMALOUS", location_mismatch=1, device_count=6),
    ]
    for norm_score in (0.0, 0.25, 0.5, 0.75, 1.0):
        for row in scenarios:
            record = run_inference(row)
            conf = score_confidence(record, anomaly_score_normalized=norm_score)
            check(
                0.0 <= conf.confidence_score <= 1.0,
                f"confidence_score {conf.confidence_score} is within [0, 1] (norm_score={norm_score})",
            )

    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly = run_anomaly_detection(pre)
    rule_result = run_rule_engine(pre, anomaly)
    confidence_result = run_confidence_scoring(rule_result)
    out_of_bounds = confidence_result.results_df[
        (confidence_result.results_df["confidence_score"] < 0.0)
        | (confidence_result.results_df["confidence_score"] > 1.0)
    ]
    check(out_of_bounds.empty, "every confidence_score on the real 350-event dataset is within [0, 1]")


def test_confidence_level_matches_thresholds():
    print("\n6. Confidence level classification matches the configured thresholds exactly")
    check(determine_confidence_level(0.0) == "LOW", "score 0.0 -> LOW")
    check(determine_confidence_level(LOW_CONFIDENCE_THRESHOLD - 0.01) == "LOW", "just below LOW threshold -> LOW")
    check(determine_confidence_level(LOW_CONFIDENCE_THRESHOLD) == "MEDIUM", "exactly at LOW threshold -> MEDIUM")
    check(
        determine_confidence_level(HIGH_CONFIDENCE_THRESHOLD - 0.01) == "MEDIUM",
        "just below HIGH threshold -> MEDIUM",
    )
    check(determine_confidence_level(HIGH_CONFIDENCE_THRESHOLD) == "HIGH", "exactly at HIGH threshold -> HIGH")
    check(determine_confidence_level(1.0) == "HIGH", "score 1.0 -> HIGH")


def test_low_confidence_triggers_human_review():
    print("\n7. LOW confidence produces human_review_recommended = True; strong HIGH-confidence cases do not")
    bare_record = run_inference(make_row(anomaly_status="ANOMALOUS"))
    low_conf = score_confidence(bare_record, anomaly_score_normalized=0.1)
    check(low_conf.confidence_level == "LOW", "sanity: this crafted case lands in LOW")
    check(low_conf.human_review_recommended is True, "LOW confidence -> human_review_recommended is True")

    strong_record = run_inference(
        make_row(anomaly_status="ANOMALOUS", failed_logins=23, data_transferred_mb=800, device_count=6)
    )
    strong_conf = score_confidence(strong_record, anomaly_score_normalized=0.95)
    check(strong_conf.confidence_level == "HIGH", "sanity: this crafted case lands in HIGH")
    check(
        strong_conf.human_review_recommended is False,
        "a strong, consistent, multi-rule HIGH-confidence case is not auto-flagged for review",
    )


def test_high_severity_low_confidence_allowed():
    print("\n8. HIGH severity + LOW confidence is a valid, supported combination")
    # RULE-PRIV-01 alone is always HIGH severity (see rule_engine.determine_severity),
    # even as the only triggered rule - but a very low batch-relative anomaly score
    # (this event barely crossed the anomalous threshold) keeps confidence LOW.
    record = run_inference(
        make_row(anomaly_status="ANOMALOUS", is_off_hours=1, event_type="privilege_escalation_attempt")
    )
    check(record.severity == "HIGH", "sanity: off-hours privilege escalation is HIGH severity even alone")

    conf = score_confidence(record, anomaly_score_normalized=0.05)
    check(
        conf.confidence_level == "LOW",
        f"a weak anomaly signal keeps confidence LOW even for a HIGH-severity incident (got {conf.confidence_score})",
    )
    check(
        record.severity == "HIGH" and conf.confidence_level == "LOW",
        "severity and confidence are independent - HIGH severity with LOW confidence is valid",
    )
    check(conf.human_review_recommended is True, "this combination is flagged for human review")


def test_true_label_not_used():
    print("\n9-10. true_label is never required by (or influential on) confidence scoring")

    row_no_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23)
    del row_no_label["true_label"]
    record_no_label = run_inference(row_no_label)
    conf_no_label = score_confidence(record_no_label, anomaly_score_normalized=0.6)  # must not raise
    check(0.0 <= conf_no_label.confidence_score <= 1.0, "confidence scoring works with no true_label column at all")

    row_normal_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="normal")
    row_suspicious_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="suspicious")
    conf_normal = score_confidence(run_inference(row_normal_label), anomaly_score_normalized=0.6)
    conf_suspicious = score_confidence(run_inference(row_suspicious_label), anomaly_score_normalized=0.6)
    check(
        conf_normal.confidence_score == conf_suspicious.confidence_score,
        "flipping true_label does not change confidence_score",
    )
    check(
        conf_normal.confidence_level == conf_suspicious.confidence_level,
        "flipping true_label does not change confidence_level",
    )
    check(
        conf_normal.human_review_recommended == conf_suspicious.human_review_recommended,
        "flipping true_label does not change human_review_recommended",
    )

    # Static check, mirroring test_rule_engine.py's approach: config.LABEL_COLUMN
    # is never imported into this module's namespace.
    import src.confidence as confidence_module

    check(
        "LABEL_COLUMN" not in vars(confidence_module),
        "config.LABEL_COLUMN is not imported into confidence.py's namespace",
    )

    import inspect

    source_lines = [
        line for line in inspect.getsource(confidence_module).splitlines()
        if not line.strip().startswith(("#", '"', "'"))
    ]
    code_only = "\n".join(source_lines)
    check(
        '"true_label"]' not in code_only and "'true_label']" not in code_only
        and '.get("true_label"' not in code_only and ".get('true_label'" not in code_only,
        "true_label is never accessed as a dict/Series key in any actual code line",
    )


def test_deterministic_repeated_calls():
    print("\n11. Same input produces the same confidence result on repeated calls")
    record = run_inference(make_row(anomaly_status="ANOMALOUS", failed_logins=23, location_mismatch=1))
    results = {score_confidence(record, anomaly_score_normalized=0.66).confidence_score for _ in range(20)}
    check(len(results) == 1, "20 repeated calls on the same input produce the same confidence_score")

    levels = {score_confidence(record, anomaly_score_normalized=0.66).confidence_level for _ in range(20)}
    check(len(levels) == 1, "20 repeated calls on the same input produce the same confidence_level")


def test_missing_optional_fields_handled_safely():
    print("\n12. Missing optional evidence fields are handled safely")

    minimal = pd.Series({"anomaly_status": "ANOMALOUS", "failed_logins": 23})
    record = run_inference(minimal)  # Phase 3 already handles this; confirm Phase 4 does too
    conf = score_confidence(record, anomaly_score_normalized=0.5)  # must not raise
    check(0.0 <= conf.confidence_score <= 1.0, "confidence scoring works from a minimal row with most fields missing")

    no_score_record = run_inference(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    del no_score_record.evidence["anomaly_score"]
    conf2 = score_confidence(no_score_record, anomaly_score_normalized=0.5)  # supporting-indicator text formats anomaly_score
    check(0.0 <= conf2.confidence_score <= 1.0, "a missing anomaly_score in evidence does not break indicator text")

    conf3 = score_confidence(record, anomaly_score_normalized=None)  # no batch context supplied at all
    check(0.0 <= conf3.confidence_score <= 1.0, "a missing anomaly_score_normalized falls back to the neutral midpoint safely")


def test_empty_input_handled_safely():
    print("\n13. Empty input is handled safely")

    empty_row = pd.Series(dtype=object)
    empty_record = run_inference(empty_row)
    conf = score_confidence(empty_record, anomaly_score_normalized=0.5)  # must not raise
    check(0.0 <= conf.confidence_score <= 1.0, "a completely empty row still produces a valid confidence result")
    check(conf.confidence_level in {"LOW", "MEDIUM", "HIGH"}, "confidence_level is valid for the empty-row case")

    empty_scores = normalize_anomaly_scores(pd.Series([], dtype=float))
    check(empty_scores.empty, "normalizing an empty Series of anomaly scores does not raise")

    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly = run_anomaly_detection(pre)

    class _EmptyFeatureResult:
        feature_df = pd.DataFrame()

    empty_rule_result = run_rule_engine(_EmptyFeatureResult(), anomaly)
    empty_confidence_result = run_confidence_scoring(empty_rule_result)
    check(empty_confidence_result.results_df.empty, "an empty rule_result produces an empty (not crashing) ConfidenceEngineResult")
    check(empty_confidence_result.total_events == 0, "total_events is 0 for empty input")
    check(empty_confidence_result.human_review_count == 0, "human_review_count is 0 for empty input")


def test_consistency_and_supporting_helpers_directly():
    print("\n(extra) Consistency and supporting-evidence helper functions behave as documented")
    # Anomalous, 0 rules, 0 supporting indicators -> LOW consistency ("bare" anomaly).
    value, tier = compute_consistency(anomalous=True, rule_count=0, supporting_count=0)
    check(tier == "low", "anomalous + 0 rules + 0 supporting indicators -> LOW consistency tier")

    # Anomalous, 1+ rule, 2+ supporting indicators -> HIGH consistency.
    value, tier = compute_consistency(anomalous=True, rule_count=1, supporting_count=2)
    check(tier == "high", "anomalous + >=1 rule + >=2 supporting indicators -> HIGH consistency tier")

    # Normal, 0 supporting indicators -> HIGH consistency (clean agreement).
    value, tier = compute_consistency(anomalous=False, rule_count=0, supporting_count=0)
    check(tier == "high", "normal + 0 supporting indicators -> HIGH consistency tier")

    # Normal, 2+ supporting indicators despite Normal classification -> LOW (conflicting).
    value, tier = compute_consistency(anomalous=False, rule_count=0, supporting_count=2)
    check(tier == "low", "normal + 2+ supporting indicators present -> LOW consistency tier (conflicting evidence)")

    evidence, count = compute_supporting_evidence(anomalous=True, facts={"high_failed_logins": True})
    check(count == 1, "compute_supporting_evidence counts exactly the True supporting facts")
    check(0.0 <= evidence <= 1.0, "supporting evidence ratio is within [0, 1]")

    rule_evidence_normal = compute_rule_evidence(anomalous=False, triggered_rules=[])
    check(
        rule_evidence_normal == RULE_EVIDENCE_NORMAL_BASELINE,
        "rule evidence for a Normal Activity event uses the fixed neutral baseline",
    )


def test_full_pipeline_and_batch_normalization():
    print("\n(extra) Full pipeline integration + batch-relative normalization sanity")
    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly = run_anomaly_detection(pre)
    rule_result = run_rule_engine(pre, anomaly)
    confidence_result = run_confidence_scoring(rule_result)

    check(confidence_result.total_events == 350, "all 350 processed events receive a confidence score")
    check(
        confidence_result.high_confidence_count
        + confidence_result.medium_confidence_count
        + confidence_result.low_confidence_count
        == confidence_result.total_events,
        "LOW + MEDIUM + HIGH confidence counts add up to the total",
    )
    for col in [
        "confidence_score", "confidence_level", "evidence_strength", "supporting_indicators",
        "confidence_reasons", "uncertainty_reason", "human_review_recommended",
    ]:
        check(col in confidence_result.results_df.columns, f"results_df contains '{col}'")

    normalized = normalize_anomaly_scores(confidence_result.results_df["anomaly_score"])
    check(normalized.min() >= 0.0 and normalized.max() <= 1.0, "batch-normalized anomaly scores stay within [0, 1]")
    check(abs(normalized.max() - 1.0) < 1e-9, "the single most anomalous-looking event normalizes to (near) 1.0")
    check(abs(normalized.min() - 0.0) < 1e-9, "the single least anomalous-looking event normalizes to (near) 0.0")


def test_streamlit_confidence_page_runs():
    # UI redesign note: "Uncertainty & Confidence" is no longer a
    # standalone page - confidence scoring now happens automatically for
    # every case (both app areas call the same src.confidence.run_
    # confidence_scoring this test exercises directly above), and its
    # per-event confidence breakdown now lives in Case Analysis's "Why
    # Was This Flagged? -> Technical details" expander and in the
    # "Explainability" section under "Review & Insights" (the dataset-wide
    # "Total Events" summary lives in that same page's "Evaluation /
    # Performance" section). Review & Insights renders exactly ONE section
    # per run (a tab-styled section switcher, not st.tabs() - st.tabs()
    # would mount all seven sections' dataframes/charts into the page at
    # once, which in real-browser testing crashed the page), so the two
    # sections below are each checked on their own AppTest instance.
    print("\n17. Streamlit 'Review & Insights' page runs without exceptions (confidence)")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("Review & Insights").run(timeout=60)
    ri_section = [r for r in at.radio if r.key == "ri_section"][0]
    ri_section.set_value("Evaluation / Performance").run(timeout=60)
    check(len(at.exception) == 0, "page renders with no exception (auto-runs anomaly detection + confidence first)")
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


def run_phase1_phase2_phase3_regression():
    print("\n" + "=" * 70)
    print("14-16. Regression: re-running the Phase 1, Phase 2, and Phase 3 test suites")
    print("=" * 70)
    import test_rule_engine  # its own main() already re-runs Phase 1 + Phase 2

    test_rule_engine.main()


def main():
    print("=" * 70)
    print("Phase 4 confidence model test suite")
    print("=" * 70)

    test_normal_event_handled_correctly()
    test_anomalous_no_rules_lower_confidence()
    test_one_rule_event_valid_confidence()
    test_multi_rule_event_valid_confidence()
    test_confidence_score_bounds()
    test_confidence_level_matches_thresholds()
    test_low_confidence_triggers_human_review()
    test_high_severity_low_confidence_allowed()
    test_true_label_not_used()
    test_deterministic_repeated_calls()
    test_missing_optional_fields_handled_safely()
    test_empty_input_handled_safely()
    test_consistency_and_supporting_helpers_directly()
    test_full_pipeline_and_batch_normalization()
    test_streamlit_confidence_page_runs()

    print("\n" + "=" * 70)
    print(f"PHASE 4 CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)

    run_phase1_phase2_phase3_regression()

    print("\n" + "=" * 70)
    print("ALL PHASE 1 + PHASE 2 + PHASE 3 + PHASE 4 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
