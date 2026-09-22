"""
test_response_planner.py
--------------------------
A small, dependency-free validation script for the Phase 6 defensive
response planner (src/response_planner.py). Same style as
test_preprocessing.py, test_anomaly_detection.py, test_rule_engine.py,
test_confidence.py, and test_genai.py: plain asserts, no pytest.

This file also folds in a full regression run of the Phase 1-5 test
suites (test_genai.py already re-runs Phase 1-4 itself), so a single
command verifies the whole pipeline still works after adding Phase 6:

    python test_response_planner.py

Two styles of test input are used on purpose:
  - Several tests build InferenceRecord/ConfidenceRecord objects DIRECTLY
    (e.g. `InferenceRecord(incident_type=..., severity=...)`), to exercise
    every branch of the conditional planner precisely and independently
    of the real confidence formula.
  - Several other tests run the REAL Phase 3 (`run_inference`) and Phase 4
    (`score_confidence`) pipeline on crafted rows, and one test runs the
    full pipeline on the real 350-event dataset - proving the planner
    integrates correctly with genuinely computed upstream results, not
    just hand-built ones.

Never modifies data/synthetic_security_logs.csv.
"""

import dataclasses

import pandas as pd

from src.anomaly_detector import run_anomaly_detection
from src.confidence import ConfidenceRecord, score_confidence
from src.data_loader import load_logs
from src.genai_analyzer import GenAIResult, build_incident_context, generate_fallback_summary
from src.preprocessor import run_preprocessing_pipeline
from src.response_planner import (
    INCIDENT_CORE_STEPS,
    PLAN_STAGE_NAMES,
    PlanStep,
    PlanValidationResult,
    ResponsePlanResult,
    generate_response_plan,
    run_response_planning,
    validate_plan,
)
from src.rule_engine import InferenceRecord, run_inference, run_rule_engine

PASSED = 0


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


# Same baseline event used by test_rule_engine.py / test_confidence.py / test_genai.py.
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


def make_pair(row: pd.Series, anomaly_score_normalized: float = 0.6):
    """Real Phase 3 + Phase 4 objects, built exactly the way the app does."""
    inference_record = run_inference(row)
    confidence_record = score_confidence(inference_record, anomaly_score_normalized=anomaly_score_normalized)
    return inference_record, confidence_record


def core_actions(incident_type: str):
    return [action for action, _ in INCIDENT_CORE_STEPS[incident_type]]


# =======================================================================
# 1. Normal Activity
# =======================================================================
def test_normal_activity_produces_no_plan():
    print("\n1. Normal Activity produces NO incident-response plan")
    inf, conf = make_pair(make_row())  # base row is NORMAL, no anomaly
    check(inf.incident_type == "Normal Activity", "sanity: the base row classifies as Normal Activity")
    plan = generate_response_plan(inf, conf)
    check(plan.steps == [], "Normal Activity produces an empty steps list")
    check(plan.priority == "ROUTINE", "Normal Activity plan priority is ROUTINE")
    check(plan.review_required is False, "Normal Activity plan never requires human review")
    check(plan.validation_status.is_valid, "Normal Activity plan passes its own validation")
    check(
        plan.included_stages == ["DETECTED", "MONITOR_OR_ESCALATE"],
        "Normal Activity only reaches the DETECTED and MONITOR_OR_ESCALATE stages",
    )


# =======================================================================
# 2-7. The six incident-specific plan templates (real Phase 3/4 objects)
# =======================================================================
def test_possible_brute_force_plan_template():
    print("\n2. 'Possible Brute Force' plan matches its required template")
    inf, conf = make_pair(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    check(inf.incident_type == "Possible Brute Force", "sanity: crafted row classifies as Possible Brute Force")
    plan = generate_response_plan(inf, conf)
    actions = [s.action for s in plan.steps[:3]]
    check(actions == core_actions("Possible Brute Force"), "the first 3 steps match the Possible Brute Force template, in order")


def test_suspicious_login_plan_template():
    print("\n3. 'Suspicious Login' plan matches its required template")
    inf, conf = make_pair(make_row(anomaly_status="ANOMALOUS", location_mismatch=1))
    check(inf.incident_type == "Suspicious Login", "sanity: crafted row classifies as Suspicious Login")
    plan = generate_response_plan(inf, conf)
    actions = [s.action for s in plan.steps[:3]]
    check(actions == core_actions("Suspicious Login"), "the first 3 steps match the Suspicious Login template, in order")


def test_suspicious_data_transfer_plan_template():
    print("\n4. 'Suspicious Data Transfer' plan matches its required template")
    inf, conf = make_pair(make_row(anomaly_status="ANOMALOUS", data_transferred_mb=200))
    check(inf.incident_type == "Suspicious Data Transfer", "sanity: crafted row classifies as Suspicious Data Transfer")
    plan = generate_response_plan(inf, conf)
    actions = [s.action for s in plan.steps[:3]]
    check(actions == core_actions("Suspicious Data Transfer"), "the first 3 steps match the Suspicious Data Transfer template, in order")


def test_port_scanning_indicator_plan_template():
    print("\n5. 'Port Scanning Indicator' plan matches its required template")
    inf, conf = make_pair(make_row(anomaly_status="ANOMALOUS", event_type="network_scan_detected"))
    check(inf.incident_type == "Port Scanning Indicator", "sanity: crafted row classifies as Port Scanning Indicator")
    plan = generate_response_plan(inf, conf)
    actions = [s.action for s in plan.steps[:3]]
    check(actions == core_actions("Port Scanning Indicator"), "the first 3 steps match the Port Scanning Indicator template, in order")


def test_unusual_network_activity_plan_template():
    print("\n6. 'Unusual Network Activity' plan matches its required template")
    inf, conf = make_pair(make_row(anomaly_status="ANOMALOUS", request_count=300))
    check(inf.incident_type == "Unusual Network Activity", "sanity: crafted row classifies as Unusual Network Activity")
    plan = generate_response_plan(inf, conf)
    actions = [s.action for s in plan.steps[:3]]
    check(actions == core_actions("Unusual Network Activity"), "the first 3 steps match the Unusual Network Activity template, in order")


def test_other_anomaly_plan_template():
    print("\n7. 'Other Anomaly' plan matches its required template (anomalous, but no rule fired)")
    inf, conf = make_pair(make_row(anomaly_status="ANOMALOUS"))  # nothing else pushed past a threshold
    check(inf.incident_type == "Other Anomaly", "sanity: an anomalous event with no rule match classifies as Other Anomaly")
    check(inf.severity == "LOW", "sanity: anomalous + no rule match is LOW severity by Phase 3's own design")
    plan = generate_response_plan(inf, conf)
    actions = [s.action for s in plan.steps[:3]]
    check(actions == core_actions("Other Anomaly"), "the first 3 steps match the Other Anomaly template, in order")


def test_unknown_incident_type_falls_back_safely():
    print("\n8. An unrecognized incident_type falls back to the Other Anomaly template instead of crashing")
    inf = InferenceRecord(incident_type="Something Not In The Knowledge Base", severity="HIGH")
    conf = ConfidenceRecord(confidence_level="MEDIUM", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)  # must not raise
    actions = [s.action for s in plan.steps[:3]]
    check(actions == core_actions("Other Anomaly"), "an unrecognized incident_type silently reuses the Other Anomaly template")
    check(plan.validation_status.is_valid, "the fallback plan still passes validation")


# =======================================================================
# 9-11. Severity-aware planning
# =======================================================================
def test_high_severity_sets_priority_and_policy_step():
    print("\n9. HIGH severity sets base priority HIGH and includes a policy-based recommendation step")
    inf = InferenceRecord(incident_type="Possible Brute Force", severity="HIGH")
    conf = ConfidenceRecord(confidence_level="MEDIUM", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    check(plan.priority == "HIGH", "HIGH severity + non-LOW confidence keeps the base priority at HIGH")
    check(
        any(s.action == "Recommend containment or escalation according to organizational policy" for s in plan.steps),
        "a policy-based recommendation step is included for HIGH severity",
    )


def test_medium_severity_sets_priority_and_policy_step():
    print("\n10. MEDIUM severity sets base priority STANDARD and still includes a policy-based recommendation step")
    inf = InferenceRecord(incident_type="Suspicious Login", severity="MEDIUM")
    conf = ConfidenceRecord(confidence_level="HIGH", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    check(plan.priority == "STANDARD", "MEDIUM severity + HIGH confidence keeps the base priority at STANDARD")
    check(
        any(s.action == "Recommend containment or escalation according to organizational policy" for s in plan.steps),
        "a policy-based recommendation step is included for MEDIUM severity too",
    )


def test_low_severity_excludes_policy_step():
    print("\n11. LOW severity sets base priority ROUTINE and EXCLUDES the policy-based recommendation step")
    inf = InferenceRecord(incident_type="Other Anomaly", severity="LOW")
    conf = ConfidenceRecord(confidence_level="HIGH", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    check(plan.priority == "ROUTINE", "LOW severity + HIGH confidence gives base priority ROUTINE")
    check(
        not any(s.action.startswith("Recommend containment") for s in plan.steps),
        "LOW severity prioritizes routine monitoring/documentation - no containment recommendation step",
    )
    check(
        any(s.action.startswith("Continue monitoring") for s in plan.steps),
        "a monitor/escalate step is still present for LOW severity",
    )


# =======================================================================
# 12-14. Confidence-aware planning
# =======================================================================
def test_high_confidence_skips_evidence_collection_step():
    print("\n12. HIGH confidence skips the dedicated evidence-collection step")
    inf = InferenceRecord(incident_type="Suspicious Data Transfer", severity="MEDIUM")
    conf = ConfidenceRecord(confidence_level="HIGH", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    check(
        not any(s.action == "Collect additional evidence" for s in plan.steps),
        "existing evidence is already strong (HIGH confidence), so no dedicated evidence step is added",
    )


def test_medium_confidence_includes_evidence_collection_step():
    print("\n13. MEDIUM confidence includes an evidence-collection step")
    inf = InferenceRecord(incident_type="Suspicious Data Transfer", severity="MEDIUM")
    conf = ConfidenceRecord(confidence_level="MEDIUM", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    evidence_steps = [s for s in plan.steps if s.action == "Collect additional evidence"]
    check(len(evidence_steps) == 1, "MEDIUM confidence adds exactly one evidence-collection step")
    check("MEDIUM" in evidence_steps[0].reason, "the evidence step's reason explains that confidence is MEDIUM")


def test_low_confidence_includes_evidence_and_review_steps():
    print("\n14. LOW confidence includes BOTH an evidence-collection step and a mandatory review step")
    inf = InferenceRecord(incident_type="Suspicious Login", severity="MEDIUM")
    conf = ConfidenceRecord(confidence_level="LOW", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    check(any(s.action == "Collect additional evidence" for s in plan.steps), "LOW confidence adds an evidence-collection step")
    check(
        any(s.action == "Human analyst review required before any high-impact response" for s in plan.steps),
        "LOW confidence alone (even without the human_review flag) adds a mandatory human-review step",
    )
    check(plan.priority == "HIGH", "MEDIUM severity base (STANDARD) is bumped one tier by LOW confidence, to HIGH")


# =======================================================================
# 15-16. The two explicit spec examples: severity and confidence are
# combined into an overall priority WITHOUT being merged into one score.
# =======================================================================
def test_high_severity_low_confidence_is_urgent():
    print("\n15. HIGH severity + LOW confidence -> URGENT priority, urgent analyst attention, no automatic action")
    inf = InferenceRecord(incident_type="Suspicious Data Transfer", severity="HIGH")
    conf = ConfidenceRecord(confidence_level="LOW", human_review_recommended=True)
    plan = generate_response_plan(inf, conf)
    check(plan.priority == "URGENT", "HIGH severity + LOW confidence produces URGENT overall priority")
    check(plan.review_required, "human review is strongly recommended (review_required is True)")
    check(
        any(s.action == "Collect additional evidence" for s in plan.steps),
        "more evidence is explicitly requested",
    )
    check("urgent" in plan.plan_summary.lower(), "the plan summary reflects the URGENT priority")
    check(plan.validation_status.is_valid, "the URGENT HIGH/LOW plan still passes validation")
    all_actions = " ".join(s.action.lower() for s in plan.steps)
    check(
        not any(kw in all_actions for kw in ["block ip", "disable account", "isolate machine", "terminate session"]),
        "no automatic/offensive action appears anywhere in this urgent plan - every step stays advisory",
    )


def test_medium_severity_high_confidence_not_merged_with_high_low_case():
    print("\n16. MEDIUM severity + HIGH confidence gives a normal STANDARD plan - proving severity/confidence aren't merged")
    inf = InferenceRecord(incident_type="Suspicious Data Transfer", severity="MEDIUM")
    conf = ConfidenceRecord(confidence_level="HIGH", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    check(plan.priority == "STANDARD", "MEDIUM severity + HIGH confidence gives a STANDARD (not bumped) priority")
    check(plan.priority != "URGENT", "this normal-investigation case is clearly distinct from the HIGH+LOW URGENT case")
    check(
        not any(s.action == "Collect additional evidence" for s in plan.steps),
        "HIGH confidence means no extra evidence-collection step is needed here",
    )
    check(
        any(s.action.startswith("Recommend containment") for s in plan.steps),
        "MEDIUM severity still gets a policy-based recommendation step, per organizational policy",
    )


# =======================================================================
# 17-18. human_review_recommended flag handling
# =======================================================================
def test_human_review_flag_adds_mandatory_step():
    print("\n17. The human_review_recommended flag alone adds the mandatory review step")
    inf = InferenceRecord(incident_type="Suspicious Login", severity="MEDIUM")
    conf = ConfidenceRecord(confidence_level="HIGH", human_review_recommended=True)  # confidence is HIGH, not LOW
    plan = generate_response_plan(inf, conf)
    review_steps = [s for s in plan.steps if s.action == "Human analyst review required before any high-impact response"]
    check(len(review_steps) == 1, "the flag alone (independent of confidence) triggers the mandatory review step")
    check(review_steps[0].requires_human_review, "the mandatory review step itself requires human review")
    check("confidence model flagged" in review_steps[0].reason.lower(), "the reason names the confidence-model flag as the trigger")


def test_no_mandatory_review_step_when_not_needed():
    print("\n18. No mandatory review step (and no review at all) when the flag is False, confidence isn't LOW, and severity is LOW")
    inf = InferenceRecord(incident_type="Other Anomaly", severity="LOW")
    conf = ConfidenceRecord(confidence_level="HIGH", human_review_recommended=False)
    plan = generate_response_plan(inf, conf)
    check(
        not any(s.action == "Human analyst review required before any high-impact response" for s in plan.steps),
        "no mandatory review step is added when nothing calls for it",
    )
    check(plan.review_required is False, "with LOW severity (no policy step) and no review trigger, review_required is False")


# =======================================================================
# 19-20. Deterministic ordering and repeatability
# =======================================================================
def test_deterministic_step_ordering():
    print("\n19. Steps are always uniquely, contiguously numbered and follow the fixed stage progression")
    inf = InferenceRecord(incident_type="Suspicious Data Transfer", severity="HIGH")
    conf = ConfidenceRecord(confidence_level="LOW", human_review_recommended=True)
    plan = generate_response_plan(inf, conf)
    numbers = [s.step_number for s in plan.steps]
    check(numbers == list(range(1, len(numbers) + 1)), "step numbers are a contiguous 1..N sequence")
    stage_indices = [PLAN_STAGE_NAMES.index(s.stage) for s in plan.steps]
    check(stage_indices == sorted(stage_indices), "each step's stage never moves backward relative to the previous step")


def test_plan_generation_is_repeatable():
    print("\n20. Generating a plan twice from equivalent inputs produces an identical result (no hidden randomness)")
    inf1 = InferenceRecord(incident_type="Port Scanning Indicator", severity="HIGH")
    conf1 = ConfidenceRecord(confidence_level="LOW", human_review_recommended=True)
    inf2 = InferenceRecord(incident_type="Port Scanning Indicator", severity="HIGH")
    conf2 = ConfidenceRecord(confidence_level="LOW", human_review_recommended=True)
    plan1 = generate_response_plan(inf1, conf1)
    plan2 = generate_response_plan(inf2, conf2)
    steps1 = [(s.step_number, s.action, s.purpose, s.reason, s.priority, s.requires_human_review, s.stage) for s in plan1.steps]
    steps2 = [(s.step_number, s.action, s.purpose, s.reason, s.priority, s.requires_human_review, s.stage) for s in plan2.steps]
    check(steps1 == steps2, "two calls with equivalent inputs produce byte-for-byte identical steps")
    check(plan1.plan_title == plan2.plan_title and plan1.priority == plan2.priority, "plan_title and priority are also identical")


# =======================================================================
# 21-24. Plan validation
# =======================================================================
def test_validate_plan_passes_for_every_real_dataset_plan():
    print("\n21. Every plan generated on the real 350-event dataset passes its own validation")
    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly_result = run_anomaly_detection(pre)
    rule_result = run_rule_engine(pre, anomaly_result)
    from src.confidence import run_confidence_scoring

    confidence_result = run_confidence_scoring(rule_result)
    planning_result = run_response_planning(confidence_result)
    invalid = [p for p in planning_result.results_df["_plan"] if not p.validation_status.is_valid]
    check(invalid == [], f"all {planning_result.total_events} real-dataset plans are valid (0 invalid, found {len(invalid)})")
    check(planning_result.total_events == 350, "sanity: the real dataset still has 350 events")


def test_validate_plan_flags_prohibited_action_keywords():
    print("\n22. validate_plan() flags a prohibited offensive/automated action keyword, and never crashes")
    bad_plan = ResponsePlanResult(
        incident_type="Possible Brute Force", severity="HIGH", confidence_level="LOW",
        human_review_recommended=True, plan_title="t", plan_summary="s",
        steps=[PlanStep(1, "Block IP of the attacker immediately", "p", "r", "HIGH", True)],
        priority="URGENT", review_required=True, planning_reason="x",
        validation_status=PlanValidationResult(is_valid=True),
    )
    result = validate_plan(bad_plan)  # must not raise
    check(isinstance(result, PlanValidationResult), "validate_plan() never raises, even on a prohibited action")
    check(not result.is_valid, "a step containing a prohibited action keyword is correctly flagged as invalid")
    check(any("block ip" in e.lower() for e in result.errors), "the specific offending keyword is named in the error")


def test_validate_plan_flags_structural_problems_without_crashing():
    print("\n23. validate_plan() flags structural problems (never crashes)")

    dup_numbers = ResponsePlanResult(
        incident_type="Other Anomaly", severity="LOW", confidence_level="HIGH", human_review_recommended=False,
        plan_title="t", plan_summary="s",
        steps=[PlanStep(1, "Review the anomaly evidence", "p", "r", "ROUTINE", False),
               PlanStep(1, "Document analyst decision and findings", "p", "r", "ROUTINE", False)],
        priority="ROUTINE", review_required=False, planning_reason="x", validation_status=PlanValidationResult(is_valid=True),
    )
    check(not validate_plan(dup_numbers).is_valid, "duplicate step numbers are flagged as invalid")

    gap_numbers = ResponsePlanResult(
        incident_type="Other Anomaly", severity="LOW", confidence_level="HIGH", human_review_recommended=False,
        plan_title="t", plan_summary="s",
        steps=[PlanStep(1, "Review the anomaly evidence", "p", "r", "ROUTINE", False),
               PlanStep(3, "Document analyst decision and findings", "p", "r", "ROUTINE", False)],
        priority="ROUTINE", review_required=False, planning_reason="x", validation_status=PlanValidationResult(is_valid=True),
    )
    check(not validate_plan(gap_numbers).is_valid, "a non-contiguous step-number sequence is flagged as invalid")

    empty_action = ResponsePlanResult(
        incident_type="Other Anomaly", severity="LOW", confidence_level="HIGH", human_review_recommended=False,
        plan_title="t", plan_summary="s",
        steps=[PlanStep(1, "   ", "p", "r", "ROUTINE", False)],
        priority="ROUTINE", review_required=False, planning_reason="x", validation_status=PlanValidationResult(is_valid=True),
    )
    check(not validate_plan(empty_action).is_valid, "an empty/blank step action is flagged as invalid")

    normal_with_steps = ResponsePlanResult(
        incident_type="Normal Activity", severity="N/A", confidence_level="HIGH", human_review_recommended=False,
        plan_title="t", plan_summary="s",
        steps=[PlanStep(1, "Review the anomaly evidence", "p", "r", "ROUTINE", False)],
        priority="ROUTINE", review_required=False, planning_reason="x", validation_status=PlanValidationResult(is_valid=True),
    )
    check(not validate_plan(normal_with_steps).is_valid, "Normal Activity is flagged as invalid if it ever produces steps")

    incident_without_steps = ResponsePlanResult(
        incident_type="Possible Brute Force", severity="HIGH", confidence_level="HIGH", human_review_recommended=False,
        plan_title="t", plan_summary="s", steps=[],
        priority="ROUTINE", review_required=False, planning_reason="x", validation_status=PlanValidationResult(is_valid=True),
    )
    check(not validate_plan(incident_without_steps).is_valid, "a real incident with zero steps is flagged as invalid")


def test_validate_plan_flags_missing_low_confidence_requirements():
    print("\n24. validate_plan() flags a LOW-confidence incident missing its required evidence/review steps")
    missing_both = ResponsePlanResult(
        incident_type="Possible Brute Force", severity="HIGH", confidence_level="LOW", human_review_recommended=True,
        plan_title="t", plan_summary="s",
        steps=[PlanStep(1, "Review failed-login activity", "p", "r", "HIGH", False)],
        priority="URGENT", review_required=False, planning_reason="x", validation_status=PlanValidationResult(is_valid=True),
    )
    result = validate_plan(missing_both)
    check(not result.is_valid, "a LOW-confidence plan missing evidence/review steps is flagged as invalid")
    check(any("evidence" in e.lower() for e in result.errors), "the missing evidence-collection step is specifically named")
    check(any("review" in e.lower() for e in result.errors), "the missing human-review step is specifically named")


# =======================================================================
# 25. true_label exclusion
# =======================================================================
def test_true_label_excluded_from_planning():
    print("\n25. Planning never reads true_label (structural AND behavioral)")
    import src.response_planner as planner_module

    check("LABEL_COLUMN" not in vars(planner_module), "config.LABEL_COLUMN is not imported into response_planner.py's namespace")

    field_names = {f.name for f in dataclasses.fields(ResponsePlanResult)}
    check("true_label" not in field_names, "ResponsePlanResult has no true_label field at all")

    row_with_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="normal")
    row_flipped_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="suspicious")
    row_no_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23)
    del row_no_label["true_label"]

    plan_with = generate_response_plan(*make_pair(row_with_label))
    plan_flipped = generate_response_plan(*make_pair(row_flipped_label))
    plan_without = generate_response_plan(*make_pair(row_no_label))

    def plan_signature(p):
        return (p.incident_type, p.severity, p.priority, tuple((s.action, s.priority, s.requires_human_review) for s in p.steps))

    check(
        plan_signature(plan_with) == plan_signature(plan_flipped) == plan_signature(plan_without),
        "flipping or removing true_label never changes the generated plan",
    )


# =======================================================================
# 26. Phase 5 GenAI recommendations are supplementary-only and optional
# =======================================================================
def test_genai_supplementary_context_is_advisory_only():
    print("\n26. Phase 5 output is optional and can only ever be supplementary context - never a plan step or a classification change")
    inf, conf = make_pair(make_row(anomaly_status="ANOMALOUS", failed_logins=23))

    plan_no_ai = generate_response_plan(inf, conf, genai_result=None)
    check(plan_no_ai.supplementary_ai_context == [], "with genai_result=None, supplementary_ai_context is an empty list")

    context = build_incident_context(inf, conf)
    real_genai_result = generate_fallback_summary(context)
    check(isinstance(real_genai_result, GenAIResult), "sanity: a real GenAIResult was built for this test")

    plan_with_ai = generate_response_plan(inf, conf, genai_result=real_genai_result)
    check(
        plan_with_ai.supplementary_ai_context == real_genai_result.recommendations,
        "the AI's recommendations are copied verbatim into supplementary_ai_context",
    )
    check(
        [(s.action, s.priority, s.requires_human_review, s.stage) for s in plan_with_ai.steps]
        == [(s.action, s.priority, s.requires_human_review, s.stage) for s in plan_no_ai.steps],
        "attaching a real Phase 5 result does not change a single generated step",
    )
    check(
        plan_with_ai.incident_type == plan_no_ai.incident_type and plan_with_ai.severity == plan_no_ai.severity,
        "attaching a real Phase 5 result does not change incident_type or severity",
    )

    # A deliberately "disagreeing" AI recommendation must still never become a step.
    disagreeing_result = dataclasses.replace(
        real_genai_result, recommendations=["This is actually Normal Activity - no response is needed."]
    )
    plan_with_disagreement = generate_response_plan(inf, conf, genai_result=disagreeing_result)
    check(
        plan_with_disagreement.incident_type == inf.incident_type and plan_with_disagreement.steps,
        "a disagreeing AI recommendation cannot suppress or change the structured plan",
    )
    check(
        all("Normal Activity" not in s.action for s in plan_with_disagreement.steps),
        "the disagreeing AI text never leaks into an actual plan step",
    )
    check(
        disagreeing_result.recommendations == plan_with_disagreement.supplementary_ai_context,
        "the disagreeing text is visible only as read-only supplementary context",
    )


# =======================================================================
# 27. Empty / minimal input
# =======================================================================
def test_empty_incident_input_handled_safely():
    print("\n27. A completely empty/minimal event is handled safely")
    empty_row = pd.Series(dtype=object)
    inf = run_inference(empty_row)  # must not raise
    conf = score_confidence(inf)  # must not raise
    check(inf.incident_type == "Normal Activity", "sanity: an empty row classifies as Normal Activity")
    plan = generate_response_plan(inf, conf)  # must not raise
    check(isinstance(plan, ResponsePlanResult), "an empty/minimal event still produces a valid ResponsePlanResult")
    check(plan.steps == [], "an empty/minimal (Normal Activity) event produces no response-plan steps")
    check(plan.validation_status.is_valid, "the empty-event plan passes validation")

    minimal_row = pd.Series({"anomaly_status": "ANOMALOUS"})
    inf2 = run_inference(minimal_row)
    conf2 = score_confidence(inf2)
    plan2 = generate_response_plan(inf2, conf2)  # must not raise
    check(isinstance(plan2, ResponsePlanResult), "a minimal anomalous row (most fields missing) still produces a valid plan")


# =======================================================================
# 28. Streamlit "Response Planning" page
# =======================================================================
def test_streamlit_response_planning_page_runs():
    # UI redesign note: "Response Planning" is no longer a standalone page
    # - its content is the "Recommended Next Steps" section on "Case
    # Analysis" (the app's landing page), which auto-runs Phases 1-4
    # first exactly as the old page did and calls the exact same
    # src.response_planner.generate_response_plan() this test exercises
    # directly above. The advisory-only warning text was made more
    # prominent (a real st.warning banner, not just a caption) and keeps
    # the same "does not execute ... against a real system" substance.
    print("\n28. Streamlit 'Case Analysis' page runs without exceptions (response plan)")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    check(len(at.exception) == 0, "the page renders with no exception (auto-runs Phases 1-4 first)")
    check(
        any("does not execute any action against a real system" in w.value for w in at.warning),
        "the required advisory-only statement is shown on the page",
    )
    check(any(sh.value == "Recommended Next Steps" for sh in at.subheader), "the response-plan section is rendered")


def run_phase1_through_phase5_regression():
    print("\n" + "=" * 70)
    print("Regression: re-running the Phase 1, 2, 3, 4, and 5 test suites")
    print("=" * 70)
    import test_genai  # its own main() already re-runs Phase 1 + 2 + 3 + 4

    test_genai.main()


def main():
    print("=" * 70)
    print("Phase 6 defensive response planning test suite")
    print("=" * 70)

    test_normal_activity_produces_no_plan()
    test_possible_brute_force_plan_template()
    test_suspicious_login_plan_template()
    test_suspicious_data_transfer_plan_template()
    test_port_scanning_indicator_plan_template()
    test_unusual_network_activity_plan_template()
    test_other_anomaly_plan_template()
    test_unknown_incident_type_falls_back_safely()
    test_high_severity_sets_priority_and_policy_step()
    test_medium_severity_sets_priority_and_policy_step()
    test_low_severity_excludes_policy_step()
    test_high_confidence_skips_evidence_collection_step()
    test_medium_confidence_includes_evidence_collection_step()
    test_low_confidence_includes_evidence_and_review_steps()
    test_high_severity_low_confidence_is_urgent()
    test_medium_severity_high_confidence_not_merged_with_high_low_case()
    test_human_review_flag_adds_mandatory_step()
    test_no_mandatory_review_step_when_not_needed()
    test_deterministic_step_ordering()
    test_plan_generation_is_repeatable()
    test_validate_plan_passes_for_every_real_dataset_plan()
    test_validate_plan_flags_prohibited_action_keywords()
    test_validate_plan_flags_structural_problems_without_crashing()
    test_validate_plan_flags_missing_low_confidence_requirements()
    test_true_label_excluded_from_planning()
    test_genai_supplementary_context_is_advisory_only()
    test_empty_incident_input_handled_safely()
    test_streamlit_response_planning_page_runs()

    print("\n" + "=" * 70)
    print(f"PHASE 6 CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)

    run_phase1_through_phase5_regression()

    print("\n" + "=" * 70)
    print("ALL PHASE 1 + PHASE 2 + PHASE 3 + PHASE 4 + PHASE 5 + PHASE 6 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
