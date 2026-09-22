"""
response_planner.py
---------------------
Phase 6 - Defensive response planning.

    Incident Classification            (Phase 3)
            |
      Severity + Confidence            (Phase 3 / Phase 4)
            |
      Conditional Planner              (this module)
            |
      Ordered Defensive Steps
            |
      Human Review
            |
      Recommended Response

This module turns the STRUCTURED results of Phases 3-5 into an ordered,
explainable, ADVISORY-ONLY defensive response plan. It is a small,
transparent CONDITIONAL PLANNER (a fixed decision tree over
incident_type / severity / confidence_level / human_review_recommended),
not a model of any kind - every plan it produces can be traced back to
the exact conditions that selected it, the same explainability
philosophy as Phase 3's rule engine.

THE SYSTEM GENERATES A RECOMMENDED DEFENSIVE WORKFLOW. IT DOES NOT
EXECUTE CYBERSECURITY ACTIONS AGAINST REAL SYSTEMS. Nothing in this
module blocks an IP, disables an account, modifies a firewall,
terminates a session, isolates a machine, or runs any command against a
real system - every step's action is phrased as a recommendation
("review", "verify", "collect", "recommend ... according to policy",
"escalate", "document"), never an imperative technical command.
`validate_plan()` enforces this with a keyword blocklist
(`config.PROHIBITED_PLAN_ACTION_KEYWORDS`) as defense in depth, on top of
the step library itself never containing such text.

IMPORTANT - true_label:
    Nothing in this module reads `config.LABEL_COLUMN`. Planning is driven
    entirely by `InferenceRecord` (Phase 3) and `ConfidenceRecord`
    (Phase 4), neither of which ever reads true_label either.

IMPORTANT - severity/confidence stay authoritative:
    This module NEVER recomputes severity or confidence, and never lets
    Phase 5's Generative AI recommendations change `incident_type` or
    `severity`. Phase 5 output (when supplied) is carried only as
    read-only `supplementary_ai_context` text on the result - informative
    for the analyst, never turned into a plan step or allowed to affect
    step selection, ordering, or priority (Requirement #11).

Conceptual planning-state progression (labels only - the application
does not execute these against a real security environment):

    DETECTED -> TRIAGED -> EVIDENCE_COLLECTION -> ANALYST_REVIEW ->
    RESPONSE_RECOMMENDATION -> MONITOR_OR_ESCALATE -> CLOSED_OR_ESCALATED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import PLAN_PRIORITY_LEVELS, PROHIBITED_PLAN_ACTION_KEYWORDS
from src.confidence import ConfidenceEngineResult, ConfidenceRecord
from src.genai_analyzer import GenAIResult
from src.rule_engine import InferenceRecord

# =======================================================================
# Conceptual planning stages (Requirement #3 / #14)
# =======================================================================
# (internal state name, human-readable label for the UI flow diagram).
# A generated plan's `included_stages` is a subset of the state names
# below - which stages actually appear depends on the conditions that
# built the plan (e.g. a Normal Activity "plan" only reaches DETECTED and
# MONITOR_OR_ESCALATE; nothing here claims these transitions run against
# a real system - they describe a recommended workflow only).
PLAN_STAGES: List[Tuple[str, str]] = [
    ("DETECTED", "Detection"),
    ("TRIAGED", "Triage"),
    ("EVIDENCE_COLLECTION", "Evidence Collection"),
    ("ANALYST_REVIEW", "Analyst Review"),
    ("RESPONSE_RECOMMENDATION", "Defensive Recommendation"),
    ("MONITOR_OR_ESCALATE", "Monitor / Escalate"),
    ("CLOSED_OR_ESCALATED", "Document / Close"),
]
PLAN_STAGE_NAMES = [name for name, _ in PLAN_STAGES]
PLAN_STAGE_LABELS = dict(PLAN_STAGES)

# Documented, fixed ordering rationale (Requirement #9's "why this
# order?"), invariant across every plan - only which optional steps are
# included, and each step's wording/priority, changes per incident.
PLAN_ORDERING_RATIONALE = (
    "Steps always follow the same investigative lifecycle, regardless of "
    "incident type: first understand what was flagged and why (review), "
    "then learn more about the specific pattern (inspect/compare), then "
    "confirm who or what is actually affected (verify), then gather more "
    "facts if the evidence is not yet strong enough (collect), then get a "
    "human sign-off before anything high-impact is considered (review), "
    "then decide on a policy-based response if warranted (recommend), "
    "then keep watching or hand off (monitor/escalate), and finally leave "
    "a record (document). This order is never shuffled or randomized - "
    "only whether an optional step (evidence collection, human review, "
    "policy-based recommendation) is included, and each step's wording "
    "and priority, changes with severity, confidence, and the human-review "
    "flag."
)


# =======================================================================
# Result containers (Requirement #1 / #12)
# =======================================================================
@dataclass
class PlanStep:
    """One ordered, advisory step in a response plan."""

    step_number: int
    action: str
    purpose: str
    reason: str
    priority: str                 # one of config.PLAN_PRIORITY_LEVELS
    requires_human_review: bool
    stage: str = "TRIAGED"        # one of PLAN_STAGE_NAMES


@dataclass
class PlanValidationResult:
    """Requirement #10 - the planner's own self-check, never a crash."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ResponsePlanResult:
    """Everything the UI needs to show one incident's response plan."""

    incident_type: str
    severity: str
    confidence_level: str
    human_review_recommended: bool
    plan_title: str
    plan_summary: str
    steps: List[PlanStep]
    priority: str                         # overall plan priority
    review_required: bool                 # True if ANY step requires human review
    planning_reason: str                  # why THIS plan was selected
    validation_status: PlanValidationResult
    included_stages: List[str] = field(default_factory=list)
    # Phase 5 output is NEVER turned into a step - it is carried here,
    # read-only, purely as supplementary context for the analyst
    # (Requirement #11). Always [] when no GenAIResult was supplied.
    supplementary_ai_context: List[str] = field(default_factory=list)


@dataclass
class ResponsePlanningResult:
    """
    Batch-level summary used by the Streamlit "Response Planning" page's
    metrics row - mirrors the same total/HIGH/MEDIUM/LOW/human-review
    summary shape already used by Phase 3's `RuleEngineResult` and Phase
    4's `ConfidenceEngineResult`, so the UI stays consistent across pages.

    `results_df` is Phase 4's `results_df` with one additional `_plan`
    column holding each row's already-computed `ResponsePlanResult` -
    following the same "compute once, store the object, reuse it" pattern
    as the existing `_inference` / `_confidence` columns.
    """

    results_df: pd.DataFrame
    total_events: int
    total_incidents_with_plan: int   # events whose incident_type != "Normal Activity"
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    human_review_required_count: int  # plans where review_required is True


# =======================================================================
# Base step library (Requirement #4) - safe, defensive, advisory only.
# =======================================================================
# Per-incident-type "core" investigative steps (Requirement #5, A-F).
# Each entry is (action, purpose). The shared tail (evidence collection,
# human review, policy-based recommendation, monitor/escalate, document)
# is built once, conditionally, in _build_tail_steps() below rather than
# repeated per type - the CONTENT matches the spec's per-type workflows,
# just without duplicating the shared closing steps six times over.
INCIDENT_CORE_STEPS: Dict[str, List[Tuple[str, str]]] = {
    "Possible Brute Force": [
        ("Review failed-login activity", "Confirm the observed authentication pattern and its frequency."),
        ("Inspect source IP and related login events", "Determine whether the activity is isolated or repeated from the same origin."),
        ("Verify the affected account", "Confirm whether the account owner recognizes and authorized this activity."),
    ],
    "Suspicious Login": [
        ("Review the login event", "Understand exactly what triggered this classification."),
        ("Verify user identity and activity", "Confirm the login was performed by the legitimate account owner."),
        ("Examine location-mismatch evidence and recent sessions/devices", "Determine whether the location/device pattern is expected for this user."),
    ],
    "Suspicious Data Transfer": [
        ("Review the transferred volume and event details", "Understand the scale and nature of the data movement."),
        ("Identify the user/session involved", "Determine who initiated the transfer."),
        ("Examine related file/network activity", "Compare the activity against expected behavior and policy."),
    ],
    "Port Scanning Indicator": [
        ("Review source and destination information", "Identify where the scanning activity originated and what it targeted."),
        ("Examine request activity and related network events", "Determine the scope and pattern of the scanning behavior."),
        ("Determine whether the activity is expected", "Rule out authorized scanning, such as a scheduled vulnerability scan."),
    ],
    "Unusual Network Activity": [
        ("Review request volume and related network events", "Understand the scale and pattern of the unusual traffic."),
        ("Identify relevant source/destination information", "Determine which systems are involved."),
        ("Compare against normal activity", "Determine how far this deviates from the user's or system's baseline."),
    ],
    "Other Anomaly": [
        ("Review the anomaly evidence", "Understand what the anomaly detector flagged and why."),
        ("Inspect related events", "Look for surrounding activity that could help explain the anomaly."),
        ("Determine whether a known incident pattern applies", "Check whether this anomaly matches a recognized category even though no rule matched it."),
    ],
}

# incident_type -> which PLAN_STAGE the core steps above belong to.
_CORE_STAGE = "TRIAGED"


def _severity_base_priority(severity: str) -> str:
    """HIGH -> HIGH, MEDIUM -> STANDARD, LOW -> ROUTINE (Requirement #6)."""
    return {"HIGH": "HIGH", "MEDIUM": "STANDARD", "LOW": "ROUTINE"}.get(severity, "ROUTINE")


def _overall_priority(severity: str, confidence_level: str) -> str:
    """
    Combine severity and confidence into one overall plan priority -
    WITHOUT merging the two concepts into a single score (Requirement
    #7): severity sets the base tier, and LOW confidence bumps the plan
    up exactly one tier because uncertain evidence needs prompt attention
    to resolve, regardless of how "serious" the rules judged it to be.
    Example from the spec: HIGH severity + LOW confidence -> URGENT.
    """
    base = _severity_base_priority(severity)
    if confidence_level != "LOW":
        return base
    bump = {"ROUTINE": "STANDARD", "STANDARD": "HIGH", "HIGH": "URGENT", "URGENT": "URGENT"}
    return bump.get(base, base)


def _build_tail_steps(
    start_number: int,
    severity: str,
    confidence_level: str,
    human_review_recommended: bool,
    overall_priority: str,
) -> List[PlanStep]:
    """
    The shared, conditional closing sequence for every non-Normal
    incident plan (Requirements #6/#7/#8/#9): evidence collection (if
    confidence isn't already HIGH), human review (if recommended or
    confidence is LOW), a policy-based recommendation (if severity is
    HIGH or MEDIUM), then monitor/escalate, then document - always in
    that fixed order.
    """
    steps: List[PlanStep] = []
    n = start_number

    # --- Evidence collection (Requirement #7) ---
    # HIGH confidence: existing evidence is already strong, so this
    # dedicated step is skipped - "fewer additional-evidence steps when
    # existing evidence is strong" - the incident-specific core steps
    # already cover the investigative substance.
    if confidence_level != "HIGH":
        if confidence_level == "LOW":
            reason = "Confidence is LOW; substantially more evidence is needed before this interpretation can be trusted."
        else:
            reason = "Confidence is MEDIUM; additional evidence can strengthen the interpretation."
        steps.append(
            PlanStep(
                step_number=n,
                action="Collect additional evidence",
                purpose="Strengthen or challenge the current interpretation before deciding on a response.",
                reason=reason,
                priority=overall_priority,
                requires_human_review=False,
                stage="EVIDENCE_COLLECTION",
            )
        )
        n += 1

    # --- Human analyst review (Requirement #7 LOW confidence / #8 flag) ---
    review_triggers = []
    if human_review_recommended:
        review_triggers.append("the confidence model flagged this event for human review")
    if confidence_level == "LOW":
        review_triggers.append("confidence is LOW, so no high-impact action should be taken without a human in the loop")
    if review_triggers:
        steps.append(
            PlanStep(
                step_number=n,
                action="Human analyst review required before any high-impact response",
                purpose="Ensure a qualified analyst evaluates the evidence before any policy-based response is pursued.",
                reason="; ".join(review_triggers).capitalize() + ".",
                priority="URGENT" if overall_priority == "URGENT" else "HIGH",
                requires_human_review=True,
                stage="ANALYST_REVIEW",
            )
        )
        n += 1

    # --- Policy-based recommendation (Requirement #6) ---
    # LOW severity explicitly prioritizes routine investigation/
    # monitoring/documentation over a containment recommendation, so this
    # step is only included for MEDIUM/HIGH severity.
    if severity in ("HIGH", "MEDIUM"):
        if severity == "HIGH":
            reason = "HIGH severity requires prompt defensive consideration."
        else:
            reason = "MEDIUM severity warrants a considered, policy-based response recommendation."
        steps.append(
            PlanStep(
                step_number=n,
                action="Recommend containment or escalation according to organizational policy",
                purpose="Give the analyst a concrete, policy-bounded defensive option to consider - never executed automatically.",
                reason=reason,
                priority=overall_priority,
                requires_human_review=True,
                stage="RESPONSE_RECOMMENDATION",
            )
        )
        n += 1

    # --- Monitor / escalate (always present) ---
    if severity == "HIGH":
        monitor_reason = "HIGH severity warrants prompt escalation if the evidence remains suspicious after review."
    elif severity == "MEDIUM":
        monitor_reason = "Escalate if the investigation confirms malicious activity; otherwise continue monitoring."
    else:
        monitor_reason = "Continue routine monitoring; escalate only if new evidence emerges."
    steps.append(
        PlanStep(
            step_number=n,
            action="Continue monitoring or escalate, per severity and findings",
            purpose="Keep watching for related activity, or hand off to a broader response process when warranted.",
            reason=monitor_reason,
            priority=overall_priority,
            requires_human_review=False,
            stage="MONITOR_OR_ESCALATE",
        )
    )
    n += 1

    # --- Document (always last) ---
    steps.append(
        PlanStep(
            step_number=n,
            action="Document analyst decision and findings",
            purpose="Preserve a record of the investigation and its outcome for future reference and audit.",
            reason="Every investigated event should leave a documented trail regardless of its outcome.",
            priority="ROUTINE",
            requires_human_review=False,
            stage="CLOSED_OR_ESCALATED",
        )
    )
    return steps


def _build_plan_title(incident_type: str, severity: str) -> str:
    if severity == "HIGH":
        prefix = "High-Priority"
    elif severity == "MEDIUM":
        prefix = "Standard"
    else:
        prefix = "Routine"
    return f"{prefix} {incident_type} Response Plan"


def _build_planning_reason(incident_type: str, severity: str, confidence_level: str, human_review_recommended: bool) -> str:
    parts = [
        f"Incident type '{incident_type}' selected the matching investigative template.",
        f"Severity '{severity}' set the base priority and whether a policy-based recommendation step is included.",
        f"Confidence '{confidence_level}' determined how much additional evidence collection is required.",
    ]
    if human_review_recommended:
        parts.append("Phase 4 explicitly recommended human review, so a mandatory analyst-review step was added.")
    return " ".join(parts)


def _included_stages(steps: List[PlanStep]) -> List[str]:
    seen = {"DETECTED"}  # every plan implicitly starts from "detected"
    for step in steps:
        seen.add(step.stage)
    return [name for name in PLAN_STAGE_NAMES if name in seen]


# =======================================================================
# Plan generation (Requirement #2, #5, #6, #7, #8, #9, #11)
# =======================================================================
def generate_response_plan(
    inference_record: InferenceRecord,
    confidence_record: ConfidenceRecord,
    genai_result: Optional[GenAIResult] = None,
) -> ResponsePlanResult:
    """
    Build the full advisory response plan for one event.

    `incident_type` and `severity` come from `inference_record` (Phase 3);
    `confidence_level`/`human_review_recommended` come from
    `confidence_record` (Phase 4). Neither is recomputed here.
    `genai_result` (Phase 5) is OPTIONAL and, when supplied, is used ONLY
    to populate `supplementary_ai_context` - it can never change which
    steps are generated, their order, or their priority (Requirement #11).
    """
    incident_type = inference_record.incident_type
    severity = inference_record.severity
    confidence_level = confidence_record.confidence_level
    human_review_recommended = confidence_record.human_review_recommended

    supplementary_context = list(genai_result.recommendations) if genai_result is not None else []

    # --- Requirement #5-G / #6 N/A: Normal Activity gets NO incident-response plan ---
    if incident_type == "Normal Activity":
        result = ResponsePlanResult(
            incident_type=incident_type,
            severity=severity,
            confidence_level=confidence_level,
            human_review_recommended=human_review_recommended,
            plan_title="No Incident Response Required",
            plan_summary=(
                "This event was not classified as an incident. No response plan is "
                "generated - continue routine monitoring and retain the event "
                "information as part of normal activity records."
            ),
            steps=[],
            priority="ROUTINE",
            review_required=False,
            planning_reason="incident_type is 'Normal Activity', so no incident-response plan applies.",
            validation_status=PlanValidationResult(is_valid=True),
            included_stages=["DETECTED", "MONITOR_OR_ESCALATE"],
            supplementary_ai_context=supplementary_context,
        )
        result.validation_status = validate_plan(result)
        return result

    overall_priority = _overall_priority(severity, confidence_level)

    core_specs = INCIDENT_CORE_STEPS.get(incident_type)
    if core_specs is None:
        # Unknown/unexpected incident_type - treat like "Other Anomaly" so
        # the app never crashes on an unrecognized category, but say so.
        core_specs = INCIDENT_CORE_STEPS["Other Anomaly"]

    steps: List[PlanStep] = []
    for i, (action, purpose) in enumerate(core_specs, start=1):
        steps.append(
            PlanStep(
                step_number=i,
                action=action,
                purpose=purpose,
                reason=f"Part of the standard '{incident_type}' investigative sequence.",
                priority=overall_priority,
                requires_human_review=False,
                stage=_CORE_STAGE,
            )
        )

    tail = _build_tail_steps(
        start_number=len(steps) + 1,
        severity=severity,
        confidence_level=confidence_level,
        human_review_recommended=human_review_recommended,
        overall_priority=overall_priority,
    )
    steps.extend(tail)

    plan_summary = (
        f"A {overall_priority.lower()}-priority investigative plan for a '{incident_type}' "
        f"event ({severity} severity, {confidence_level} confidence). "
        f"{'Human analyst review is required before any high-impact response. ' if any(s.requires_human_review for s in steps) else ''}"
        "All steps are advisory recommendations for a human analyst."
    ).strip()

    result = ResponsePlanResult(
        incident_type=incident_type,
        severity=severity,
        confidence_level=confidence_level,
        human_review_recommended=human_review_recommended,
        plan_title=_build_plan_title(incident_type, severity),
        plan_summary=plan_summary,
        steps=steps,
        priority=overall_priority,
        review_required=any(s.requires_human_review for s in steps),
        planning_reason=_build_planning_reason(incident_type, severity, confidence_level, human_review_recommended),
        validation_status=PlanValidationResult(is_valid=True),  # replaced below
        included_stages=_included_stages(steps),
        supplementary_ai_context=supplementary_context,
    )
    result.validation_status = validate_plan(result)
    return result


# =======================================================================
# Plan validation (Requirement #10)
# =======================================================================
def validate_plan(plan: ResponsePlanResult) -> PlanValidationResult:
    """
    Self-check a generated plan. NEVER raises - any problem is reported
    as an error/warning in the returned `PlanValidationResult` so the UI
    can show a clear warning instead of crashing.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if plan.incident_type == "Normal Activity":
        if plan.steps:
            errors.append("Normal Activity must not produce any incident-response steps.")
    else:
        if not plan.steps:
            errors.append(f"'{plan.incident_type}' is an incident but no steps were generated.")

    # Every step has an order; order is unique and sequential from 1.
    numbers = [s.step_number for s in plan.steps]
    if len(set(numbers)) != len(numbers):
        errors.append("Step numbers are not unique.")
    if numbers and sorted(numbers) != list(range(1, len(numbers) + 1)):
        errors.append("Step numbers are not a contiguous sequence starting at 1.")

    # Steps are not empty.
    for step in plan.steps:
        if not step.action or not step.action.strip():
            errors.append(f"Step {step.step_number} has an empty action.")
        if not step.reason or not step.reason.strip():
            warnings.append(f"Step {step.step_number} has no stated reason.")

    # High-impact steps (policy-based recommendation / analyst review)
    # must require human review.
    for step in plan.steps:
        lowered = step.action.lower()
        if ("contain" in lowered or "escalat" in lowered) and not step.requires_human_review:
            # "Continue monitoring or escalate" is a low-impact fallback
            # step, not a containment recommendation - only the explicit
            # containment/review steps are held to this rule.
            if "recommend containment" in lowered or "human analyst review" in lowered:
                errors.append(f"Step {step.step_number} ('{step.action}') is high-impact but does not require human review.")

    # Low-confidence incidents include evidence collection and review.
    if plan.incident_type != "Normal Activity" and plan.confidence_level == "LOW":
        has_evidence_step = any("evidence" in s.action.lower() for s in plan.steps)
        has_review_step = any(s.requires_human_review for s in plan.steps)
        if not has_evidence_step:
            errors.append("LOW confidence incident is missing an evidence-collection step.")
        if not has_review_step:
            errors.append("LOW confidence incident is missing a human-review step.")

    # No prohibited offensive/automated action anywhere in the plan.
    for step in plan.steps:
        lowered = step.action.lower()
        for keyword in PROHIBITED_PLAN_ACTION_KEYWORDS:
            if keyword in lowered:
                errors.append(f"Step {step.step_number} contains a prohibited action keyword: '{keyword}'.")

    if plan.priority not in PLAN_PRIORITY_LEVELS:
        errors.append(f"Plan priority '{plan.priority}' is not one of {PLAN_PRIORITY_LEVELS}.")

    return PlanValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=warnings)


# =======================================================================
# Batch orchestration (Requirement #13 - planning summary metrics)
# =======================================================================
def run_response_planning(confidence_result: ConfidenceEngineResult) -> ResponsePlanningResult:
    """
    Run `generate_response_plan()` once for every event already scored by
    Phase 4, and summarize the results for the Streamlit dashboard.

    This NEVER re-runs anomaly detection, the rule engine, or confidence
    scoring - it only reads the `_inference` / `_confidence` objects
    Phase 3/4 already computed and stored in `confidence_result.results_df`.
    Phase 5 (GenAI) output is intentionally NOT used here: it is only ever
    attached to a single, analyst-selected event's plan on the Streamlit
    page, never to this batch summary.
    """
    results_df = confidence_result.results_df.copy()

    plans: List[ResponsePlanResult] = [
        generate_response_plan(row["_inference"], row["_confidence"])
        for _, row in results_df.iterrows()
    ]
    results_df["_plan"] = plans

    incidents = [p for p in plans if p.incident_type != "Normal Activity"]

    return ResponsePlanningResult(
        results_df=results_df,
        total_events=len(results_df),
        total_incidents_with_plan=len(incidents),
        high_severity_count=sum(1 for p in incidents if p.severity == "HIGH"),
        medium_severity_count=sum(1 for p in incidents if p.severity == "MEDIUM"),
        low_severity_count=sum(1 for p in incidents if p.severity == "LOW"),
        human_review_required_count=sum(1 for p in plans if p.review_required),
    )
