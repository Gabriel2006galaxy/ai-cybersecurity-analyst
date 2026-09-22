"""
confidence.py
--------------
Phase 4 - Uncertainty and confidence scoring.

    Phase 2 Anomaly Detection
            |
    Phase 3 Facts + Rules + Inference
            |
      Evidence Strength
            |
      Confidence Score
            |
      Confidence Level
            |
      Uncertainty Explanation
            |
      Human Review Recommendation

This module answers a question Phase 3 deliberately does NOT answer:
"How strong and consistent is the evidence behind the current incident
interpretation?" It sits entirely on TOP of Phase 2 (`AnomalyDetectionResult`)
and Phase 3 (`RuleEngineResult` / `InferenceRecord`) - it never re-runs the
anomaly model, never re-evaluates rule conditions, and never changes an
`incident_type` or `severity` that Phase 3 already produced.

IMPORTANT - this is NOT a probability:
    `confidence_score` is an EXPLAINABLE, EVIDENCE-BASED score in [0, 1],
    built from a small number of transparent, hand-weighted components
    (see `score_confidence()` below). It is NOT a calibrated statistical
    probability that an attack occurred, and it is NOT a guarantee that
    `incident_type`/`severity` are correct. Language throughout this
    module and the Streamlit page deliberately says "confidence score:
    0.85" / "confidence level: HIGH", never "85% probability of attack".

IMPORTANT - true_label:
    Nothing in this module reads `config.LABEL_COLUMN`, imports it, or
    accepts it as a parameter. Every input here comes from Phase 3's
    `InferenceRecord` (facts derived from observed fields + Phase 2's
    `anomaly_status`) and a batch-relative transform of Phase 2's
    `anomaly_score`. Component weights and tier cutoffs (config.py) were
    picked from documented reasoning about what should count as stronger
    or weaker evidence, not by tuning against true_label or any evaluation
    metric (accuracy/F1/etc.).

IMPORTANT - severity vs. confidence:
    Phase 3's `severity` answers "how serious is this incident according
    to the rules that fired?". This module's `confidence_score` /
    `confidence_level` answer a different question: "how strong and
    consistent is the evidence supporting that interpretation?". The two
    are computed independently and are never merged into one number - a
    HIGH-severity, LOW-confidence event is valid and expected (see
    `test_confidence.py::test_high_severity_low_confidence_allowed`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import (
    CONFIDENCE_WEIGHT_ANOMALY,
    CONFIDENCE_WEIGHT_CONSISTENCY,
    CONFIDENCE_WEIGHT_RULES,
    CONFIDENCE_WEIGHT_SUPPORTING,
    CONSISTENCY_HIGH,
    CONSISTENCY_LOW,
    CONSISTENCY_MEDIUM,
    DATA_TRANSFER_THRESHOLD,
    DEVICE_COUNT_THRESHOLD,
    FAILED_LOGIN_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    REQUEST_COUNT_THRESHOLD,
    RULE_EVIDENCE_MULTIPLE_RULES,
    RULE_EVIDENCE_NORMAL_BASELINE,
    RULE_EVIDENCE_ONE_RULE,
    RULE_EVIDENCE_ZERO_RULES,
    STRONG_RULE_EVIDENCE_BONUS,
)
from src.rule_engine import HIGH_RISK_CONCLUSIONS, InferenceRecord, RuleEngineResult, TriggeredRule

# =======================================================================
# Supporting-indicator facts (Requirement #7)
# =======================================================================
# Exactly the "observed indicator" facts from Phase 3's FACT_DESCRIPTIONS,
# excluding "anomalous_event" (that IS the anomaly-evidence component, see
# compute_anomaly_evidence()) and "unusual_port" (a technical detail that
# only matters combined with high_request_count inside RULE-SCAN-01 - it
# isn't listed as its own indicator in the Phase 4 spec).
SUPPORTING_FACT_NAMES: List[str] = [
    "high_failed_logins",
    "high_request_count",
    "high_data_transfer",
    "off_hours_activity",
    "location_mismatch",
    "multiple_devices",
    "privilege_escalation_attempt",
    "network_scan_indicator",
]


# =======================================================================
# Result containers
# =======================================================================
@dataclass
class EvidenceComponent:
    """One weighted component of the blended confidence score."""

    value: float        # raw component score, in [0, 1]
    weight: float        # config.py weight applied to this component
    contribution: float  # value * weight - this component's share of the final score


@dataclass
class ConfidenceRecord:
    """Everything needed to explain ONE event's confidence assessment."""

    confidence_score: float = 0.0
    confidence_level: str = "LOW"
    evidence_strength: str = "WEAK"
    supporting_indicators: List[str] = field(default_factory=list)
    confidence_reasons: List[str] = field(default_factory=list)
    uncertainty_reason: str = ""
    human_review_recommended: bool = False
    components: Dict[str, EvidenceComponent] = field(default_factory=dict)


@dataclass
class ConfidenceEngineResult:
    results_df: pd.DataFrame
    total_events: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    human_review_count: int


# =======================================================================
# 1. Batch-relative anomaly-score normalization (Requirement #5)
# =======================================================================
def normalize_anomaly_scores(anomaly_scores: pd.Series) -> pd.Series:
    """
    Deterministic BATCH-RELATIVE min-max normalization of `anomaly_score`
    to the range [0, 1]:

        normalized = (score - batch_min) / (batch_max - batch_min)

    This is NOT a probability - it is purely a bounded rescaling so the
    Isolation Forest's decision-function-derived score can be combined
    with the other 0-1 evidence components below. "Batch-relative" means
    the same raw `anomaly_score` can map to a different normalized value
    if it is scored alongside a different set of events; this is
    documented and deterministic (the same batch always produces the same
    result) - it never reads `true_label`.

    If every score in the batch is identical (e.g. a batch of size 1, or a
    degenerate model output), every event gets the neutral midpoint 0.5
    instead of dividing by zero.
    """
    if anomaly_scores.empty:
        return anomaly_scores.copy()

    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    spread = max_score - min_score
    if spread < 1e-9:
        return pd.Series(0.5, index=anomaly_scores.index)
    return (anomaly_scores - min_score) / spread


# =======================================================================
# 2. Evidence components (Requirements #4-#8)
# =======================================================================
def compute_anomaly_evidence(anomalous: bool, normalized_score: float) -> float:
    """
    Component 1 - Anomaly evidence.

    How strongly the Phase 2 signal supports the CURRENT classification,
    whichever it is:
      - anomalous event: a HIGH normalized score (very isolated relative
        to the rest of the batch) is strong evidence FOR the anomalous
        interpretation.
      - normal event: a LOW normalized score (clearly unremarkable
        relative to the rest of the batch) is strong evidence FOR the
        normal interpretation. A borderline score near the batch's high
        end - even though it landed on the NORMAL side of the model's
        threshold - is weaker evidence, i.e. less confidence in "normal".
    """
    normalized_score = min(max(normalized_score, 0.0), 1.0)
    return normalized_score if anomalous else (1.0 - normalized_score)


def compute_rule_evidence(anomalous: bool, triggered_rules: List[TriggeredRule]) -> float:
    """
    Component 2 - Rule evidence (Requirement #6).

    Uses the NUMBER of triggered Phase 3 rules as a base tier, then adds a
    small bonus if any triggered rule's conclusion is a designated
    high-risk indicator (`rule_engine.HIGH_RISK_CONCLUSIONS`). Rules never
    evaluate for a Normal Activity classification, so a fixed neutral
    baseline is used there instead (see RULE_EVIDENCE_NORMAL_BASELINE in
    config.py). Deliberately does NOT assume "more rules = definitely
    true" on its own - this is only one of four weighted components.
    """
    if not anomalous:
        return RULE_EVIDENCE_NORMAL_BASELINE

    n = len(triggered_rules)
    if n == 0:
        base = RULE_EVIDENCE_ZERO_RULES
    elif n == 1:
        base = RULE_EVIDENCE_ONE_RULE
    else:
        base = RULE_EVIDENCE_MULTIPLE_RULES

    strong = any(tr.conclusion in HIGH_RISK_CONCLUSIONS for tr in triggered_rules)
    if strong:
        base = min(1.0, base + STRONG_RULE_EVIDENCE_BONUS)
    return base


def compute_supporting_evidence(anomalous: bool, facts: Dict[str, bool]) -> Tuple[float, int]:
    """
    Component 3 - Supporting-indicator evidence (Requirement #7).

    Counts how many of the SUPPORTING_FACT_NAMES observed-indicator facts
    are True, and expresses that as a fraction of the total. Symmetric
    with the anomaly-evidence component: for an anomalous event, more
    present indicators support the anomalous/incident interpretation; for
    a normal event, FEWER present indicators support the normal
    interpretation (an indicator crossing its threshold on an otherwise
    "normal" event is mild evidence AGAINST that classification, not for
    it - this is what feeds the "conflicting evidence" consistency case).

    Returns (evidence_value, raw_count) so callers can reuse the count for
    the consistency component and the human-readable indicator list.
    """
    count = sum(1 for name in SUPPORTING_FACT_NAMES if facts.get(name))
    total = len(SUPPORTING_FACT_NAMES)
    ratio = count / total if total else 0.0
    evidence = ratio if anomalous else (1.0 - ratio)
    return evidence, count


def compute_consistency(anomalous: bool, rule_count: int, supporting_count: int) -> Tuple[float, str]:
    """
    Component 4 - Evidence consistency (Requirement #8).

    A simple, deterministic three-tier check of whether the anomaly
    signal, the rule engine, and the supporting indicators agree:

    Anomalous event:
      HIGH   - at least one rule matched AND 2+ supporting indicators are
               present (multiple independent signals corroborate a
               specific incident interpretation).
      MEDIUM - at least one rule matched, OR at least one supporting
               indicator is present, but not both conditions for HIGH
               (partial agreement).
      LOW    - anomalous with ZERO rules matched AND ZERO supporting
               indicators present - a "bare" anomaly with no explainable
               corroboration. This is the
               "anomaly detected, but evidence is insufficient to
               confidently assign a specific incident type" case.

    Normal event:
      HIGH   - zero supporting indicators are present - a clean agreement
               that nothing about this event looks unusual.
      MEDIUM - exactly one supporting indicator is present despite the
               normal classification (minor, single-signal ambiguity).
      LOW    - two or more supporting indicators are present despite the
               normal classification - multiple individual signals look
               suspicious even though the overall model verdict is
               "normal": genuinely conflicting evidence worth a second
               look.
    """
    if anomalous:
        if rule_count > 0 and supporting_count >= 2:
            return CONSISTENCY_HIGH, "high"
        if rule_count > 0 or supporting_count > 0:
            return CONSISTENCY_MEDIUM, "medium"
        return CONSISTENCY_LOW, "low"
    else:
        if supporting_count == 0:
            return CONSISTENCY_HIGH, "high"
        if supporting_count == 1:
            return CONSISTENCY_MEDIUM, "medium"
        return CONSISTENCY_LOW, "low"


def determine_evidence_strength(anomalous: bool, rule_count: int, supporting_count: int) -> str:
    """
    A plain-language WEAK / MODERATE / STRONG summary of how MUCH
    corroborating evidence exists - independent of which classification
    it happens to support. Distinct from `confidence_level` (which
    reflects the direction-aware blended score): a "STRONG" evidence
    event can still be a confident NORMAL classification, not just a
    confident incident.
    """
    if not anomalous:
        if supporting_count == 0:
            return "STRONG"
        if supporting_count == 1:
            return "MODERATE"
        return "WEAK"

    if rule_count >= 1 and supporting_count >= 2:
        return "STRONG"
    if rule_count >= 1 or supporting_count >= 1:
        return "MODERATE"
    return "WEAK"


def determine_confidence_level(score: float) -> str:
    """LOW / MEDIUM / HIGH using the configurable thresholds in config.py."""
    if score < LOW_CONFIDENCE_THRESHOLD:
        return "LOW"
    if score < HIGH_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def determine_human_review_recommended(
    confidence_level: str, anomalous: bool, rule_count: int, consistency_tier: str
) -> bool:
    """
    Requirement #11 - a RECOMMENDATION FLAG ONLY. Phase 7 will build the
    actual analyst review workflow; nothing here performs any action.

    True when any of:
      - overall confidence is LOW;
      - the event is anomalous but no rule currently supports a specific
        incident interpretation (a "bare" anomaly - insufficient
        evidence), regardless of what the blended score came out to;
      - evidence consistency is the LOW tier (conflicting or insufficient
        agreement across anomaly/rule/indicator evidence), surfaced even
        if the blended score didn't happen to land in LOW.
    """
    return (
        confidence_level == "LOW"
        or (anomalous and rule_count == 0)
        or consistency_tier == "low"
    )


# =======================================================================
# 3. Human-readable explanations
# =======================================================================
def _describe_supporting_fact(fact_name: str, evidence: Dict) -> str:
    """One human-readable line for a True supporting fact, using the
    event's own observed values (mirrors rule_engine's `explain` style)."""
    if fact_name == "high_failed_logins":
        return f"failed_logins = {evidence.get('failed_logins')} (>= {FAILED_LOGIN_THRESHOLD})"
    if fact_name == "high_request_count":
        return f"request_count = {evidence.get('request_count')} (>= {REQUEST_COUNT_THRESHOLD})"
    if fact_name == "high_data_transfer":
        return f"data_transferred_mb = {evidence.get('data_transferred_mb')} (>= {DATA_TRANSFER_THRESHOLD})"
    if fact_name == "off_hours_activity":
        return f"occurred off-hours (hour = {evidence.get('hour')})"
    if fact_name == "location_mismatch":
        return f"login_location = '{evidence.get('login_location')}' differs from the user's usual location"
    if fact_name == "multiple_devices":
        return f"device_count = {evidence.get('device_count')} (>= {DEVICE_COUNT_THRESHOLD})"
    if fact_name == "privilege_escalation_attempt":
        return f"event_type = '{evidence.get('event_type')}' (privilege escalation)"
    if fact_name == "network_scan_indicator":
        return f"event_type = '{evidence.get('event_type')}' (network scan)"
    return fact_name  # safety net; every SUPPORTING_FACT_NAMES entry is handled above


def build_supporting_indicators(record: InferenceRecord) -> List[str]:
    """
    The full, ordered list of human-readable evidence lines behind this
    event's confidence assessment: the base anomaly signal, then any true
    supporting indicators, then any triggered rules by name.
    """
    indicators: List[str] = []

    anomaly_score = record.evidence.get("anomaly_score", 0.0) or 0.0
    if record.facts.get("anomalous_event"):
        indicators.append(f"Anomalous event (Phase 2 anomaly_score = {anomaly_score:.3f})")
    else:
        indicators.append(f"Not flagged anomalous by Phase 2 (anomaly_score = {anomaly_score:.3f})")

    for name in SUPPORTING_FACT_NAMES:
        if record.facts.get(name):
            indicators.append(_describe_supporting_fact(name, record.evidence))

    for tr in record.triggered_rules:
        indicators.append(f"{tr.rule_id} triggered ({tr.name})")

    return indicators


def build_confidence_reasons(
    anomalous: bool,
    normalized_score: float,
    anomaly_evidence: float,
    rule_count: int,
    strong_rule: bool,
    supporting_count: int,
    consistency_tier: str,
) -> List[str]:
    """Short, plain-language bullets explaining WHY the score came out this way."""
    reasons: List[str] = []

    if anomaly_evidence >= 0.7:
        reasons.append(f"Strong anomaly evidence (normalized score {normalized_score:.2f} within this batch).")
    elif anomaly_evidence >= 0.4:
        reasons.append(f"Moderate anomaly evidence (normalized score {normalized_score:.2f} within this batch).")
    else:
        reasons.append(f"Weak anomaly evidence (normalized score {normalized_score:.2f} within this batch).")

    if not anomalous:
        reasons.append("Rule engine does not evaluate rules for Normal Activity (neutral baseline used).")
    elif rule_count == 0:
        reasons.append("No knowledge-base rule currently supports a specific incident interpretation.")
    elif rule_count == 1:
        line = "Rule evidence supports the classification (1 rule triggered)."
        if strong_rule:
            line += " Includes a high-risk rule pattern."
        reasons.append(line)
    else:
        reasons.append(f"Rule evidence strongly supports the classification ({rule_count} rules triggered).")

    total = len(SUPPORTING_FACT_NAMES)
    if anomalous:
        reasons.append(f"{supporting_count} of {total} supporting indicators are present.")
    elif supporting_count == 0:
        reasons.append(f"None of the {total} supporting indicators are present, consistent with Normal Activity.")
    else:
        reasons.append(f"{supporting_count} of {total} supporting indicators are present despite the Normal classification.")

    if consistency_tier == "high":
        reasons.append("Evidence is highly consistent across anomaly detection, rules, and indicators.")
    elif consistency_tier == "medium":
        reasons.append("Evidence is moderately consistent - some but not all sources agree.")
    else:
        reasons.append("Evidence is inconsistent or insufficient across sources - treat this classification with caution.")

    return reasons


def build_uncertainty_reason(anomalous: bool, rule_count: int, consistency_tier: str, confidence_level: str) -> str:
    """A single, focused sentence naming the primary source of uncertainty
    (or confirming that none stands out)."""
    if anomalous and rule_count == 0:
        return (
            "Anomaly detected, but no knowledge-base rule currently supports a specific "
            "incident interpretation - evidence is insufficient to confidently assign a category."
        )
    if not anomalous and consistency_tier == "low":
        return (
            "Classified Normal Activity, but multiple individual indicators look unusual - "
            "evidence is conflicting and worth a second look."
        )
    if consistency_tier == "low":
        return "Evidence from anomaly detection, rules, and indicators does not consistently agree."
    if confidence_level == "LOW":
        return "Overall evidence is limited; treat this classification as tentative."
    return "No significant sources of uncertainty were identified for this event."


# =======================================================================
# 4. Per-event scoring + orchestration
# =======================================================================
def score_confidence(
    record: InferenceRecord, anomaly_score_normalized: Optional[float] = None
) -> ConfidenceRecord:
    """
    Run the full Evidence -> Confidence Score -> Confidence Level ->
    Uncertainty -> Human Review chain for ONE already-computed Phase 3
    `InferenceRecord`.

    `anomaly_score_normalized` should be this event's batch-relative
    normalized anomaly score (see `normalize_anomaly_scores()`), computed
    once per batch by `run_confidence_scoring()`. It defaults to the
    neutral midpoint 0.5 ONLY for standalone/unit-test use outside a full
    batch run, where no batch context exists to normalize against.
    """
    anomalous = bool(record.facts.get("anomalous_event", False))
    rule_count = len(record.triggered_rules)
    strong_rule = any(tr.conclusion in HIGH_RISK_CONCLUSIONS for tr in record.triggered_rules)

    if anomaly_score_normalized is None:
        anomaly_score_normalized = 0.5
    normalized_score = min(max(float(anomaly_score_normalized), 0.0), 1.0)

    anomaly_evidence = compute_anomaly_evidence(anomalous, normalized_score)
    rule_evidence = compute_rule_evidence(anomalous, record.triggered_rules)
    supporting_evidence, supporting_count = compute_supporting_evidence(anomalous, record.facts)
    consistency_evidence, consistency_tier = compute_consistency(anomalous, rule_count, supporting_count)

    raw_score = (
        CONFIDENCE_WEIGHT_ANOMALY * anomaly_evidence
        + CONFIDENCE_WEIGHT_RULES * rule_evidence
        + CONFIDENCE_WEIGHT_SUPPORTING * supporting_evidence
        + CONFIDENCE_WEIGHT_CONSISTENCY * consistency_evidence
    )
    # Constrained to [0, 1] (Requirement #4). Given every component is
    # already in [0, 1] and the weights sum to 1.0, this clip is a
    # defensive safety net rather than something normally needed.
    score = round(min(max(raw_score, 0.0), 1.0), 4)

    level = determine_confidence_level(score)
    strength = determine_evidence_strength(anomalous, rule_count, supporting_count)
    human_review = determine_human_review_recommended(level, anomalous, rule_count, consistency_tier)

    components = {
        "anomaly_evidence": EvidenceComponent(
            round(anomaly_evidence, 4), CONFIDENCE_WEIGHT_ANOMALY, round(anomaly_evidence * CONFIDENCE_WEIGHT_ANOMALY, 4)
        ),
        "rule_evidence": EvidenceComponent(
            round(rule_evidence, 4), CONFIDENCE_WEIGHT_RULES, round(rule_evidence * CONFIDENCE_WEIGHT_RULES, 4)
        ),
        "supporting_evidence": EvidenceComponent(
            round(supporting_evidence, 4),
            CONFIDENCE_WEIGHT_SUPPORTING,
            round(supporting_evidence * CONFIDENCE_WEIGHT_SUPPORTING, 4),
        ),
        "consistency_evidence": EvidenceComponent(
            round(consistency_evidence, 4),
            CONFIDENCE_WEIGHT_CONSISTENCY,
            round(consistency_evidence * CONFIDENCE_WEIGHT_CONSISTENCY, 4),
        ),
    }

    return ConfidenceRecord(
        confidence_score=score,
        confidence_level=level,
        evidence_strength=strength,
        supporting_indicators=build_supporting_indicators(record),
        confidence_reasons=build_confidence_reasons(
            anomalous, normalized_score, anomaly_evidence, rule_count, strong_rule, supporting_count, consistency_tier
        ),
        uncertainty_reason=build_uncertainty_reason(anomalous, rule_count, consistency_tier, level),
        human_review_recommended=human_review,
        components=components,
    )


def run_confidence_scoring(rule_result: RuleEngineResult) -> ConfidenceEngineResult:
    """
    Full Phase 4 pipeline: batch-normalize `anomaly_score` once, then run
    `score_confidence()` for every event already classified by Phase 3.

    Never re-runs preprocessing, the anomaly model, or the rule engine -
    `rule_result` is passed in already computed (Requirement: do not
    modify or duplicate Phase 2/Phase 3 logic).
    """
    results_df = rule_result.results_df

    if results_df.empty:
        empty = pd.DataFrame()
        return ConfidenceEngineResult(
            results_df=empty,
            total_events=0,
            high_confidence_count=0,
            medium_confidence_count=0,
            low_confidence_count=0,
            human_review_count=0,
        )

    normalized_scores = normalize_anomaly_scores(results_df["anomaly_score"])

    confidence_scores: List[float] = []
    confidence_levels: List[str] = []
    evidence_strengths: List[str] = []
    supporting_indicators_col: List[List[str]] = []
    confidence_reasons_col: List[List[str]] = []
    uncertainty_reasons: List[str] = []
    human_review_col: List[bool] = []
    confidence_records: List[ConfidenceRecord] = []

    for idx, row in results_df.iterrows():
        record: InferenceRecord = row["_inference"]
        norm_score = float(normalized_scores.loc[idx])
        conf = score_confidence(record, anomaly_score_normalized=norm_score)

        confidence_records.append(conf)
        confidence_scores.append(conf.confidence_score)
        confidence_levels.append(conf.confidence_level)
        evidence_strengths.append(conf.evidence_strength)
        supporting_indicators_col.append(conf.supporting_indicators)
        confidence_reasons_col.append(conf.confidence_reasons)
        uncertainty_reasons.append(conf.uncertainty_reason)
        human_review_col.append(conf.human_review_recommended)

    out_df = results_df.copy()
    out_df["confidence_score"] = confidence_scores
    out_df["confidence_level"] = confidence_levels
    out_df["evidence_strength"] = evidence_strengths
    out_df["supporting_indicators"] = supporting_indicators_col
    out_df["confidence_reasons"] = confidence_reasons_col
    out_df["uncertainty_reason"] = uncertainty_reasons
    out_df["human_review_recommended"] = human_review_col
    out_df["_confidence"] = confidence_records  # full record, for the UI detail view

    total = len(out_df)
    high_count = int((out_df["confidence_level"] == "HIGH").sum())
    medium_count = int((out_df["confidence_level"] == "MEDIUM").sum())
    low_count = int((out_df["confidence_level"] == "LOW").sum())
    review_count = int(sum(out_df["human_review_recommended"]))

    return ConfidenceEngineResult(
        results_df=out_df,
        total_events=total,
        high_confidence_count=high_count,
        medium_confidence_count=medium_count,
        low_confidence_count=low_count,
        human_review_count=review_count,
    )
