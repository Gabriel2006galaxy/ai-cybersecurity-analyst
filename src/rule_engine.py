"""
rule_engine.py
---------------
Phase 3 - Knowledge-based rule engine & incident classification.

    Phase 2 Anomaly Detection
            |
      Observed Evidence
            |
      Derived Facts
            |
      Knowledge / Rules
            |
      Inference
            |
      Incident Classification
            |
      Initial Severity

This module is a small, transparent knowledge-based system: it turns each
processed event into explicit boolean FACTS, evaluates a fixed set of
hand-written RULES against those facts, and uses the rules that fired to
deterministically pick an incident category and an initial severity. No
machine learning happens here - every inference can be traced back to a
named rule and the exact evidence that satisfied it, which is the point
of using a rule engine at all (explainability).

IMPORTANT - true_label:
    Nothing in this module reads `config.LABEL_COLUMN`. Facts are derived
    only from the observed event's own fields and from Phase 2's
    `anomaly_status` (itself produced without true_label - see
    `src/anomaly_detector.py`). Rule thresholds (config.py) were chosen
    from how the synthetic dataset was constructed and general defensive-
    security reasoning, not by tuning against true_label/evaluation
    metrics.

Reuse, not duplication:
    `run_rule_engine()` takes the already-computed Phase 1
    `PreprocessingResult` (for the engineered features `is_off_hours` and
    `location_mismatch`, computed once in Phase 1) and the already-computed
    Phase 2 `AnomalyDetectionResult` (for `anomaly_status`/`anomaly_score`)
    and simply joins them by row index. It never recomputes preprocessing
    or retrains/re-runs the anomaly model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from config import (
    COMMON_PORTS,
    DATA_TRANSFER_THRESHOLD,
    DEVICE_COUNT_THRESHOLD,
    FAILED_LOGIN_THRESHOLD,
    NETWORK_SCAN_EVENT_TYPES,
    PRIVILEGE_ESCALATION_EVENT_TYPES,
    REQUEST_COUNT_THRESHOLD,
    SEVERE_DATA_TRANSFER_THRESHOLD,
)
from src.anomaly_detector import AnomalyDetectionResult
from src.preprocessor import PreprocessingResult

# =======================================================================
# Knowledge representation: facts
# =======================================================================
# Every fact is a plain boolean, derived only from the observed event and
# Phase 2's anomaly_status - never from true_label. FACT_DESCRIPTIONS is
# used by the UI to explain what each fact name means.
FACT_DESCRIPTIONS: Dict[str, str] = {
    "anomalous_event": "Phase 2's Isolation Forest flagged this event as ANOMALOUS.",
    "high_failed_logins": f"failed_logins >= {FAILED_LOGIN_THRESHOLD}.",
    "high_request_count": f"request_count >= {REQUEST_COUNT_THRESHOLD}.",
    "unusual_port": f"port is outside the common-ports list {COMMON_PORTS}.",
    "high_data_transfer": f"data_transferred_mb >= {DATA_TRANSFER_THRESHOLD}.",
    "off_hours_activity": "Event occurred during off-hours (Phase 1's is_off_hours).",
    "location_mismatch": "login_location differs from this user's usual location (Phase 1).",
    "multiple_devices": f"device_count >= {DEVICE_COUNT_THRESHOLD}.",
    "privilege_escalation_attempt": f"event_type is one of {sorted(PRIVILEGE_ESCALATION_EVENT_TYPES)}.",
    "network_scan_indicator": f"event_type is one of {sorted(NETWORK_SCAN_EVENT_TYPES)}.",
}


def extract_facts(row: pd.Series) -> Dict[str, bool]:
    """
    Derive the boolean fact set for a single processed event.

    `row` is expected to come from the DataFrame built by
    `run_rule_engine()` (Phase 1 engineered features + Phase 2
    predictions), but every field is read defensively with `.get(...)` and
    a safe default, so a row missing an optional column never crashes -
    it just can't support the facts that depend on that column.
    """
    def _get(col, default=0):
        # pd.Series.get() already returns `default` for a missing key, but
        # it returns NaN (not `default`) for a present-but-empty value, so
        # normalize that case too.
        value = row.get(col, default)
        return default if pd.isna(value) else value

    facts: Dict[str, bool] = {
        "anomalous_event": _get("anomaly_status", "NORMAL") == "ANOMALOUS",
        "high_failed_logins": _get("failed_logins", 0) >= FAILED_LOGIN_THRESHOLD,
        "high_request_count": _get("request_count", 0) >= REQUEST_COUNT_THRESHOLD,
        "unusual_port": _get("port", COMMON_PORTS[0]) not in COMMON_PORTS,
        "high_data_transfer": _get("data_transferred_mb", 0.0) >= DATA_TRANSFER_THRESHOLD,
        "off_hours_activity": bool(_get("is_off_hours", 0)),
        "location_mismatch": bool(_get("location_mismatch", 0)),
        "multiple_devices": _get("device_count", 0) >= DEVICE_COUNT_THRESHOLD,
        "privilege_escalation_attempt": _get("event_type", "") in PRIVILEGE_ESCALATION_EVENT_TYPES,
        "network_scan_indicator": _get("event_type", "") in NETWORK_SCAN_EVENT_TYPES,
    }
    return facts


# =======================================================================
# Knowledge representation: rules
# =======================================================================
@dataclass
class Rule:
    """
    A single knowledge-base rule.

    rule_id      : stable identifier shown to the analyst, e.g. "RULE-BF-01"
    name         : short human label, e.g. "Possible Brute Force"
    condition    : function(facts, row) -> bool - the precise IF clause
    conclusion   : the inferred fact name this rule adds, e.g. "possible_brute_force"
    incident_type: which of the fixed incident categories this rule maps to
    condition_text: the exact logical form, for display ("AND"/"OR" spelled out)
    explain      : function(row) -> str - a human-readable explanation using
                   the row's actual observed values
    """

    rule_id: str
    name: str
    condition: Callable[[Dict[str, bool], pd.Series], bool]
    conclusion: str
    incident_type: str
    condition_text: str
    explain: Callable[[pd.Series], str]


RULES: List[Rule] = [
    Rule(
        rule_id="RULE-BF-01",
        name="Possible Brute Force",
        condition_text="anomalous_event AND high_failed_logins",
        condition=lambda f, r: f["anomalous_event"] and f["high_failed_logins"],
        conclusion="possible_brute_force",
        incident_type="Possible Brute Force",
        explain=lambda r: (
            f"failed_logins = {r.get('failed_logins')} >= threshold "
            f"{FAILED_LOGIN_THRESHOLD}, on an event already flagged ANOMALOUS "
            f"by Phase 2 (anomaly_score = {r.get('anomaly_score', 0.0):.3f})."
        ),
    ),
    Rule(
        rule_id="RULE-NET-01",
        name="Unusual Network Activity",
        condition_text="anomalous_event AND (high_request_count OR network_scan_indicator)",
        condition=lambda f, r: f["anomalous_event"] and (f["high_request_count"] or f["network_scan_indicator"]),
        conclusion="unusual_network_activity",
        incident_type="Unusual Network Activity",
        explain=lambda r: (
            f"request_count = {r.get('request_count')} "
            f"(threshold {REQUEST_COUNT_THRESHOLD}) and/or event_type = "
            f"'{r.get('event_type')}' indicates scanning, on an ANOMALOUS event."
        ),
    ),
    Rule(
        rule_id="RULE-DATA-01",
        name="Suspicious Data Transfer",
        condition_text="anomalous_event AND high_data_transfer",
        condition=lambda f, r: f["anomalous_event"] and f["high_data_transfer"],
        conclusion="suspicious_data_transfer",
        incident_type="Suspicious Data Transfer",
        explain=lambda r: (
            f"data_transferred_mb = {r.get('data_transferred_mb', 0.0):.1f} >= threshold "
            f"{DATA_TRANSFER_THRESHOLD}, on an ANOMALOUS event."
        ),
    ),
    Rule(
        rule_id="RULE-LOC-01",
        name="Suspicious Login Location",
        condition_text="anomalous_event AND location_mismatch",
        condition=lambda f, r: f["anomalous_event"] and f["location_mismatch"],
        conclusion="suspicious_login_location",
        incident_type="Suspicious Login",
        explain=lambda r: (
            f"login_location = '{r.get('login_location')}' differs from this "
            f"user's usual location, on an ANOMALOUS event."
        ),
    ),
    Rule(
        rule_id="RULE-PRIV-01",
        name="Off-Hours Privilege Activity",
        condition_text="anomalous_event AND off_hours_activity AND privilege_escalation_attempt",
        condition=lambda f, r: (
            f["anomalous_event"] and f["off_hours_activity"] and f["privilege_escalation_attempt"]
        ),
        conclusion="off_hours_privilege_activity",
        incident_type="Other Anomaly",
        explain=lambda r: (
            f"event_type = '{r.get('event_type')}' occurred off-hours "
            f"(hour = {r.get('hour')}), on an ANOMALOUS event - a combination "
            f"treated as high-risk regardless of category."
        ),
    ),
    Rule(
        rule_id="RULE-DEV-01",
        name="Unusual Multi-Device Activity",
        condition_text="anomalous_event AND multiple_devices",
        condition=lambda f, r: f["anomalous_event"] and f["multiple_devices"],
        conclusion="unusual_multi_device_activity",
        incident_type="Other Anomaly",
        explain=lambda r: (
            f"device_count = {r.get('device_count')} >= threshold "
            f"{DEVICE_COUNT_THRESHOLD}, on an ANOMALOUS event "
            "(possible shared/compromised credentials)."
        ),
    ),
    Rule(
        rule_id="RULE-SCAN-01",
        name="Port Scanning Indicator",
        condition_text="anomalous_event AND (network_scan_indicator OR (unusual_port AND high_request_count))",
        condition=lambda f, r: (
            f["anomalous_event"] and (f["network_scan_indicator"] or (f["unusual_port"] and f["high_request_count"]))
        ),
        conclusion="port_scanning_indicator",
        incident_type="Port Scanning Indicator",
        explain=lambda r: (
            f"event_type = '{r.get('event_type')}' indicates scanning, and/or "
            f"port = {r.get('port')} is unusual together with "
            f"request_count = {r.get('request_count')}, on an ANOMALOUS event."
        ),
    ),
]

RULES_BY_ID: Dict[str, Rule] = {rule.rule_id: rule for rule in RULES}

# --- Fixed incident categories (exactly these seven) ---
INCIDENT_TYPES: List[str] = [
    "Normal Activity",
    "Suspicious Login",
    "Possible Brute Force",
    "Unusual Network Activity",
    "Suspicious Data Transfer",
    "Port Scanning Indicator",
    "Other Anomaly",
]

# --- Deterministic priority order used ONLY to pick a single primary
# incident_type when more than one rule (of different categories) fires on
# the same event. Lower index = higher priority. "Normal Activity" is
# never in this list - it's reached only when anomalous_event is False,
# before any rule is even evaluated, so it can never compete with a rule.
#
# Rationale (documented, not arbitrary):
#   1. Possible Brute Force        - an active credential-guessing attempt;
#                                     the most time-sensitive pattern.
#   2. Port Scanning Indicator     - active reconnaissance, often a
#                                     precursor to a more damaging attack.
#   3. Suspicious Data Transfer    - potential active data exfiltration.
#   4. Suspicious Login            - an identity/access anomaly
#                                     (e.g. impossible travel).
#   5. Unusual Network Activity    - a broad traffic-volume signal; ranked
#                                     below the more specific techniques
#                                     above so, e.g., a brute-force attempt
#                                     that also has a high request count is
#                                     still reported as "Possible Brute
#                                     Force" rather than the more generic
#                                     "Unusual Network Activity".
#   6. Other Anomaly                - meaningful (off-hours privilege
#                                     escalation, multi-device activity)
#                                     but not one of the specifically
#                                     named categories above.
INCIDENT_TYPE_PRIORITY: List[str] = [
    "Possible Brute Force",
    "Port Scanning Indicator",
    "Suspicious Data Transfer",
    "Suspicious Login",
    "Unusual Network Activity",
    "Other Anomaly",
]

# Rule conclusions that, on their own, are considered strong/high-risk
# indicators for severity purposes (see determine_severity()).
HIGH_RISK_CONCLUSIONS = {"off_hours_privilege_activity"}


# =======================================================================
# Result containers
# =======================================================================
@dataclass
class TriggeredRule:
    rule_id: str
    name: str
    conclusion: str
    incident_type: str
    explanation: str


@dataclass
class InferenceRecord:
    """Everything needed to explain ONE event's classification."""

    facts: Dict[str, bool] = field(default_factory=dict)
    derived_facts: List[str] = field(default_factory=list)         # fact names that are True
    triggered_rules: List[TriggeredRule] = field(default_factory=list)
    incident_type: str = "Normal Activity"
    severity: str = "N/A"
    reason: str = ""
    evidence: Dict = field(default_factory=dict)


@dataclass
class RuleEngineResult:
    results_df: pd.DataFrame
    rules: List[Rule]
    total_events: int
    normal_count: int
    low_count: int
    medium_count: int
    high_count: int


# =======================================================================
# Rule evaluation
# =======================================================================
def evaluate_rules(facts: Dict[str, bool], row: pd.Series) -> List[TriggeredRule]:
    """
    Evaluate every rule against `facts`/`row` and return ALL rules that
    fired (never stops at the first match - Requirement #4).
    """
    triggered: List[TriggeredRule] = []
    for rule in RULES:
        if rule.condition(facts, row):
            triggered.append(
                TriggeredRule(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    conclusion=rule.conclusion,
                    incident_type=rule.incident_type,
                    explanation=rule.explain(row),
                )
            )
    return triggered


def classify_incident(anomalous: bool, triggered_rules: List[TriggeredRule]) -> str:
    """
    Deterministic incident classification:
      - not anomalous                -> "Normal Activity"
      - anomalous, no rule fired      -> "Other Anomaly"
      - anomalous, rule(s) fired      -> the highest-priority matched
                                          incident_type (see
                                          INCIDENT_TYPE_PRIORITY above)
    """
    if not anomalous:
        return "Normal Activity"
    if not triggered_rules:
        return "Other Anomaly"

    matched_types = {tr.incident_type for tr in triggered_rules}
    for incident_type in INCIDENT_TYPE_PRIORITY:
        if incident_type in matched_types:
            return incident_type
    return "Other Anomaly"  # safety net; every rule's incident_type is in the priority list


def determine_severity(
    anomalous: bool, triggered_rules: List[TriggeredRule], row: pd.Series
) -> str:
    """
    Deterministic initial severity - NOT a probability, purely rule-based:

      N/A    : event is not anomalous (severity doesn't apply)
      LOW    : anomalous, but no specific rule matched (weak/generic anomaly)
      MEDIUM : exactly one rule matched, and it doesn't meet a HIGH condition
      HIGH   : any of -
                 (a) 2 or more distinct rules matched (multiple corroborating
                     suspicious indicators), OR
                 (b) the off-hours privilege-escalation rule matched
                     (RULE-PRIV-01 -> "off_hours_privilege_activity"), OR
                 (c) the suspicious-data-transfer rule matched AND
                     data_transferred_mb >= SEVERE_DATA_TRANSFER_THRESHOLD
                     ("severe data-transfer behavior")
    """
    if not anomalous:
        return "N/A"
    if not triggered_rules:
        return "LOW"

    conclusions = {tr.conclusion for tr in triggered_rules}
    severe_transfer = (
        "suspicious_data_transfer" in conclusions
        and row.get("data_transferred_mb", 0) >= SEVERE_DATA_TRANSFER_THRESHOLD
    )
    is_high = (
        len(triggered_rules) >= 2
        or bool(conclusions & HIGH_RISK_CONCLUSIONS)
        or severe_transfer
    )
    return "HIGH" if is_high else "MEDIUM"


def build_reason(incident_type: str, severity: str, triggered_rules: List[TriggeredRule]) -> str:
    """One-line, analyst-facing summary of the classification."""
    if incident_type == "Normal Activity":
        return "Not flagged as anomalous by the Phase 2 anomaly detector."
    if not triggered_rules:
        return (
            "Flagged ANOMALOUS by Phase 2, but no specific rule pattern "
            "matched - classified as a generic anomaly."
        )
    rule_ids = ", ".join(tr.rule_id for tr in triggered_rules)
    return f"{incident_type} ({severity}) - matched rule(s): {rule_ids}."


# =======================================================================
# Per-row inference + orchestration
# =======================================================================
EVIDENCE_FIELDS = [
    "failed_logins",
    "request_count",
    "port",
    "data_transferred_mb",
    "session_duration",
    "device_count",
    "event_type",
    "login_location",
    "hour",
    "is_off_hours",
    "location_mismatch",
    "anomaly_score",
    "anomaly_status",
]


def run_inference(row: pd.Series) -> InferenceRecord:
    """Run the full Evidence -> Facts -> Rules -> Inference chain for one event."""
    facts = extract_facts(row)
    derived_facts = [name for name, value in facts.items() if value]
    triggered = evaluate_rules(facts, row)
    incident_type = classify_incident(facts["anomalous_event"], triggered)
    severity = determine_severity(facts["anomalous_event"], triggered, row)
    reason = build_reason(incident_type, severity, triggered)
    evidence = {field: row.get(field) for field in EVIDENCE_FIELDS if field in row.index}

    return InferenceRecord(
        facts=facts,
        derived_facts=derived_facts,
        triggered_rules=triggered,
        incident_type=incident_type,
        severity=severity,
        reason=reason,
        evidence=evidence,
    )


def run_rule_engine(
    preprocessing_result: PreprocessingResult, anomaly_result: AnomalyDetectionResult
) -> RuleEngineResult:
    """
    Full Phase 3 pipeline: join Phase 1's engineered features with Phase 2's
    predictions (by row index - both are built from the same cleaned_df),
    then run Evidence -> Facts -> Rules -> Inference for every event.

    Never re-runs preprocessing or the anomaly model - both are passed in
    already computed.
    """
    feature_df = preprocessing_result.feature_df
    predictions = anomaly_result.results_df[["anomaly_prediction", "anomaly_score", "anomaly_status"]]

    if feature_df.empty:
        empty = pd.DataFrame()
        return RuleEngineResult(
            results_df=empty, rules=RULES, total_events=0,
            normal_count=0, low_count=0, medium_count=0, high_count=0,
        )

    combined = feature_df.join(predictions, how="left")

    incident_types: List[str] = []
    severities: List[str] = []
    triggered_rule_ids: List[List[str]] = []
    triggered_rule_labels: List[str] = []
    derived_facts_col: List[List[str]] = []
    reasons: List[str] = []
    evidence_col: List[Dict] = []
    inference_records: List[InferenceRecord] = []

    for _, row in combined.iterrows():
        record = run_inference(row)
        inference_records.append(record)
        incident_types.append(record.incident_type)
        severities.append(record.severity)
        triggered_rule_ids.append([tr.rule_id for tr in record.triggered_rules])
        triggered_rule_labels.append(
            ", ".join(tr.rule_id for tr in record.triggered_rules) if record.triggered_rules else "-"
        )
        derived_facts_col.append(record.derived_facts)
        reasons.append(record.reason)
        evidence_col.append(record.evidence)

    results_df = combined.copy()
    results_df["incident_type"] = incident_types
    results_df["severity"] = severities
    results_df["triggered_rule_ids"] = triggered_rule_ids
    results_df["triggered_rules"] = triggered_rule_labels
    results_df["derived_facts"] = derived_facts_col
    results_df["reason"] = reasons
    results_df["evidence"] = evidence_col
    results_df["_inference"] = inference_records  # full record, for the UI detail view

    total = len(results_df)
    normal_count = int((results_df["incident_type"] == "Normal Activity").sum())
    low_count = int((results_df["severity"] == "LOW").sum())
    medium_count = int((results_df["severity"] == "MEDIUM").sum())
    high_count = int((results_df["severity"] == "HIGH").sum())

    return RuleEngineResult(
        results_df=results_df,
        rules=RULES,
        total_events=total,
        normal_count=normal_count,
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
    )
