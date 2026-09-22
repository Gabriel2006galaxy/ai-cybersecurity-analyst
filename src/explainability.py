"""
explainability.py
-------------------
Phase 10 - Explainability layer over the EXISTING outputs of Phases 2-9.

    Anomaly Result (Phase 2)
            |
    Rule / Classification Result (Phase 3)
            |
    Confidence Result (Phase 4)
            |
    RAG Retrieval Result (Phase 8)
            |
    GenAI Result (Phase 5)
            |
    Response Plan (Phase 6)
            |
    Human Review (Phase 7)
            |
      Structured Explanation           (this module)

This module builds plain, structured explanations from objects that were
ALREADY computed by earlier phases. It never re-runs the anomaly
detector, never re-evaluates rules, never recomputes a confidence score,
never performs RAG retrieval, never calls the GenAI model, and never
generates a response plan - every function below takes an already-built
`InferenceRecord` / `ConfidenceRecord` / `RetrievalResult` / `GenAIResult`
/ `ResponsePlanResult` (or a plain feedback row) and reads from it only.

IMPORTANT - never invents a reason:
    Every explanation string below is built by directly interpolating
    fields that already exist on the object it explains (the same
    `explain()` text a rule already carries, the same `confidence_reasons`
    Phase 4 already computed, the same `analyst_notes` Phase 7 already
    stored, etc.). Nothing here introduces a new fact, a new number, or a
    plausible-sounding justification that isn't already present on the
    input record - a stage that has no evidence for something simply says
    so, rather than making something up.

IMPORTANT - true_label:
    Never read here. None of `InferenceRecord`, `ConfidenceRecord`,
    `RetrievalResult`, `GenAIResult`, or `ResponsePlanResult` has a
    `true_label` field to begin with (see the Phase 3/4/8/5/6 modules),
    and `config.LABEL_COLUMN` is never imported into this module.

IMPORTANT - distinguishing SYSTEM facts from RAG, GenAI text, and human
decision (Phase 10 UI requirement #6):
    `AnomalyExplanation`, `RuleExplanation`, `ClassificationExplanation`,
    `SeverityExplanation`, `ConfidenceExplanation`, `SupportingIndicators
    Explanation`, and `UncertaintyExplanation` describe SYSTEM-GENERATED
    FACTS (Phases 2-4 - deterministic, rule/formula-based). `RagExplanation`
    describes RAG REFERENCE MATERIAL (Phase 8 - retrieved background text,
    never a decision). `GenAIExplanation` describes GENERATIVE AI TEXT
    (Phase 5 - advisory explanation of the system facts, explicitly
    labeled with whether it is the AI's own text or the deterministic
    fallback). `HumanReviewExplanation` describes the HUMAN ANALYST
    DECISION (Phase 7 - the only place a real decision is made). Keeping
    these as separate dataclasses, rather than one flat blob, is what
    keeps that distinction visible on the Explainability page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.confidence import ConfidenceRecord
from src.genai_analyzer import GenAIResult
from src.response_planner import ResponsePlanResult
from src.retriever import RetrievalResult
from src.rule_engine import InferenceRecord

# =======================================================================
# Result containers - one per explanation area required by the spec
# =======================================================================
@dataclass
class AnomalyExplanation:
    """WHY an incident was classified as normal/anomalous (Requirement #1)."""

    anomaly_status: str
    anomaly_score: float
    explanation: str


@dataclass
class RuleExplanation:
    """One triggered rule, exactly as Phase 3 already explained it -
    never re-derived or reworded here."""

    rule_id: str
    name: str
    conclusion: str
    explanation: str


@dataclass
class ClassificationExplanation:
    """WHICH rules contributed to the classification, and the resulting
    incident_type (Requirement #2)."""

    incident_type: str
    reason: str
    triggered_rules: List[RuleExplanation] = field(default_factory=list)


@dataclass
class SeverityExplanation:
    """HOW severity was determined (Requirement #3)."""

    severity: str
    explanation: str


@dataclass
class ConfidenceExplanation:
    """HOW confidence was determined (Requirement #4)."""

    confidence_score: float
    confidence_level: str
    evidence_strength: str
    reasons: List[str] = field(default_factory=list)
    components: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: str = ""


@dataclass
class SupportingIndicatorsExplanation:
    """The supporting evidence behind the confidence assessment."""

    indicators: List[str] = field(default_factory=list)


@dataclass
class UncertaintyExplanation:
    """Why this classification should (or shouldn't) be treated with
    caution, and whether human review is recommended."""

    uncertainty_reason: str
    human_review_recommended: bool


@dataclass
class RagExplanation:
    """WHICH RAG knowledge sources were retrieved (Requirement #5) - RAG
    REFERENCE MATERIAL, never a decision."""

    ran: bool  # whether retrieval was even performed for this view
    query_text: str = ""
    items: List[Dict] = field(default_factory=list)  # knowledge_id/title/topic/similarity_score/source_type
    count: int = 0
    summary: str = ""


@dataclass
class GenAIExplanation:
    """WHAT information was given to the GenAI module, and how it
    responded (Requirement #6) - GENERATIVE AI TEXT, clearly marked as
    either the model's own output or the deterministic fallback, and
    never able to change the authoritative values restated alongside it."""

    ran: bool  # whether a GenAI result exists for this view
    model_used: Optional[str] = None
    source: str = ""  # "ollama" | "ollama_text_fallback" | "fallback"
    used_fallback: bool = True
    retrieved_knowledge_supplied: bool = False
    retrieved_knowledge_count: int = 0
    authoritative_incident_type: str = ""
    authoritative_severity: str = ""
    authoritative_confidence_level: str = ""
    summary: str = ""


@dataclass
class ResponsePlanExplanation:
    """WHAT response plan was generated, and why (Requirement #7)."""

    ran: bool
    plan_title: str = ""
    priority: str = ""
    planning_reason: str = ""
    step_count: int = 0
    review_required: bool = False
    summary: str = ""


@dataclass
class HumanReviewExplanation:
    """WHAT analyst decision was eventually recorded (Requirement #8) -
    the HUMAN ANALYST DECISION, the only place a real decision is made."""

    reviewed: bool
    analyst_decision: Optional[str] = None
    review_status: Optional[str] = None
    analyst_notes: Optional[str] = None
    reviewed_at: Optional[str] = None
    summary: str = ""


@dataclass
class IncidentExplanation:
    """Everything the Explainability & Audit Trail page needs for one
    incident, bundled into a single object."""

    incident_id: str
    anomaly: AnomalyExplanation
    classification: ClassificationExplanation
    severity: SeverityExplanation
    confidence: ConfidenceExplanation
    supporting_indicators: SupportingIndicatorsExplanation
    uncertainty: UncertaintyExplanation
    rag: RagExplanation
    genai: GenAIExplanation
    response_plan: ResponsePlanExplanation
    human_review: HumanReviewExplanation


# =======================================================================
# 1. Anomaly detection explanation (Requirement #1)
# =======================================================================
def explain_anomaly_detection(record: InferenceRecord) -> AnomalyExplanation:
    """Explain Phase 2's anomaly verdict using only the values Phase 3
    already carried forward on `record.evidence` - never re-run the
    Isolation Forest."""
    status = str(record.evidence.get("anomaly_status", "NORMAL"))
    score = float(record.evidence.get("anomaly_score", 0.0) or 0.0)
    if status == "ANOMALOUS":
        text = (
            f"The Isolation Forest marked this event as ANOMALOUS "
            f"(anomaly_score = {score:.3f}) because its derived feature "
            "pattern differed from the learned normal baseline more than "
            "the configured contamination threshold allows."
        )
    else:
        text = (
            f"The Isolation Forest marked this event as NORMAL "
            f"(anomaly_score = {score:.3f}) - its derived feature pattern "
            "was consistent with the learned normal baseline."
        )
    return AnomalyExplanation(anomaly_status=status, anomaly_score=score, explanation=text)


# =======================================================================
# 2. Rule evidence + classification explanation (Requirement #2)
# =======================================================================
def explain_rule_evidence(record: InferenceRecord) -> List[RuleExplanation]:
    """One `RuleExplanation` per rule Phase 3 already found to have
    triggered - `tr.explanation` is reused verbatim (it was already built
    from the event's real observed values by `src/rule_engine.py`), never
    reworded or re-derived here."""
    return [
        RuleExplanation(
            rule_id=tr.rule_id,
            name=tr.name,
            conclusion=tr.conclusion,
            explanation=f"{tr.rule_id} triggered because {tr.explanation}",
        )
        for tr in record.triggered_rules
    ]


def explain_classification(record: InferenceRecord) -> ClassificationExplanation:
    """WHICH rules contributed to the incident_type Phase 3 already
    chose, using its own already-built `reason` string (see
    `src.rule_engine.build_reason()`) rather than a second explanation."""
    return ClassificationExplanation(
        incident_type=record.incident_type,
        reason=record.reason,
        triggered_rules=explain_rule_evidence(record),
    )


# =======================================================================
# 3. Severity explanation (Requirement #3)
# =======================================================================
def explain_severity(record: InferenceRecord) -> SeverityExplanation:
    severity = record.severity
    rule_ids = [tr.rule_id for tr in record.triggered_rules]
    if record.incident_type == "Normal Activity":
        text = "Severity is 'N/A' because this event was not classified as an incident (not flagged anomalous)."
    elif not rule_ids:
        text = (
            f"Severity was set to '{severity}' - the event was flagged ANOMALOUS by Phase 2, "
            "but no specific knowledge-base rule matched, so a generic LOW severity applies."
        )
    else:
        text = (
            f"Severity was set to '{severity}' by the knowledge-based rule engine, based on "
            f"{len(rule_ids)} triggered rule(s): {', '.join(rule_ids)}."
        )
    return SeverityExplanation(severity=severity, explanation=text)


# =======================================================================
# 4. Confidence + supporting-indicator + uncertainty explanation
#    (Requirement #4, plus "supporting indicators" / "uncertainty
#    explanation" from the spec's required-structured-data list)
# =======================================================================
def explain_confidence(record: ConfidenceRecord) -> ConfidenceExplanation:
    components = {
        name: {"value": c.value, "weight": c.weight, "contribution": c.contribution}
        for name, c in record.components.items()
    }
    reasons_text = " ".join(record.confidence_reasons) if record.confidence_reasons else "No specific reasons were recorded."
    summary = (
        f"{record.confidence_level} confidence (score {record.confidence_score:.2f}, "
        f"'{record.evidence_strength}' evidence strength) resulted from the existing "
        f"evidence-based confidence components. {reasons_text}"
    ).strip()
    return ConfidenceExplanation(
        confidence_score=record.confidence_score,
        confidence_level=record.confidence_level,
        evidence_strength=record.evidence_strength,
        reasons=list(record.confidence_reasons),
        components=components,
        summary=summary,
    )


def explain_supporting_indicators(record: ConfidenceRecord) -> SupportingIndicatorsExplanation:
    return SupportingIndicatorsExplanation(indicators=list(record.supporting_indicators))


def explain_uncertainty(record: ConfidenceRecord) -> UncertaintyExplanation:
    return UncertaintyExplanation(
        uncertainty_reason=record.uncertainty_reason,
        human_review_recommended=record.human_review_recommended,
    )


# =======================================================================
# 5. RAG explanation (Requirement #5)
# =======================================================================
def explain_rag(retrieval_result: Optional[RetrievalResult]) -> RagExplanation:
    """
    Explain WHICH RAG knowledge sources were retrieved, purely from an
    already-computed `RetrievalResult` (Phase 8) - `retrieve()` /
    `retrieve_knowledge_for_incident()` are never called here. Accepts
    `None` for a view where retrieval simply hasn't happened yet for this
    incident (e.g. the analyst never opened Generative AI Analysis for
    it) - that is reported plainly, never as an empty retrieval.
    """
    if retrieval_result is None:
        return RagExplanation(ran=False, summary="RAG retrieval has not been run for this incident yet.")

    items = [
        {
            "knowledge_id": item.knowledge_id,
            "title": item.title,
            "topic": item.topic,
            "source_type": item.source_type,
            "similarity_score": item.similarity_score,
        }
        for item in retrieval_result.items
    ]
    if items:
        summary = (
            f"{len(items)} knowledge-base source(s) were retrieved for this incident: "
            + ", ".join(f"{i['knowledge_id']} ({i['title']}, similarity {i['similarity_score']:.2f})" for i in items)
        )
    else:
        summary = "No knowledge-base entry met the minimum similarity threshold for this incident."
    return RagExplanation(
        ran=True,
        query_text=retrieval_result.query_text,
        items=items,
        count=len(items),
        summary=summary,
    )


# =======================================================================
# 6. GenAI explanation (Requirement #6)
# =======================================================================
def explain_genai(
    genai_result: Optional[GenAIResult],
    inference_record: InferenceRecord,
    confidence_record: ConfidenceRecord,
) -> GenAIExplanation:
    """
    Explain WHAT information was given to the GenAI module and how it
    responded, purely from an already-computed `GenAIResult` (Phase 5) -
    `generate_ai_case_summary()` is never called here. The authoritative
    incident_type/severity/confidence_level are always taken from Phase
    3/4's own records (never from the GenAI result), so this explanation
    can never let AI-generated text redefine them. Accepts `None` for a
    view where no GenAI summary has been generated for this incident yet.
    """
    authoritative_incident_type = inference_record.incident_type
    authoritative_severity = inference_record.severity
    authoritative_confidence_level = confidence_record.confidence_level

    if genai_result is None:
        return GenAIExplanation(
            ran=False,
            authoritative_incident_type=authoritative_incident_type,
            authoritative_severity=authoritative_severity,
            authoritative_confidence_level=authoritative_confidence_level,
            summary="No Generative AI summary has been generated for this incident yet.",
        )

    if genai_result.used_fallback:
        summary = (
            "The deterministic, evidence-based fallback was used (Generative AI was unavailable "
            "or its response could not be safely used) - "
        )
    else:
        summary = f"A local Ollama model ('{genai_result.model_used}') generated this explanation - "
    summary += (
        f"it explains incident_type='{authoritative_incident_type}', severity='{authoritative_severity}', "
        f"confidence_level='{authoritative_confidence_level}' exactly as already decided by the system; "
        f"{genai_result.retrieved_knowledge_count} retrieved knowledge source(s) were supplied as background context."
    )

    return GenAIExplanation(
        ran=True,
        model_used=genai_result.model_used,
        source=genai_result.source,
        used_fallback=genai_result.used_fallback,
        retrieved_knowledge_supplied=genai_result.retrieved_knowledge_count > 0,
        retrieved_knowledge_count=genai_result.retrieved_knowledge_count,
        authoritative_incident_type=authoritative_incident_type,
        authoritative_severity=authoritative_severity,
        authoritative_confidence_level=authoritative_confidence_level,
        summary=summary,
    )


# =======================================================================
# 7. Response-plan explanation (Requirement #7)
# =======================================================================
def explain_response_plan(plan: Optional[ResponsePlanResult]) -> ResponsePlanExplanation:
    """
    Explain WHAT response plan was generated and why, purely from an
    already-computed `ResponsePlanResult` (Phase 6) -
    `generate_response_plan()` is never called here.
    """
    if plan is None:
        return ResponsePlanExplanation(ran=False, summary="No response plan has been generated for this incident yet.")

    summary = (
        f"'{plan.plan_title}' ({plan.priority} priority, {len(plan.steps)} step(s)) was selected because: "
        f"{plan.planning_reason}"
    )
    return ResponsePlanExplanation(
        ran=True,
        plan_title=plan.plan_title,
        priority=plan.priority,
        planning_reason=plan.planning_reason,
        step_count=len(plan.steps),
        review_required=plan.review_required,
        summary=summary,
    )


# =======================================================================
# 8. Human-review explanation (Requirement #8)
# =======================================================================
def explain_human_review(latest_feedback_row: Optional[Dict]) -> HumanReviewExplanation:
    """
    Explain WHAT analyst decision was eventually recorded, purely from an
    already-stored feedback row (Phase 7, e.g. from
    `src.feedback.get_review_status_map()` /
    `src.feedback.get_feedback_by_incident()`) - this module never reads
    from or writes to the feedback database itself. `latest_feedback_row`
    may be a plain dict or a pandas Series; only `.get(...)` and `len(...)`
    are used (never a direct `if latest_feedback_row:` truthiness check,
    which raises `ValueError` for a pandas Series), so either works, and
    `None`/empty (no review yet) is handled explicitly.
    """
    if latest_feedback_row is None or len(latest_feedback_row) == 0:
        return HumanReviewExplanation(reviewed=False, summary="This incident has not yet been reviewed by an analyst.")

    decision = latest_feedback_row.get("analyst_decision")
    status = latest_feedback_row.get("review_status")
    notes = latest_feedback_row.get("analyst_notes") or ""
    timestamp = latest_feedback_row.get("timestamp")
    summary = f"The analyst recorded a final decision of '{decision}' (status: {status}) at {timestamp}."
    if notes:
        summary += f" Analyst notes: {notes}"
    return HumanReviewExplanation(
        reviewed=True,
        analyst_decision=decision,
        review_status=status,
        analyst_notes=notes,
        reviewed_at=timestamp,
        summary=summary,
    )


# =======================================================================
# Top-level orchestration - the one function app.py needs per incident
# =======================================================================
def build_incident_explanation(
    incident_id: str,
    inference_record: InferenceRecord,
    confidence_record: ConfidenceRecord,
    retrieval_result: Optional[RetrievalResult] = None,
    genai_result: Optional[GenAIResult] = None,
    response_plan: Optional[ResponsePlanResult] = None,
    latest_feedback_row: Optional[Dict] = None,
) -> IncidentExplanation:
    """
    Assemble the full, structured explanation for one incident from
    whatever has ALREADY been computed for it elsewhere in the app. Every
    argument beyond `incident_id`/`inference_record`/`confidence_record`
    is optional and `None` by default - RAG/GenAI/a response plan/a human
    review may simply not have happened yet for this incident, and that
    is reported plainly rather than treated as an error.
    """
    return IncidentExplanation(
        incident_id=incident_id,
        anomaly=explain_anomaly_detection(inference_record),
        classification=explain_classification(inference_record),
        severity=explain_severity(inference_record),
        confidence=explain_confidence(confidence_record),
        supporting_indicators=explain_supporting_indicators(confidence_record),
        uncertainty=explain_uncertainty(confidence_record),
        rag=explain_rag(retrieval_result),
        genai=explain_genai(genai_result, inference_record, confidence_record),
        response_plan=explain_response_plan(response_plan),
        human_review=explain_human_review(latest_feedback_row),
    )
