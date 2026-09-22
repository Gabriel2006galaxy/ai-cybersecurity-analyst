"""
evaluation.py
-------------
Phase 11 - Evaluation & Performance Dashboard.

    Phase 2-10 outputs (already computed elsewhere)
            |
    src.evaluation  <- THIS MODULE: reads / aggregates only
            |
    Descriptive, measured metrics for the Evaluation & Performance page

This module answers one question: "how is the system actually behaving,
measured against what already exists?" It is deliberately NOT part of the
decision pipeline - nothing here changes an anomaly result, a rule
threshold, a confidence score, RAG settings, the GenAI model, a response
plan, or an analyst decision. It only reads outputs that Phases 2-10
already produced (or re-runs an existing, already-tested function such as
`retrieve_knowledge_for_incident()` or `evaluate_against_ground_truth()`
in exactly the same way every other page already does) and turns them
into descriptive counts, rates, and offline evaluation metrics.

IMPORTANT - true_label (config.LABEL_COLUMN):
    Every earlier phase (preprocessing, anomaly detection, the rule
    engine, confidence scoring, RAG, GenAI, response planning, analyst
    review, feedback storage, continuous improvement, explainability, the
    audit trail) has never read `true_label` as an input to a decision -
    that has been true since Phase 0 and remains true here. Phase 11 is
    the first phase that is explicitly allowed to READ `true_label`, and
    only for OFFLINE EVALUATION: measuring how well the unsupervised
    Isolation Forest agrees with the ground truth baked into this
    synthetic dataset, strictly after predictions already exist.

    In this codebase, `true_label` is read in exactly two places, both
    clearly marked "post-hoc"/"offline evaluation only", and neither is
    ever called from inside a decision path:
      1. `src.anomaly_detector.evaluate_against_ground_truth()` - added
         back in Phase 2 for this exact purpose, and unchanged here.
         `evaluate_anomaly_detection()` below simply REUSES it rather
         than recomputing accuracy/precision/recall/F1 from scratch.
      2. `evaluate_dataset_overview()` below, which reads `true_label`
         only to report the dataset's ground-truth normal/anomalous
         split for display - never to influence anything.
    Every metric derived from `true_label` is labeled "Offline
    Evaluation" wherever it is displayed (see `render_evaluation_page` in
    app.py), and every dataclass below that carries such a metric also
    carries a `measured`/`note` flag so the page can say plainly when a
    metric is NOT available rather than inventing one.

DO NOT FABRICATE METRICS:
    Every function below either (a) reads a count/rate directly off an
    already-computed result object (`AnomalyDetectionResult`,
    `RuleEngineResult`, `ConfidenceEngineResult`, `FeedbackSummary`, the
    audit trail's own event log), or (b) re-runs an existing, already-
    tested function in a simple loop for descriptive aggregation (RAG
    retrieval, once per anomalous event - the same call
    `render_genai_analysis_page()` already makes one incident at a time).
    Nothing is estimated, interpolated, or hard-coded. Where a metric
    genuinely cannot be computed from real project data (no `true_label`
    column, an empty audit trail, no analyst feedback yet, Ollama not
    running), the corresponding dataclass reports `measured=False` (or an
    equivalent explicit flag) with a clear `note` explaining why, instead
    of a fabricated value.

PERFORMANCE TIMING:
    `measure_anomaly_detection_timing()` and `measure_rag_retrieval_timing()`
    are cheap, deterministic, local-only operations (training/predicting
    on this dataset's few hundred rows, or TF-IDF retrieval) and are safe
    to run on every page load. `measure_genai_generation_timing()` may
    perform a real network call to a local Ollama server and is
    deliberately opt-in only (the Evaluation page calls it from a button,
    never automatically) - see its docstring. All three explicitly label
    their result as a local-machine sample, never a production benchmark,
    and none of them changes any existing AI behavior; they only wrap an
    already-existing function call with a timer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from config import AUDIT_STAGES, KNOWLEDGE_BASE_DIR, LABEL_COLUMN, OLLAMA_MODEL_NAME
from src.anomaly_detector import (
    AnomalyDetectionResult,
    evaluate_against_ground_truth,
    predict_anomalies,
    train_isolation_forest,
)
from src.audit_trail import load_audit_events
from src.confidence import ConfidenceEngineResult
from src.feedback import summarize_feedback
from src.genai_analyzer import GenAIResult, IncidentContext, generate_ai_case_summary, get_ollama_status
from src.retriever import KnowledgeEntry, load_knowledge_base, retrieve_knowledge_for_incident
from src.rule_engine import RuleEngineResult

# A GENAI_ANALYSIS audit event's descriptive_details is written by
# app.py's `_log_case_processing_audit()` as exactly
# "source=<source>, used_fallback=<True|False>." - parsed here (read-only)
# to derive descriptive GenAI counts from real, already-recorded audit
# history. Never used to write or change an audit event.
_GENAI_DETAILS_RE = re.compile(r"source=([^,]+),\s*used_fallback=(True|False)")


# =======================================================================
# Result containers
# =======================================================================
@dataclass
class DatasetOverview:
    total_events: int = 0
    predicted_normal_count: int = 0
    predicted_anomalous_count: int = 0
    ground_truth_available: bool = False
    ground_truth_normal_count: Optional[int] = None
    ground_truth_anomalous_count: Optional[int] = None


@dataclass
class AnomalyEvaluationMetrics:
    """Offline Evaluation (true_label) when `measured` is True."""

    measured: bool
    total_events: int = 0
    predicted_anomalous: int = 0
    predicted_normal: int = 0
    ground_truth_anomalous: int = 0
    ground_truth_normal: int = 0
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    confusion_matrix_labels: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ClassificationEvaluationMetrics:
    total_events: int = 0
    incident_type_counts: Dict[str, int] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    rule_firing_counts: Dict[str, int] = field(default_factory=dict)
    multi_rule_event_count: int = 0
    no_rule_anomaly_count: int = 0


@dataclass
class ConfidenceEvaluationMetrics:
    total_events: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    human_review_count: int = 0


@dataclass
class RagEvaluationMetrics:
    knowledge_base_size: int = 0
    retrieval_attempted_count: int = 0
    retrieval_hit_count: int = 0
    retrieval_empty_count: int = 0
    top_match_counts_by_incident_type: Dict[str, int] = field(default_factory=dict)
    controlled_eligible_count: int = 0
    controlled_success_count: int = 0
    controlled_success_rate: Optional[float] = None
    note: str = ""


@dataclass
class GenAIEvaluationMetrics:
    model_name: str = ""
    ollama_available: bool = False
    model_available: bool = False
    ollama_status_message: str = ""
    generations_attempted: int = 0
    successful_generations: int = 0
    fallback_generations: int = 0
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    measured: bool = False
    note: str = ""


@dataclass
class FeedbackEvaluationMetrics:
    total_reviews: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    modified_count: int = 0
    approval_rate: float = 0.0
    rejection_rate: float = 0.0
    modification_rate: float = 0.0


@dataclass
class AuditEvaluationMetrics:
    total_events: int = 0
    events_per_stage: Dict[str, int] = field(default_factory=dict)


@dataclass
class TimingMeasurement:
    """One local-machine timing sample - never a production benchmark."""

    stage: str
    measured: bool
    elapsed_seconds: Optional[float] = None
    sample_size: Optional[int] = None
    note: str = ""


@dataclass
class EvaluationReport:
    dataset_overview: DatasetOverview
    anomaly_metrics: AnomalyEvaluationMetrics
    classification_metrics: ClassificationEvaluationMetrics
    confidence_metrics: ConfidenceEvaluationMetrics
    rag_metrics: RagEvaluationMetrics
    genai_metrics: GenAIEvaluationMetrics
    feedback_metrics: FeedbackEvaluationMetrics
    audit_metrics: AuditEvaluationMetrics


# =======================================================================
# A. Dataset Overview
# =======================================================================
def evaluate_dataset_overview(anomaly_result: AnomalyDetectionResult) -> DatasetOverview:
    """
    Predicted counts never require `true_label` (they come straight off
    `AnomalyDetectionResult`, exactly as the AI Anomaly Detection page
    already shows them). The ground-truth split is read from
    `true_label` ONLY for descriptive display - OFFLINE EVALUATION.
    """
    df = anomaly_result.results_df
    ground_truth_available = LABEL_COLUMN in df.columns and not df.empty

    gt_normal: Optional[int] = None
    gt_anomalous: Optional[int] = None
    if ground_truth_available:
        gt_anomalous = int((df[LABEL_COLUMN] == "suspicious").sum())
        gt_normal = int(len(df) - gt_anomalous)

    return DatasetOverview(
        total_events=anomaly_result.total_events,
        predicted_normal_count=anomaly_result.normal_events,
        predicted_anomalous_count=anomaly_result.anomalous_events,
        ground_truth_available=ground_truth_available,
        ground_truth_normal_count=gt_normal,
        ground_truth_anomalous_count=gt_anomalous,
    )


# =======================================================================
# B. Anomaly Detection Performance (Offline Evaluation)
# =======================================================================
def evaluate_anomaly_detection(anomaly_result: AnomalyDetectionResult) -> AnomalyEvaluationMetrics:
    """
    REUSES `src.anomaly_detector.evaluate_against_ground_truth()` - the
    same, already-existing, already-tested post-hoc evaluation function
    used since Phase 2 - rather than recomputing accuracy/precision/
    recall/F1 from scratch. Never re-trains or re-runs the detector.
    """
    metrics = evaluate_against_ground_truth(anomaly_result.results_df)

    if metrics is None:
        return AnomalyEvaluationMetrics(
            measured=False,
            total_events=anomaly_result.total_events,
            predicted_anomalous=anomaly_result.anomalous_events,
            predicted_normal=anomaly_result.normal_events,
            note=(
                "No true_label column is available in this dataset, so "
                "accuracy/precision/recall/F1 cannot be measured. "
                "Prediction counts are still shown above - they never "
                "require ground truth."
            ),
        )

    cm = metrics["confusion_matrix"]
    return AnomalyEvaluationMetrics(
        measured=True,
        total_events=anomaly_result.total_events,
        predicted_anomalous=anomaly_result.anomalous_events,
        predicted_normal=anomaly_result.normal_events,
        ground_truth_anomalous=int(metrics["support_suspicious"]),
        ground_truth_normal=int(metrics["support_normal"]),
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1_score=float(metrics["f1_score"]),
        confusion_matrix=[[int(v) for v in row] for row in cm],
        confusion_matrix_labels=list(metrics["confusion_matrix_labels"]),
        note=(
            "Offline Evaluation: measured against true_label for this "
            "synthetic dataset only. Not a claim of production-level "
            "accuracy on real-world traffic."
        ),
    )


# =======================================================================
# C. Incident / Classification Analysis
# =======================================================================
def evaluate_classification(rule_result: RuleEngineResult) -> ClassificationEvaluationMetrics:
    """
    Purely descriptive aggregation over `RuleEngineResult.results_df`
    (already computed by Phase 3's `run_rule_engine()`) - never re-runs
    the rule engine and never reads `true_label`.
    """
    df = rule_result.results_df
    if df is None or df.empty:
        return ClassificationEvaluationMetrics(total_events=0)

    incident_type_counts = {str(k): int(v) for k, v in df["incident_type"].value_counts().items()}
    severity_counts = {str(k): int(v) for k, v in df["severity"].value_counts().items()}

    rule_counts: Dict[str, int] = {}
    multi_rule_event_count = 0
    no_rule_anomaly_count = 0
    for incident_type, rule_ids in zip(df["incident_type"], df["triggered_rule_ids"]):
        ids = rule_ids or []
        if len(ids) >= 2:
            multi_rule_event_count += 1
        if incident_type != "Normal Activity" and len(ids) == 0:
            no_rule_anomaly_count += 1
        for rule_id in ids:
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

    return ClassificationEvaluationMetrics(
        total_events=rule_result.total_events,
        incident_type_counts=incident_type_counts,
        severity_counts=severity_counts,
        rule_firing_counts=dict(sorted(rule_counts.items())),
        multi_rule_event_count=multi_rule_event_count,
        no_rule_anomaly_count=no_rule_anomaly_count,
    )


# =======================================================================
# D. Confidence & Human Review
# =======================================================================
def evaluate_confidence(confidence_result: ConfidenceEngineResult) -> ConfidenceEvaluationMetrics:
    """Direct pass-through of counts Phase 4's `run_confidence_scoring()`
    already computed - no recomputation, no true_label."""
    return ConfidenceEvaluationMetrics(
        total_events=confidence_result.total_events,
        high_count=confidence_result.high_confidence_count,
        medium_count=confidence_result.medium_confidence_count,
        low_count=confidence_result.low_confidence_count,
        human_review_count=confidence_result.human_review_count,
    )


# =======================================================================
# E. RAG Performance
# =======================================================================
def evaluate_rag(
    confidence_result: ConfidenceEngineResult, knowledge_base_dir: Optional[str] = None
) -> RagEvaluationMetrics:
    """
    Re-runs `retrieve_knowledge_for_incident()` once per anomalous event -
    the exact same function every other page already calls one incident
    at a time - purely to aggregate descriptive retrieval statistics. This
    does not change RAG settings, the knowledge base, or any stored
    result.

    "Controlled success" is a legitimately-available check, not an
    invented one: every knowledge-base entry is hand-tagged in Phase 8
    with an `incident_types` list (see `knowledge_base/*.json`). A
    retrieval "succeeds" here if the incident's own classified
    incident_type (from Phase 3, never true_label) appears in the top
    retrieved entry's own `incident_types` tag list.
    """
    kb = load_knowledge_base(knowledge_base_dir)
    kb_by_id: Dict[str, KnowledgeEntry] = {entry.knowledge_id: entry for entry in kb}
    df = confidence_result.results_df

    if df is None or df.empty:
        return RagEvaluationMetrics(
            knowledge_base_size=len(kb),
            note="No events available to run retrieval against.",
        )
    if not kb:
        return RagEvaluationMetrics(
            knowledge_base_size=0,
            note="The knowledge base is empty, so retrieval cannot be evaluated.",
        )

    anomalous_rows = df[df["incident_type"] != "Normal Activity"]
    if anomalous_rows.empty:
        return RagEvaluationMetrics(
            knowledge_base_size=len(kb),
            note="No anomalous events in this dataset to retrieve knowledge for.",
        )

    attempted = 0
    hits = 0
    empty = 0
    top_match_counts: Dict[str, int] = {}
    controlled_eligible = 0
    controlled_success = 0

    for _, row in anomalous_rows.iterrows():
        inf = row["_inference"]
        conf = row["_confidence"]
        result = retrieve_knowledge_for_incident(inf, conf, knowledge_base_dir=knowledge_base_dir)
        attempted += 1
        if not result.items:
            empty += 1
            continue

        hits += 1
        top_item = result.items[0]
        top_match_counts[inf.incident_type] = top_match_counts.get(inf.incident_type, 0) + 1

        top_entry = kb_by_id.get(top_item.knowledge_id)
        if top_entry is not None:
            controlled_eligible += 1
            if inf.incident_type in top_entry.incident_types:
                controlled_success += 1

    controlled_rate = (controlled_success / controlled_eligible) if controlled_eligible else None

    return RagEvaluationMetrics(
        knowledge_base_size=len(kb),
        retrieval_attempted_count=attempted,
        retrieval_hit_count=hits,
        retrieval_empty_count=empty,
        top_match_counts_by_incident_type=top_match_counts,
        controlled_eligible_count=controlled_eligible,
        controlled_success_count=controlled_success,
        controlled_success_rate=controlled_rate,
        note=(
            "Retrieval re-run once per anomalous event via the same "
            "retrieve_knowledge_for_incident() every other page already "
            "calls, purely for descriptive evaluation - it changes "
            "nothing stored. \"Controlled success\" uses each knowledge "
            "entry's own hand-authored incident_types tag as the only "
            "available notion of a \"correct\" top match."
        ),
    )


# =======================================================================
# F. Generative AI Performance
# =======================================================================
def evaluate_genai(model_name: Optional[str] = None) -> GenAIEvaluationMetrics:
    """
    Model/availability come from `get_ollama_status()` (already used on
    the Generative AI Analysis page). Attempted/successful/fallback
    counts and the source breakdown are derived read-only from this
    session's own audit trail (`GENAI_ANALYSIS` events already written by
    `app.py`'s `_log_case_processing_audit()`) - they reflect GenAI
    analyses actually performed and recorded so far, never a fabricated
    or estimated figure, and never a re-run of GenAI generation itself
    (which would be slow/non-deterministic and is explicitly out of
    scope for this descriptive count).
    """
    resolved_model_name = model_name or OLLAMA_MODEL_NAME
    status = get_ollama_status(resolved_model_name)

    events = load_audit_events()
    if events is None or events.empty or "stage" not in events.columns:
        genai_events = pd.DataFrame()
    else:
        genai_events = events[events["stage"] == "GENAI_ANALYSIS"]

    attempted = int(len(genai_events))
    successful = 0
    fallback = 0
    source_breakdown: Dict[str, int] = {}

    if attempted:
        for details in genai_events.get("descriptive_details", pd.Series(dtype=str)).fillna(""):
            match = _GENAI_DETAILS_RE.search(str(details))
            if not match:
                continue
            source = match.group(1).strip()
            used_fallback = match.group(2) == "True"
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
            if used_fallback:
                fallback += 1
            else:
                successful += 1

    note = (
        "Measured from this project's audit trail (data/audit_trail.db): "
        "counts reflect GenAI analyses actually performed and recorded so "
        "far in Generative AI Analysis / Response Planning / Analyst "
        "Review / Explainability, not a re-run over the whole dataset."
        if attempted
        else (
            "No GENAI_ANALYSIS audit events recorded yet. Visit Generative "
            "AI Analysis for at least one incident to populate this "
            "section - generation counts are never estimated."
        )
    )

    return GenAIEvaluationMetrics(
        model_name=resolved_model_name,
        ollama_available=status.ollama_available,
        model_available=status.model_available,
        ollama_status_message=status.message,
        generations_attempted=attempted,
        successful_generations=successful,
        fallback_generations=fallback,
        source_breakdown=source_breakdown,
        measured=attempted > 0,
        note=note,
    )


# =======================================================================
# G. Analyst Feedback
# =======================================================================
def evaluate_feedback() -> FeedbackEvaluationMetrics:
    """Direct pass-through of Phase 7's own `summarize_feedback()` -
    never recomputed, never touches true_label."""
    summary = summarize_feedback()
    return FeedbackEvaluationMetrics(
        total_reviews=summary.total_reviews,
        approved_count=summary.approved_count,
        rejected_count=summary.rejected_count,
        modified_count=summary.modified_count,
        approval_rate=summary.approval_rate,
        rejection_rate=summary.rejection_rate,
        modification_rate=summary.modification_rate,
    )


# =======================================================================
# H. Audit Activity
# =======================================================================
def evaluate_audit() -> AuditEvaluationMetrics:
    """Direct pass-through/aggregation of Phase 10's own audit trail -
    never writes an event, never touches true_label."""
    events = load_audit_events()
    if events is None or events.empty or "stage" not in events.columns:
        return AuditEvaluationMetrics(total_events=0, events_per_stage={stage: 0 for stage in AUDIT_STAGES})

    counts = events["stage"].value_counts().to_dict()
    events_per_stage = {stage: int(counts.get(stage, 0)) for stage in AUDIT_STAGES}
    return AuditEvaluationMetrics(total_events=int(len(events)), events_per_stage=events_per_stage)


# =======================================================================
# Orchestration - the one function the Streamlit page calls
# =======================================================================
def build_evaluation_report(
    anomaly_result: AnomalyDetectionResult,
    rule_result: RuleEngineResult,
    confidence_result: ConfidenceEngineResult,
    knowledge_base_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> EvaluationReport:
    return EvaluationReport(
        dataset_overview=evaluate_dataset_overview(anomaly_result),
        anomaly_metrics=evaluate_anomaly_detection(anomaly_result),
        classification_metrics=evaluate_classification(rule_result),
        confidence_metrics=evaluate_confidence(confidence_result),
        rag_metrics=evaluate_rag(confidence_result, knowledge_base_dir),
        genai_metrics=evaluate_genai(model_name),
        feedback_metrics=evaluate_feedback(),
        audit_metrics=evaluate_audit(),
    )


# =======================================================================
# Performance timing (local-machine samples only - see module docstring)
# =======================================================================
def measure_anomaly_detection_timing(preprocessing_result) -> TimingMeasurement:
    """
    Times one fresh train+predict cycle on the full ml_df, using the same
    `train_isolation_forest()` / `predict_anomalies()` functions the AI
    Anomaly Detection page already calls (same config, same fixed
    random_state, so the result is identical - only the wall-clock time
    is new information). Cheap (a few hundred rows) and deterministic, so
    it is safe to run automatically.
    """
    ml_df = getattr(preprocessing_result, "ml_df", None)
    if ml_df is None or ml_df.empty:
        return TimingMeasurement(
            stage="ANOMALY_DETECTION", measured=False, note="No feature matrix available to time."
        )
    try:
        start = time.perf_counter()
        model = train_isolation_forest(ml_df)
        predict_anomalies(model, ml_df)
        elapsed = time.perf_counter() - start
    except Exception:  # noqa: BLE001 - timing must never crash the app
        return TimingMeasurement(
            stage="ANOMALY_DETECTION", measured=False, note="Timing could not be measured safely."
        )
    return TimingMeasurement(
        stage="ANOMALY_DETECTION",
        measured=True,
        elapsed_seconds=round(elapsed, 4),
        sample_size=len(ml_df),
        note="Local machine timing for training + predicting once on the full dataset. Not a production benchmark.",
    )


def measure_rag_retrieval_timing(
    confidence_result: ConfidenceEngineResult, knowledge_base_dir: Optional[str] = None
) -> TimingMeasurement:
    """Times retrieval (TF-IDF + cosine similarity) once per anomalous
    event - cheap and deterministic, safe to run automatically."""
    df = confidence_result.results_df
    if df is None or df.empty:
        return TimingMeasurement(stage="RAG_RETRIEVAL", measured=False, note="No events available to time.")

    anomalous_rows = df[df["incident_type"] != "Normal Activity"]
    if anomalous_rows.empty:
        return TimingMeasurement(
            stage="RAG_RETRIEVAL", measured=False, note="No anomalous events available to time retrieval against."
        )
    try:
        start = time.perf_counter()
        for _, row in anomalous_rows.iterrows():
            retrieve_knowledge_for_incident(row["_inference"], row["_confidence"], knowledge_base_dir=knowledge_base_dir)
        elapsed = time.perf_counter() - start
    except Exception:  # noqa: BLE001 - timing must never crash the app
        return TimingMeasurement(stage="RAG_RETRIEVAL", measured=False, note="Timing could not be measured safely.")

    n = len(anomalous_rows)
    return TimingMeasurement(
        stage="RAG_RETRIEVAL",
        measured=True,
        elapsed_seconds=round(elapsed, 4),
        sample_size=n,
        note=f"Local machine timing for {n} retrieval call(s). Not a production benchmark.",
    )


def measure_genai_generation_timing(context: IncidentContext) -> TimingMeasurement:
    """
    Times exactly ONE `generate_ai_case_summary()` call. Deliberately
    OPT-IN ONLY - the Evaluation page calls this from an explicit button,
    never automatically on page load, because it may perform a real
    network call to a local Ollama server and can take anywhere from
    under a second (deterministic fallback) to tens of seconds (a real
    model generation) depending on the local machine and whether Ollama
    is running. It never changes existing GenAI behavior - it only wraps
    the existing call with a timer.
    """
    try:
        start = time.perf_counter()
        result: GenAIResult = generate_ai_case_summary(context)
        elapsed = time.perf_counter() - start
    except Exception:  # noqa: BLE001 - timing must never crash the app
        return TimingMeasurement(stage="GENAI_ANALYSIS", measured=False, note="Timing could not be measured safely.")

    return TimingMeasurement(
        stage="GENAI_ANALYSIS",
        measured=True,
        elapsed_seconds=round(elapsed, 4),
        sample_size=1,
        note=(
            f"Local machine timing for ONE generation (source={result.source}, "
            f"used_fallback={result.used_fallback}). Not a production benchmark - "
            "timing varies by machine, model, and whether Ollama is running."
        ),
    )
