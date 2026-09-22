"""
continuous_improvement.py
--------------------------
Phase 9 - Continuous Improvement from Analyst Feedback.

    Analyst Feedback (Phase 7, SQLite)
            |
      Pattern Detection              (this module)
            |
      Improvement Suggestion         (this module)
            |
      Human / Developer Review
            |
      Manual Future Change

This module turns the analyst feedback already persisted by Phase 7
(`src/feedback.py`) into deterministic, descriptive, TRACEABLE
improvement insights for a human developer/analyst to review. It is an
OFFLINE, HUMAN-SUPERVISED reporting step - not a learning system.

IMPORTANT - this is analysis, never automatic learning:
    Every function here READS already-stored feedback and returns a
    report describing patterns it found. Nothing in this module writes
    back to `config.py`, `src/anomaly_detector.py`, `src/rule_engine.py`,
    `src/confidence.py`, `src/retriever.py`, or `src/response_planner.py`
    - there is not even an import path here that could do that. This
    module does not retrain any model, does not tune any threshold, does
    not edit any rule, does not change RAG configuration or the
    knowledge base, does not change response-planning logic, and never
    executes a cybersecurity action of any kind. A human developer reads
    the suggestions produced here and decides, by hand, whether and how
    to act on them - that decision happens outside this codebase.

IMPORTANT - read-only with respect to the feedback database:
    This module only ever calls `src.feedback.load_feedback()` (a plain
    SELECT). It never calls `save_feedback()` or any other write path,
    and it never issues its own SQL. The database implementation itself
    is NOT duplicated here - Phase 7's `src/feedback.py` remains the only
    module that talks to SQLite.

IMPORTANT - true_label:
    Never read here. A stored `FeedbackRecord` has no `true_label` field
    to begin with (see `src/feedback.py`), and this module never touches
    the raw dataset or `config.LABEL_COLUMN` either - it only reads the
    feedback DataFrame that `load_feedback()` returns.

IMPORTANT - traceability, never fabrication:
    Every `ImprovementSuggestion` below carries the exact `incident_id`s
    and `feedback_id`s it was derived from. A pattern that does not
    reach `config.CI_MIN_PATTERN_OCCURRENCES` distinct supporting
    records simply produces no suggestion - nothing here invents an
    example or a plausible-sounding pattern that isn't backed by actual
    stored feedback.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import (
    CI_MIN_PATTERN_OCCURRENCES,
    CI_TOP_N_MODIFIED_STEPS,
    CI_TOP_N_NOTE_THEMES,
    CI_TOP_N_SUGGESTIONS,
    FEEDBACK_NOTE_STOPWORDS,
)
from src.feedback import _WORD_RE, FeedbackSummary, load_feedback, summarize_feedback

# =======================================================================
# Result containers
# =======================================================================
@dataclass
class IncidentTypeBreakdown:
    """One incident_type's review outcomes (Requirement #2/#3)."""

    incident_type: str
    total_reviews: int
    approve_count: int
    reject_count: int
    modify_count: int
    disagreement_count: int  # reject_count + modify_count
    disagreement_rate: float  # disagreement_count / total_reviews
    supporting_incident_ids: List[str] = field(default_factory=list)
    supporting_feedback_ids: List[str] = field(default_factory=list)


@dataclass
class PlanStepModification:
    """One REJECT/MODIFY-adjacent difference between an original and
    analyst-modified plan step, for one feedback record (internal - used
    to build `FrequentStepModification` below)."""

    incident_type: str
    change_type: str  # "wording_changed" | "step_removed" | "step_added"
    original_action: Optional[str]
    modified_action: Optional[str]
    incident_id: str
    feedback_id: str


@dataclass
class FrequentStepModification:
    """A response-plan step that analysts have repeatedly changed
    (Requirement #5/#6) - always backed by >= CI_MIN_PATTERN_OCCURRENCES
    distinct feedback records."""

    incident_type: str
    original_action: str
    change_type: str
    occurrence_count: int
    example_modified_actions: List[str] = field(default_factory=list)
    supporting_incident_ids: List[str] = field(default_factory=list)
    supporting_feedback_ids: List[str] = field(default_factory=list)


@dataclass
class NoteTheme:
    """A word that recurs across multiple DISTINCT analyst notes
    (Requirement #4) - counted once per note, not once per occurrence
    within a note, so a theme genuinely means "multiple analysts/reviews
    mentioned this," not "one note repeated a word."""

    keyword: str
    occurrence_count: int
    supporting_incident_ids: List[str] = field(default_factory=list)
    supporting_feedback_ids: List[str] = field(default_factory=list)


@dataclass
class ImprovementSuggestion:
    """One human-readable, fully traceable suggestion (Requirement #8 +
    the Traceability section) - never automatically acted on."""

    suggestion_id: str
    category: str  # "incident_type_disagreement" | "response_plan_wording" | "analyst_note_theme"
    title: str
    description: str
    supporting_incident_ids: List[str] = field(default_factory=list)
    supporting_feedback_ids: List[str] = field(default_factory=list)


@dataclass
class ContinuousImprovementReport:
    """Everything the Feedback History page's "Continuous Improvement
    Insights" section needs, in one deterministic, read-only snapshot."""

    total_feedback_records: int
    approved_count: int
    rejected_count: int
    modified_count: int
    approval_rate: float
    rejection_rate: float
    modification_rate: float
    most_reviewed_incident_types: List[Tuple[str, int]] = field(default_factory=list)
    incident_type_breakdowns: List[IncidentTypeBreakdown] = field(default_factory=list)
    frequent_step_modifications: List[FrequentStepModification] = field(default_factory=list)
    note_themes: List[NoteTheme] = field(default_factory=list)
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)


# =======================================================================
# 1-3. Decision counts/rates + per-incident-type breakdown
#      (Requirement #1/#2/#3) - counts/rates reuse Phase 7's own
#      summarize_feedback(), never recomputed independently here.
# =======================================================================
def _safe_str(value) -> str:
    # Covers both a real None (e.g. from row.get()/dict access) and pandas'
    # own NaN stand-in for None (e.g. a groupby() key from a column that
    # contains None - pandas represents that group's key as float('nan'),
    # not None, so `value is None` alone would miss it and fall through to
    # str(nan) == "nan").
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def compute_incident_type_breakdowns(feedback_df: pd.DataFrame) -> List[IncidentTypeBreakdown]:
    """
    One `IncidentTypeBreakdown` per incident_type present in the feedback,
    sorted by total review count (most-reviewed first, Requirement #2),
    each carrying which specific incidents were REJECTed or MODIFYed so a
    developer can see exactly where analysts disagreed with the system
    (Requirement #3/#7).
    """
    if feedback_df is None or feedback_df.empty:
        return []

    breakdowns: List[IncidentTypeBreakdown] = []
    for incident_type, group in feedback_df.groupby("incident_type", dropna=False):
        incident_type = _safe_str(incident_type) or "(unknown)"
        total = len(group)
        approve = int((group["analyst_decision"] == "APPROVE").sum())
        reject = int((group["analyst_decision"] == "REJECT").sum())
        modify = int((group["analyst_decision"] == "MODIFY").sum())
        disagreement = reject + modify
        disagreeing_rows = group[group["analyst_decision"].isin(["REJECT", "MODIFY"])]
        breakdowns.append(
            IncidentTypeBreakdown(
                incident_type=incident_type,
                total_reviews=total,
                approve_count=approve,
                reject_count=reject,
                modify_count=modify,
                disagreement_count=disagreement,
                disagreement_rate=round(disagreement / total, 4) if total else 0.0,
                supporting_incident_ids=sorted(set(disagreeing_rows["incident_id"].dropna().astype(str))),
                supporting_feedback_ids=sorted(set(disagreeing_rows["feedback_id"].dropna().astype(str))),
            )
        )

    breakdowns.sort(key=lambda b: (-b.total_reviews, b.incident_type))
    return breakdowns


# =======================================================================
# 5-6. Repeated response-plan modifications / frequently modified steps
#      (Requirement #5/#6) - reuses src.feedback's own plan dict shape,
#      never a second plan-diffing implementation elsewhere.
# =======================================================================
def _extract_plan_step_diffs(feedback_df: pd.DataFrame) -> List[PlanStepModification]:
    """
    For every MODIFY record with both an `original_plan` and a
    `modified_plan`, compare their steps position-by-position (aligned by
    `step_number`) and record what changed. Malformed/missing plan data
    is skipped for that one record rather than raising (Requirement:
    "malformed/missing feedback handling").
    """
    diffs: List[PlanStepModification] = []
    if feedback_df is None or feedback_df.empty:
        return diffs

    modify_rows = feedback_df[feedback_df["analyst_decision"] == "MODIFY"]
    for _, row in modify_rows.iterrows():
        original_plan = row.get("original_plan")
        modified_plan = row.get("modified_plan")
        if not isinstance(original_plan, dict) or not isinstance(modified_plan, dict):
            continue  # malformed/missing plan JSON for this record - skip safely

        incident_type = _safe_str(row.get("incident_type")) or "(unknown)"
        incident_id = _safe_str(row.get("incident_id"))
        feedback_id = _safe_str(row.get("feedback_id"))

        original_steps = {
            s.get("step_number"): s for s in (original_plan.get("steps") or []) if isinstance(s, dict)
        }
        modified_steps = {
            s.get("step_number"): s for s in (modified_plan.get("steps") or []) if isinstance(s, dict)
        }

        for step_number in sorted(set(original_steps) | set(modified_steps), key=lambda n: (n is None, n)):
            original_step = original_steps.get(step_number)
            modified_step = modified_steps.get(step_number)

            if original_step is not None and modified_step is not None:
                original_action = original_step.get("action")
                modified_action = modified_step.get("action")
                if original_action != modified_action:
                    diffs.append(
                        PlanStepModification(
                            incident_type=incident_type,
                            change_type="wording_changed",
                            original_action=original_action,
                            modified_action=modified_action,
                            incident_id=incident_id,
                            feedback_id=feedback_id,
                        )
                    )
            elif original_step is not None and modified_step is None:
                diffs.append(
                    PlanStepModification(
                        incident_type=incident_type,
                        change_type="step_removed",
                        original_action=original_step.get("action"),
                        modified_action=None,
                        incident_id=incident_id,
                        feedback_id=feedback_id,
                    )
                )
            elif original_step is None and modified_step is not None:
                diffs.append(
                    PlanStepModification(
                        incident_type=incident_type,
                        change_type="step_added",
                        original_action=None,
                        modified_action=modified_step.get("action"),
                        incident_id=incident_id,
                        feedback_id=feedback_id,
                    )
                )
    return diffs


def _aggregate_frequent_step_modifications(
    diffs: List[PlanStepModification],
    min_occurrences: int = CI_MIN_PATTERN_OCCURRENCES,
    top_n: int = CI_TOP_N_MODIFIED_STEPS,
) -> List[FrequentStepModification]:
    """
    Group step diffs by (incident_type, original_action) - the exact,
    already-stored step wording the system generated - so a repeated
    pattern means "analysts changed THIS SPECIFIC system-generated step
    more than once," never an invented category.
    """
    groups: Dict[Tuple[str, str], List[PlanStepModification]] = defaultdict(list)
    for d in diffs:
        if d.change_type == "step_added" or not d.original_action:
            continue  # nothing to group an added step under; needs an original action
        groups[(d.incident_type, d.original_action)].append(d)

    frequent: List[FrequentStepModification] = []
    for (incident_type, original_action), group_diffs in groups.items():
        distinct_records = {(d.incident_id, d.feedback_id) for d in group_diffs}
        if len(distinct_records) < min_occurrences:
            continue
        example_modified = sorted(
            {d.modified_action for d in group_diffs if d.change_type == "wording_changed" and d.modified_action}
        )[:3]
        dominant_change_type = "wording_changed" if example_modified else group_diffs[0].change_type
        frequent.append(
            FrequentStepModification(
                incident_type=incident_type,
                original_action=original_action,
                change_type=dominant_change_type,
                occurrence_count=len(distinct_records),
                example_modified_actions=example_modified,
                supporting_incident_ids=sorted({d.incident_id for d in group_diffs}),
                supporting_feedback_ids=sorted({d.feedback_id for d in group_diffs}),
            )
        )

    frequent.sort(key=lambda f: (-f.occurrence_count, f.incident_type, f.original_action))
    return frequent[:top_n]


# =======================================================================
# 4. Repeated analyst-note themes (Requirement #4) - reuses the exact
#    same tokenizer/stopword list Phase 7's summarize_feedback() uses
#    (`src.feedback._WORD_RE`, `config.FEEDBACK_NOTE_STOPWORDS`), so
#    "theme" here means the same thing as "keyword" there - this just
#    adds per-theme traceability that the Phase 7 summary doesn't need.
# =======================================================================
def _extract_note_themes(
    feedback_df: pd.DataFrame,
    min_occurrences: int = CI_MIN_PATTERN_OCCURRENCES,
    top_n: int = CI_TOP_N_NOTE_THEMES,
) -> List[NoteTheme]:
    if feedback_df is None or feedback_df.empty:
        return []

    keyword_records: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for _, row in feedback_df.iterrows():
        note = row.get("analyst_notes") or ""
        if not note:
            continue
        incident_id = _safe_str(row.get("incident_id"))
        feedback_id = _safe_str(row.get("feedback_id"))
        # A set: a word repeated twice in ONE note still counts as one
        # occurrence of that theme for THAT feedback record.
        words_in_note = {
            word for word in _WORD_RE.findall(note.lower())
            if len(word) >= 4 and word not in FEEDBACK_NOTE_STOPWORDS
        }
        for word in words_in_note:
            keyword_records[word].append((incident_id, feedback_id))

    themes: List[NoteTheme] = []
    for keyword, records in keyword_records.items():
        if len(records) < min_occurrences:
            continue
        themes.append(
            NoteTheme(
                keyword=keyword,
                occurrence_count=len(records),
                supporting_incident_ids=sorted({r[0] for r in records}),
                supporting_feedback_ids=sorted({r[1] for r in records}),
            )
        )

    themes.sort(key=lambda t: (-t.occurrence_count, t.keyword))
    return themes[:top_n]


# =======================================================================
# 7-8. Disagreement areas + suggested improvements (Requirement #7/#8)
# =======================================================================
def _generate_suggestions(
    breakdowns: List[IncidentTypeBreakdown],
    frequent_steps: List[FrequentStepModification],
    note_themes: List[NoteTheme],
    min_occurrences: int = CI_MIN_PATTERN_OCCURRENCES,
    top_n: int = CI_TOP_N_SUGGESTIONS,
) -> List[ImprovementSuggestion]:
    """
    Every suggestion below is derived strictly from the already-computed
    breakdowns/steps/themes above, each of which is itself derived
    strictly from stored feedback records - nothing here is invented.
    Suggestions are built in a fixed category order and sorted within
    each category, then given sequential IDs, so the SAME feedback data
    always produces the SAME suggestions in the SAME order (Requirement:
    "deterministic output").
    """
    candidates: List[ImprovementSuggestion] = []

    # Category 1: incident types where analysts repeatedly disagreed
    # (REJECT/MODIFY) with the system's generated plan.
    for b in sorted(breakdowns, key=lambda b: (-b.disagreement_count, b.incident_type)):
        if b.disagreement_count < min_occurrences:
            continue
        candidates.append(
            ImprovementSuggestion(
                suggestion_id="",  # assigned after final sort, below
                category="incident_type_disagreement",
                title=f"Review the system analysis/response plan for '{b.incident_type}'.",
                description=(
                    f"Analysts recorded REJECT or MODIFY on {b.disagreement_count} of "
                    f"{b.total_reviews} reviewed '{b.incident_type}' incident(s) "
                    f"({b.disagreement_rate:.0%} disagreement) - REJECT: {b.reject_count}, "
                    f"MODIFY: {b.modify_count}. This may indicate the generated response "
                    "plan, severity, or classification for this incident type is worth a "
                    "closer look."
                ),
                supporting_incident_ids=b.supporting_incident_ids,
                supporting_feedback_ids=b.supporting_feedback_ids,
            )
        )

    # Category 2: specific response-plan steps analysts keep rewriting.
    for f in frequent_steps:
        verb = "modified the wording of" if f.change_type == "wording_changed" else "removed"
        examples = (
            f" Analysts' rewordings included: {', '.join(f.example_modified_actions)}."
            if f.example_modified_actions
            else ""
        )
        candidates.append(
            ImprovementSuggestion(
                suggestion_id="",
                category="response_plan_wording",
                title=f"Review the wording of the '{f.original_action}' step for {f.incident_type}.",
                description=(
                    f"Analysts {verb} this system-generated step in {f.occurrence_count} "
                    f"separate '{f.incident_type}' reviews.{examples} Consider revisiting "
                    "this step's template wording in src/response_planner.py."
                ),
                supporting_incident_ids=f.supporting_incident_ids,
                supporting_feedback_ids=f.supporting_feedback_ids,
            )
        )

    # Category 3: recurring analyst-note themes.
    for t in note_themes:
        if t.occurrence_count < min_occurrences:
            continue
        candidates.append(
            ImprovementSuggestion(
                suggestion_id="",
                category="analyst_note_theme",
                title=f"Review analyst notes recurringly mentioning '{t.keyword}'.",
                description=(
                    f"The word '{t.keyword}' appeared in {t.occurrence_count} distinct "
                    "analyst notes. Recurring note language can indicate a shared concern "
                    "worth investigating (e.g. a common false-positive pattern or a "
                    "frequently-needed piece of evidence)."
                ),
                supporting_incident_ids=t.supporting_incident_ids,
                supporting_feedback_ids=t.supporting_feedback_ids,
            )
        )

    candidates = candidates[:top_n]
    for i, suggestion in enumerate(candidates, start=1):
        suggestion.suggestion_id = f"CI-{i:03d}"
    return candidates


# =======================================================================
# Top-level orchestration - the one function app.py (and tests) call
# =======================================================================
def generate_continuous_improvement_report(
    feedback_df: Optional[pd.DataFrame] = None,
    db_path: Optional[str] = None,
) -> ContinuousImprovementReport:
    """
    Build the full, deterministic Phase 9 report from stored feedback.
    NEVER raises - `load_feedback()` (Phase 7) already never raises, and
    every helper above defensively handles missing/malformed plan data,
    so an empty or partially-malformed database safely produces a mostly-
    empty report rather than an exception.
    """
    if feedback_df is None:
        feedback_df = load_feedback(db_path)

    # Reuse Phase 7's own counts/rates - never recomputed independently.
    summary: FeedbackSummary = summarize_feedback(feedback_df)

    if feedback_df is None or feedback_df.empty:
        return ContinuousImprovementReport(
            total_feedback_records=0,
            approved_count=0,
            rejected_count=0,
            modified_count=0,
            approval_rate=0.0,
            rejection_rate=0.0,
            modification_rate=0.0,
        )

    breakdowns = compute_incident_type_breakdowns(feedback_df)
    most_reviewed = [(b.incident_type, b.total_reviews) for b in breakdowns]

    diffs = _extract_plan_step_diffs(feedback_df)
    frequent_steps = _aggregate_frequent_step_modifications(diffs)

    note_themes = _extract_note_themes(feedback_df)

    suggestions = _generate_suggestions(breakdowns, frequent_steps, note_themes)

    return ContinuousImprovementReport(
        total_feedback_records=summary.total_reviews,
        approved_count=summary.approved_count,
        rejected_count=summary.rejected_count,
        modified_count=summary.modified_count,
        approval_rate=summary.approval_rate,
        rejection_rate=summary.rejection_rate,
        modification_rate=summary.modification_rate,
        most_reviewed_incident_types=most_reviewed,
        incident_type_breakdowns=breakdowns,
        frequent_step_modifications=frequent_steps,
        note_themes=note_themes,
        suggestions=suggestions,
    )
