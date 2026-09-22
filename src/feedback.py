"""
feedback.py
------------
Phase 7 - Human analyst review & feedback loop.

    System Analysis
            |
      AI Recommendation
            |
      Defensive Response Plan
            |
      HUMAN ANALYST REVIEW
            |
      Approve / Reject / Modify
            |
      Persistent Feedback (this module)
            |
      Feedback Insights
            |
      Future Improvement

This module is the ONLY place that talks to the feedback database. It
persists the analyst's final decision on an incident's response plan -
never anything the system decides automatically - using Python's
built-in `sqlite3` module (no external database dependency). Feedback
survives an application restart because it lives in a real file on disk
(`config.FEEDBACK_DB_PATH`), not in `st.session_state`.

IMPORTANT - this module NEVER executes a cybersecurity action. It only
reads and writes small, structured feedback records. Nothing here calls
out to a network, a firewall, an account system, or any real system.

IMPORTANT - true_label:
    Nothing in this module reads `config.LABEL_COLUMN`. A `FeedbackRecord`
    has no field for it at all - it is structurally impossible to store,
    exactly like every earlier phase's structured result.

IMPORTANT - authoritative values stay authoritative:
    `incident_type`, `severity`, and `confidence_score`/`confidence_level`
    are written into a `FeedbackRecord` from the SAME already-computed
    Phase 3 `InferenceRecord` / Phase 4 `ConfidenceRecord` objects every
    other phase reads - this module never recomputes or accepts an
    analyst-edited value for any of them. Only the response PLAN's STEPS
    can be modified by an analyst (Requirement #4); everything else in a
    `FeedbackRecord` is copied verbatim from the existing system analysis.

IMPORTANT - validation is reused, not duplicated:
    A modified plan is validated with `src.response_planner.validate_plan()`
    - the exact same function Phase 6 already uses - never a second,
    parallel implementation of plan-validity rules.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from config import (
    ANALYST_DECISIONS,
    DECISION_TO_REVIEW_STATUS,
    FEEDBACK_DB_PATH,
    FEEDBACK_NOTE_STOPWORDS,
    REVIEW_STATUS_PENDING,
)
from src.response_planner import PlanStep, PlanValidationResult, ResponsePlanResult, validate_plan

TABLE_NAME = "feedback"

# Columns stored exactly as named here - kept as a single source of truth
# so the CREATE TABLE statement, INSERT statement, and row-parsing code
# can never silently drift apart.
_COLUMNS = [
    "feedback_id",
    "incident_id",
    "timestamp",
    "incident_type",
    "severity",
    "confidence_level",
    "confidence_score",
    "human_review_recommended",
    "analyst_decision",
    "review_status",
    "original_plan",
    "modified_plan",
    "analyst_notes",
    "triggered_rule_ids",
    "supporting_indicators",
    "ai_model_name",
    "ai_summary_source",
]


# =======================================================================
# Result containers
# =======================================================================
@dataclass
class FeedbackRecord:
    """One analyst decision on one incident's response plan."""

    feedback_id: str
    incident_id: str
    timestamp: str  # ISO 8601, UTC
    incident_type: str
    severity: str
    confidence_level: str
    confidence_score: float
    human_review_recommended: bool
    analyst_decision: str  # one of config.ANALYST_DECISIONS
    review_status: str  # one of config.REVIEW_STATUSES (never PENDING once stored)
    original_plan: dict  # the untouched, system-generated plan (see plan_to_dict)
    analyst_notes: str = ""
    modified_plan: Optional[dict] = None  # only set when analyst_decision == "MODIFY"
    triggered_rule_ids: List[str] = field(default_factory=list)
    supporting_indicators: List[str] = field(default_factory=list)
    ai_model_name: Optional[str] = None
    ai_summary_source: Optional[str] = None


@dataclass
class FeedbackValidationResult:
    """Requirement #5/#7 - a feedback record's own self-check, never raised."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class FeedbackSummary:
    """Requirement #12/#13 - descriptive insights only, never automatic learning."""

    total_reviews: int
    approved_count: int
    rejected_count: int
    modified_count: int
    approval_rate: float
    rejection_rate: float
    modification_rate: float
    incident_type_counts: Dict[str, int] = field(default_factory=dict)
    note_keyword_counts: Dict[str, int] = field(default_factory=dict)


# =======================================================================
# Stable, human-readable incident IDs (Requirement #14)
# =======================================================================
def build_incident_id(event_index: int) -> str:
    """
    Deterministically derive a stable, human-readable incident ID from an
    event's DataFrame row index, e.g. `build_incident_id(0) == "INC-0000"`.

    Why this is stable, not just "stable during a session":
      - `data/synthetic_security_logs.csv` is never regenerated or
        reordered after Phase 0 (checksum-verified after every phase), so
        row 0 is always the same event.
      - Every upstream stage (preprocessing, Isolation Forest with a fixed
        `RANDOM_STATE`, the rule engine, confidence scoring) is fully
        deterministic, so the SAME row index always produces the SAME
        `InferenceRecord`/`ConfidenceRecord` on every run.
    Together, `INC-{index:04d}` refers to the same real incident today,
    tomorrow, or after a full application restart - not just within one
    running session.
    """
    return f"INC-{int(event_index):04d}"


# =======================================================================
# Plan <-> plain-dict (de)serialization, for JSON storage in SQLite
# =======================================================================
def plan_to_dict(plan: ResponsePlanResult) -> dict:
    """A `ResponsePlanResult` (and its nested `PlanStep`/`PlanValidationResult`
    objects) as a plain, JSON-serializable dict - used for both storage and
    on-screen display of a saved plan snapshot."""
    return dataclasses.asdict(plan)


def dict_to_plan(data: dict) -> ResponsePlanResult:
    """
    The inverse of `plan_to_dict()` - rebuilds real `PlanStep` /
    `PlanValidationResult` / `ResponsePlanResult` objects from a plain
    dict (e.g. one just loaded back from SQLite, or built by the Streamlit
    "modify plan" UI). Needed so a modified plan can be re-validated with
    the exact same `src.response_planner.validate_plan()` Phase 6 uses.
    """
    data = dict(data)  # never mutate the caller's dict
    steps = [PlanStep(**step) for step in data.get("steps", [])]
    validation_raw = data.get("validation_status") or {"is_valid": True, "errors": [], "warnings": []}
    validation_status = PlanValidationResult(
        is_valid=validation_raw.get("is_valid", True),
        errors=list(validation_raw.get("errors", [])),
        warnings=list(validation_raw.get("warnings", [])),
    )
    return ResponsePlanResult(
        incident_type=data.get("incident_type", ""),
        severity=data.get("severity", ""),
        confidence_level=data.get("confidence_level", ""),
        human_review_recommended=bool(data.get("human_review_recommended", False)),
        plan_title=data.get("plan_title", ""),
        plan_summary=data.get("plan_summary", ""),
        steps=steps,
        priority=data.get("priority", "ROUTINE"),
        review_required=bool(data.get("review_required", False)),
        planning_reason=data.get("planning_reason", ""),
        validation_status=validation_status,
        included_stages=list(data.get("included_stages", [])),
        supplementary_ai_context=list(data.get("supplementary_ai_context", [])),
    )


def generate_feedback_id(incident_id: str) -> str:
    """A unique, collision-resistant feedback ID that still shows which
    incident it belongs to at a glance, e.g. `FB-INC-0011-3f9a2c1b`."""
    return f"FB-{incident_id}-{uuid.uuid4().hex[:8]}"


# =======================================================================
# Database connection helpers - every public function below goes through
# these, so the table is guaranteed to exist no matter which function is
# called first (Requirement #7: "the application should not fail if the
# database does not exist yet").
# =======================================================================
def _resolve_db_path(db_path: Optional[str]) -> str:
    return db_path if db_path else FEEDBACK_DB_PATH


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = _resolve_db_path(db_path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ensure_table(conn)
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            feedback_id             TEXT PRIMARY KEY,
            incident_id             TEXT NOT NULL,
            timestamp               TEXT NOT NULL,
            incident_type           TEXT,
            severity                TEXT,
            confidence_level        TEXT,
            confidence_score        REAL,
            human_review_recommended INTEGER,
            analyst_decision        TEXT NOT NULL,
            review_status           TEXT NOT NULL,
            original_plan           TEXT,
            modified_plan           TEXT,
            analyst_notes           TEXT,
            triggered_rule_ids      TEXT,
            supporting_indicators   TEXT,
            ai_model_name           TEXT,
            ai_summary_source       TEXT
        )
        """
    )
    conn.commit()


def initialize_feedback_database(db_path: Optional[str] = None) -> bool:
    """
    Create the feedback database file and its table if they don't exist
    yet. Safe to call any number of times (idempotent) and safe to call
    before anything else has touched the database. Never raises - returns
    False (instead of crashing the Streamlit app) if the database file's
    location genuinely cannot be created or opened (Requirement #7:
    "database connection failures" handled gracefully).
    """
    try:
        conn = _connect(db_path)
        conn.close()
        return True
    except sqlite3.Error:
        return False
    except OSError:
        # e.g. the configured path's directory can't be created/written to
        return False


# =======================================================================
# Validation (Requirement #5) - reuses src.response_planner.validate_plan,
# never a second implementation of plan-validity rules.
# =======================================================================
def validate_feedback(record: FeedbackRecord) -> FeedbackValidationResult:
    """
    Self-check a feedback record before it is saved. NEVER raises - any
    problem is reported as an error in the returned
    `FeedbackValidationResult`, exactly like Phase 6's `validate_plan()`.
    """
    errors: List[str] = []

    if record.analyst_decision not in ANALYST_DECISIONS:
        errors.append(
            f"'{record.analyst_decision}' is not a valid analyst decision "
            f"(must be one of {ANALYST_DECISIONS})."
        )

    if not record.incident_id or not str(record.incident_id).strip():
        errors.append("incident_id is required.")

    if not record.original_plan:
        errors.append("original_plan is required (the system-generated plan being reviewed).")

    if record.analyst_decision == "MODIFY":
        if not record.modified_plan:
            errors.append("A MODIFY decision requires a modified_plan.")
        else:
            try:
                modified = dict_to_plan(record.modified_plan)
            except Exception as exc:  # noqa: BLE001 - never let a bad dict crash validation
                errors.append(f"modified_plan could not be interpreted as a valid plan: {exc}")
            else:
                plan_check = validate_plan(modified)
                if not plan_check.is_valid:
                    errors.extend(f"Modified plan: {e}" for e in plan_check.errors)
    else:
        # APPROVE/REJECT should never carry a modified_plan.
        if record.modified_plan:
            errors.append("Only a MODIFY decision may include a modified_plan.")
        # Defense in depth: even the untouched original plan must still be
        # a valid, safe plan before a decision on it can be saved.
        if record.analyst_decision == "APPROVE":
            try:
                original = dict_to_plan(record.original_plan)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"original_plan could not be interpreted as a valid plan: {exc}")
            else:
                plan_check = validate_plan(original)
                if not plan_check.is_valid:
                    errors.extend(f"Original plan: {e}" for e in plan_check.errors)

    return FeedbackValidationResult(is_valid=(len(errors) == 0), errors=errors)


# =======================================================================
# Save (Requirement #1/#6/#7)
# =======================================================================
def save_feedback(record: FeedbackRecord, db_path: Optional[str] = None) -> "tuple[bool, List[str]]":
    """
    Validate and persist one analyst decision. NEVER raises - always
    returns `(success, errors)` so the Streamlit page can show a clear
    message instead of crashing (Requirement #7).

    If validation fails, NOTHING is written (Requirement #5: "do not save
    the final decision" when validation fails).
    """
    check = validate_feedback(record)
    if not check.is_valid:
        return False, check.errors

    try:
        conn = _connect(db_path)
    except (sqlite3.Error, OSError) as exc:
        return False, [f"Could not open the feedback database: {exc}"]

    try:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (
                feedback_id, incident_id, timestamp, incident_type, severity,
                confidence_level, confidence_score, human_review_recommended,
                analyst_decision, review_status, original_plan, modified_plan,
                analyst_notes, triggered_rule_ids, supporting_indicators,
                ai_model_name, ai_summary_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.feedback_id,
                record.incident_id,
                record.timestamp,
                record.incident_type,
                record.severity,
                record.confidence_level,
                float(record.confidence_score) if record.confidence_score is not None else None,
                1 if record.human_review_recommended else 0,
                record.analyst_decision,
                record.review_status,
                json.dumps(record.original_plan),
                json.dumps(record.modified_plan) if record.modified_plan else None,
                record.analyst_notes or "",
                json.dumps(list(record.triggered_rule_ids or [])),
                json.dumps(list(record.supporting_indicators or [])),
                record.ai_model_name,
                record.ai_summary_source,
            ),
        )
        conn.commit()
        return True, []
    except sqlite3.IntegrityError:
        # Duplicate feedback_id (primary key collision) - handled
        # gracefully, per Requirement #7, rather than crashing or
        # silently overwriting a previous analyst's decision.
        return False, [f"A feedback record with ID '{record.feedback_id}' already exists."]
    except sqlite3.Error as exc:
        return False, [f"Could not save feedback: {exc}"]
    finally:
        conn.close()


# =======================================================================
# Load (Requirement #1/#7)
# =======================================================================
def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["human_review_recommended"] = bool(d.get("human_review_recommended"))
    for json_field in ("original_plan", "modified_plan", "triggered_rule_ids", "supporting_indicators"):
        raw = d.get(json_field)
        if raw:
            try:
                d[json_field] = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                d[json_field] = None  # a malformed stored record never crashes a read
        else:
            d[json_field] = None if json_field in ("original_plan", "modified_plan") else []
    return d


def load_feedback(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load every stored feedback record, newest first. NEVER raises - an
    empty or missing database returns an empty (but correctly-columned)
    DataFrame, and a database error is treated the same way, so the
    Streamlit app can always render safely (Requirement #7/"empty
    feedback database is handled safely").
    """
    try:
        conn = _connect(db_path)
    except (sqlite3.Error, OSError):
        return pd.DataFrame(columns=_COLUMNS)

    try:
        rows = conn.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY timestamp DESC").fetchall()
    except sqlite3.Error:
        return pd.DataFrame(columns=_COLUMNS)
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame([_row_to_dict(r) for r in rows])


def get_feedback_by_incident(incident_id: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """All feedback records for one incident, newest first. An incident
    with no feedback yet safely returns an empty DataFrame."""
    all_feedback = load_feedback(db_path)
    if all_feedback.empty:
        return all_feedback
    return all_feedback[all_feedback["incident_id"] == incident_id].reset_index(drop=True)


def get_review_status_map(feedback_df: pd.DataFrame) -> Dict[str, str]:
    """
    For each incident_id present in `feedback_df`, the review_status of
    its MOST RECENT feedback record. Incidents with no feedback at all
    simply won't appear here - callers treat "not in this map" as
    `config.REVIEW_STATUS_PENDING` (Requirement #8). This is a pure
    function with no database access, so it works equally well on a
    freshly-loaded DataFrame or on one already held in memory.
    """
    if feedback_df is None or feedback_df.empty:
        return {}
    # load_feedback() already orders newest-first, but sort defensively
    # here too in case a caller passes in a differently-ordered frame.
    ordered = feedback_df.sort_values("timestamp", ascending=False)
    status_map: Dict[str, str] = {}
    for _, row in ordered.iterrows():
        incident_id = row["incident_id"]
        if incident_id not in status_map:  # first time seen = most recent (already sorted)
            status_map[incident_id] = row["review_status"]
    return status_map


# =======================================================================
# Feedback insights (Requirement #12/#13) - descriptive summaries only,
# never automatic retraining or rule/model changes.
# =======================================================================
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _keyword_counts(notes: List[str], top_n: int = 15) -> Dict[str, int]:
    counter: Counter = Counter()
    for note in notes:
        if not note:
            continue
        for word in _WORD_RE.findall(note.lower()):
            if len(word) < 4 or word in FEEDBACK_NOTE_STOPWORDS:
                continue
            counter[word] += 1
    return dict(counter.most_common(top_n))


def summarize_feedback(feedback_df: Optional[pd.DataFrame] = None, db_path: Optional[str] = None) -> FeedbackSummary:
    """
    Descriptive statistics computed purely from stored feedback - approval/
    rejection/modification rates, the most common incident types reviewed,
    and simple keyword counts from analyst notes. This is explicitly a
    reporting/insight function, not a training or rule-update mechanism
    (Requirement #12 - "do NOT implement automatic retraining"). Never
    uses `true_label`, which isn't even stored in a feedback record.
    """
    if feedback_df is None:
        feedback_df = load_feedback(db_path)

    total = len(feedback_df)
    if total == 0:
        return FeedbackSummary(
            total_reviews=0, approved_count=0, rejected_count=0, modified_count=0,
            approval_rate=0.0, rejection_rate=0.0, modification_rate=0.0,
        )

    approved = int((feedback_df["analyst_decision"] == "APPROVE").sum())
    rejected = int((feedback_df["analyst_decision"] == "REJECT").sum())
    modified = int((feedback_df["analyst_decision"] == "MODIFY").sum())

    incident_type_counts = feedback_df["incident_type"].value_counts().to_dict()
    keyword_counts = _keyword_counts(feedback_df["analyst_notes"].fillna("").tolist())

    return FeedbackSummary(
        total_reviews=total,
        approved_count=approved,
        rejected_count=rejected,
        modified_count=modified,
        approval_rate=round(approved / total, 4),
        rejection_rate=round(rejected / total, 4),
        modification_rate=round(modified / total, 4),
        incident_type_counts=incident_type_counts,
        note_keyword_counts=keyword_counts,
    )
