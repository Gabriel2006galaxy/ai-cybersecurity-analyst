"""
audit_trail.py
----------------
Phase 10 - Persistent, append-oriented audit trail.

    Incident Processed
            |
      ANOMALY_DETECTION
            |
      RULE_CLASSIFICATION
            |
      CONFIDENCE
            |
      RAG_RETRIEVAL
            |
      GENAI_ANALYSIS
            |
      RESPONSE_PLAN
            |
      ANALYST_REVIEW
            |
      Persistent Audit Trail (this module)

This module is the ONLY place that talks to the audit-trail database. It
records, for one incident at a time, WHICH pipeline stage ran and a small,
structured summary of what that stage already produced - it never decides
anything and never recomputes anything itself. Every value written here
was already computed by an earlier phase (`src/anomaly_detector.py`,
`src/rule_engine.py`, `src/confidence.py`, `src/retriever.py`,
`src/genai_analyzer.py`, `src/response_planner.py`, `src/feedback.py`)
before this module ever sees it.

IMPORTANT - a SEPARATE database from Phase 7's feedback store:
    Analyst decisions still live ONLY in `config.FEEDBACK_DB_PATH`
    (`src/feedback.py`). This module uses its own file,
    `config.AUDIT_DB_PATH` (`data/audit_trail.db`), and never opens or
    modifies the feedback database's schema or rows.

IMPORTANT - observational only, never a write path into any decision:
    Nothing in this module can change an anomaly result, a rule
    classification, a severity, a confidence score, a RAG retrieval, a
    GenAI output, a response plan, or an analyst decision. It only
    RECORDS, after the fact, a structured summary of what already
    happened. There is no function here that takes an incident and
    produces a new decision - every public function here either writes
    one descriptive row or reads rows back.

IMPORTANT - append-only / traceability:
    `write_audit_event()` only ever INSERTs a new row (its primary key is
    a freshly generated `audit_id`, so it can never overwrite an existing
    event). Nothing in this module ever UPDATEs or DELETEs a stored row.
    The optional `skip_if_exists=True` flag makes a write a no-op when an
    event for the same `(incident_id, stage)` has already been recorded -
    this is what keeps a Streamlit rerun (which re-executes the whole
    page script on every widget interaction) from appending a new,
    identical row every time the page happens to redraw; it is never used
    to overwrite or replace a prior event.

IMPORTANT - true_label:
    Nothing in this module reads `config.LABEL_COLUMN`, imports it, or
    accepts it as a parameter. `AuditEvent` has no field for it at all.

IMPORTANT - data minimization:
    An audit event stores a small, structured provenance summary (IDs,
    the classification/severity/confidence values already decided,
    which rules/knowledge sources were involved, a short free-text
    `descriptive_details` line) - never a copy of the raw dataset row,
    never a full JSON dump of a response plan or a GenAI response. That
    level of detail is already available elsewhere (the Analyst Review /
    Feedback History pages, Phase 7's own database) when needed.

IMPORTANT - never crashes the app:
    Every public function here follows the exact `save_feedback()` /
    `load_feedback()` pattern from `src/feedback.py`: writes return
    `(success, messages)` and never raise; reads always return a
    (possibly empty) DataFrame/list and never raise, even if the database
    file is missing, corrupt, or its directory cannot be created.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import AUDIT_DB_PATH, AUDIT_STAGE_ORDER, AUDIT_STAGES

TABLE_NAME = "audit_trail"

# Columns stored exactly as named here - a single source of truth so the
# CREATE TABLE statement, INSERT statement, and row-parsing code can never
# silently drift apart (mirrors `src/feedback.py`'s `_COLUMNS` pattern).
_COLUMNS = [
    "audit_id",
    "incident_id",
    "timestamp",
    "stage",
    "event_type",
    "incident_type",
    "severity",
    "confidence_level",
    "confidence_score",
    "relevant_rule_ids",
    "supporting_indicators",
    "retrieved_knowledge_ids",
    "ai_model_name",
    "human_review_recommended",
    "analyst_decision",
    "descriptive_details",
]


# =======================================================================
# Result containers
# =======================================================================
@dataclass
class AuditEvent:
    """One recorded observation of a pipeline stage having run for one
    incident. Every field is optional except the five that identify and
    place the event (`audit_id`, `incident_id`, `timestamp`, `stage`,
    `event_type`) - a stage that doesn't produce a given value (e.g. no
    RAG sources for an event with no knowledge-base match) simply leaves
    that field empty/None rather than inventing one."""

    audit_id: str
    incident_id: str
    timestamp: str  # ISO 8601, UTC
    stage: str  # one of config.AUDIT_STAGES
    event_type: str  # short machine-readable label, e.g. "RULE_CLASSIFICATION_RESULT"
    incident_type: Optional[str] = None
    severity: Optional[str] = None
    confidence_level: Optional[str] = None
    confidence_score: Optional[float] = None
    relevant_rule_ids: List[str] = field(default_factory=list)
    supporting_indicators: List[str] = field(default_factory=list)
    retrieved_knowledge_ids: List[str] = field(default_factory=list)
    ai_model_name: Optional[str] = None
    human_review_recommended: Optional[bool] = None
    analyst_decision: Optional[str] = None
    descriptive_details: str = ""


@dataclass
class IncidentAuditSummary:
    """A human-readable summary of one incident's full audit history -
    what `render_explainability_audit_page()` needs for the "AUDIT
    TIMELINE" section, without the caller having to re-derive it."""

    incident_id: str
    total_events: int
    stages_recorded: List[str] = field(default_factory=list)  # in AUDIT_STAGES order, de-duplicated
    first_event_timestamp: Optional[str] = None
    last_event_timestamp: Optional[str] = None
    latest_analyst_decision: Optional[str] = None
    events: List[AuditEvent] = field(default_factory=list)  # chronological, oldest first


# =======================================================================
# (De)serialization helpers
# =======================================================================
def _dump_list(values) -> str:
    return json.dumps(list(values) if values else [])


def _load_list(raw) -> List[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(v) for v in data] if isinstance(data, list) else []


def generate_audit_id() -> str:
    """A unique audit event ID, e.g. `AUD-3f9a2c1b7e0d4c56`."""
    return f"AUD-{uuid.uuid4().hex}"


# =======================================================================
# Database connection helpers - every public function below goes through
# these, exactly mirroring `src/feedback.py`'s pattern, so the table is
# guaranteed to exist no matter which function is called first.
# =======================================================================
def _resolve_db_path(db_path: Optional[str]) -> str:
    return db_path if db_path else AUDIT_DB_PATH


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
            audit_id                 TEXT PRIMARY KEY,
            incident_id              TEXT NOT NULL,
            timestamp                TEXT NOT NULL,
            stage                    TEXT NOT NULL,
            event_type               TEXT NOT NULL,
            incident_type            TEXT,
            severity                 TEXT,
            confidence_level         TEXT,
            confidence_score         REAL,
            relevant_rule_ids        TEXT,
            supporting_indicators    TEXT,
            retrieved_knowledge_ids  TEXT,
            ai_model_name            TEXT,
            human_review_recommended INTEGER,
            analyst_decision         TEXT,
            descriptive_details      TEXT
        )
        """
    )
    conn.commit()


def initialize_audit_database(db_path: Optional[str] = None) -> bool:
    """
    Create the audit-trail database file and its table if they don't
    exist yet. Idempotent and safe to call any number of times, including
    before anything else has touched the database. Never raises - returns
    False (never crashes the Streamlit app) if the configured location
    genuinely cannot be created or opened.
    """
    try:
        conn = _connect(db_path)
        conn.close()
        return True
    except sqlite3.Error:
        return False
    except OSError:
        return False


# =======================================================================
# Write (append-only)
# =======================================================================
def event_exists(incident_id: str, stage: str, db_path: Optional[str] = None) -> bool:
    """
    True if at least one audit event already exists for this
    `(incident_id, stage)` pair. Used by `write_audit_event(...,
    skip_if_exists=True)` to avoid appending a duplicate, near-identical
    row every time Streamlit reruns a page for the same incident. Never
    raises - a database problem is treated as "no matching event found"
    so a write attempt can still proceed and surface its own error.
    """
    if not incident_id or not stage:
        return False
    try:
        conn = _connect(db_path)
    except (sqlite3.Error, OSError):
        return False
    try:
        row = conn.execute(
            f"SELECT 1 FROM {TABLE_NAME} WHERE incident_id = ? AND stage = ? LIMIT 1",
            (incident_id, stage),
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def write_audit_event(
    incident_id: str,
    stage: str,
    event_type: str,
    *,
    incident_type: Optional[str] = None,
    severity: Optional[str] = None,
    confidence_level: Optional[str] = None,
    confidence_score: Optional[float] = None,
    relevant_rule_ids: Optional[List[str]] = None,
    supporting_indicators: Optional[List[str]] = None,
    retrieved_knowledge_ids: Optional[List[str]] = None,
    ai_model_name: Optional[str] = None,
    human_review_recommended: Optional[bool] = None,
    analyst_decision: Optional[str] = None,
    descriptive_details: str = "",
    db_path: Optional[str] = None,
    skip_if_exists: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Append one audit event. NEVER raises - always returns `(success,
    messages)`, exactly like `src.feedback.save_feedback()`, so a caller
    (a Streamlit page) can show a clear message or simply ignore a
    non-fatal skip instead of crashing.

    Validation (malformed input handled safely, never a crash):
      - `incident_id` and `event_type` must be non-empty.
      - `stage` must be one of `config.AUDIT_STAGES`.
    A failed validation writes nothing and returns `(False, [errors])`.

    `skip_if_exists=True` makes this a safe no-op - returning `(True,
    [...])` with an explanatory message, not an error - when an event for
    this `(incident_id, stage)` already exists. Pass this for the
    deterministic pipeline stages (anomaly/rule/confidence/RAG/GenAI/
    response-plan) that Streamlit recomputes on every rerun, so the same
    incident's trace gets exactly one entry per stage instead of one per
    page redraw. Leave it False (the default) for stages that are
    genuinely new events every time they happen - e.g. ANALYST_REVIEW,
    where every analyst decision (its own new `feedback_id`) is a
    distinct, real event worth its own row.
    """
    errors: List[str] = []
    if not incident_id or not str(incident_id).strip():
        errors.append("incident_id is required.")
    if not event_type or not str(event_type).strip():
        errors.append("event_type is required.")
    if stage not in AUDIT_STAGES:
        errors.append(f"'{stage}' is not a valid audit stage (must be one of {AUDIT_STAGES}).")
    if errors:
        return False, errors

    if skip_if_exists and event_exists(incident_id, stage, db_path):
        return True, [f"An audit event already exists for {incident_id}/{stage}; skipped duplicate write."]

    event = AuditEvent(
        audit_id=generate_audit_id(),
        incident_id=str(incident_id),
        timestamp=datetime.now(timezone.utc).isoformat(),
        stage=stage,
        event_type=str(event_type),
        incident_type=incident_type,
        severity=severity,
        confidence_level=confidence_level,
        confidence_score=float(confidence_score) if confidence_score is not None else None,
        relevant_rule_ids=list(relevant_rule_ids or []),
        supporting_indicators=list(supporting_indicators or []),
        retrieved_knowledge_ids=list(retrieved_knowledge_ids or []),
        ai_model_name=ai_model_name,
        human_review_recommended=human_review_recommended,
        analyst_decision=analyst_decision,
        descriptive_details=descriptive_details or "",
    )

    try:
        conn = _connect(db_path)
    except (sqlite3.Error, OSError) as exc:
        return False, [f"Could not open the audit-trail database: {exc}"]

    try:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME} (
                audit_id, incident_id, timestamp, stage, event_type,
                incident_type, severity, confidence_level, confidence_score,
                relevant_rule_ids, supporting_indicators, retrieved_knowledge_ids,
                ai_model_name, human_review_recommended, analyst_decision,
                descriptive_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.audit_id,
                event.incident_id,
                event.timestamp,
                event.stage,
                event.event_type,
                event.incident_type,
                event.severity,
                event.confidence_level,
                event.confidence_score,
                _dump_list(event.relevant_rule_ids),
                _dump_list(event.supporting_indicators),
                _dump_list(event.retrieved_knowledge_ids),
                event.ai_model_name,
                None if event.human_review_recommended is None else (1 if event.human_review_recommended else 0),
                event.analyst_decision,
                event.descriptive_details,
            ),
        )
        conn.commit()
        return True, []
    except sqlite3.IntegrityError:
        # A collision on the freshly generated audit_id is effectively
        # impossible (128-bit uuid4), but handled gracefully rather than
        # crashing, exactly like Phase 7's save_feedback().
        return False, [f"An audit event with ID '{event.audit_id}' already exists."]
    except sqlite3.Error as exc:
        return False, [f"Could not write the audit event: {exc}"]
    finally:
        conn.close()


# =======================================================================
# Read
# =======================================================================
def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["human_review_recommended"] = (
        None if d.get("human_review_recommended") is None else bool(d["human_review_recommended"])
    )
    for list_field in ("relevant_rule_ids", "supporting_indicators", "retrieved_knowledge_ids"):
        d[list_field] = _load_list(d.get(list_field))
    return d


def load_audit_events(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load every recorded audit event, newest first. NEVER raises - a
    missing/empty/broken database safely returns an empty (but
    correctly-columned) DataFrame, mirroring `src.feedback.load_feedback()`.
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


def get_events_by_incident(incident_id: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Every audit event for one incident, in CHRONOLOGICAL order (oldest
    first) - the natural reading order for a trace like `INC-0007:
    ANOMALY_DETECTION -> RULE_CLASSIFICATION -> ... -> ANALYST_REVIEW`.
    An incident with no recorded events safely returns an empty
    DataFrame.
    """
    all_events = load_audit_events(db_path)
    if all_events.empty:
        return all_events
    matching = all_events[all_events["incident_id"] == incident_id]
    return matching.sort_values("timestamp", ascending=True).reset_index(drop=True)


def get_events_by_stage(stage: str, db_path: Optional[str] = None) -> pd.DataFrame:
    """Every audit event recorded for one pipeline stage, newest first,
    across all incidents. An unrecognized/absent stage safely returns an
    empty DataFrame rather than raising."""
    all_events = load_audit_events(db_path)
    if all_events.empty:
        return all_events
    return all_events[all_events["stage"] == stage].reset_index(drop=True)


def summarize_incident_audit_history(incident_id: str, db_path: Optional[str] = None) -> IncidentAuditSummary:
    """
    Build the human-readable audit-history summary for one incident, used
    directly by the Explainability & Audit Trail page's "AUDIT TIMELINE"
    section. Never raises - an incident with no recorded events safely
    returns an all-empty summary.
    """
    events_df = get_events_by_incident(incident_id, db_path)
    if events_df.empty:
        return IncidentAuditSummary(incident_id=incident_id, total_events=0)

    events: List[AuditEvent] = [AuditEvent(**{col: row[col] for col in _COLUMNS}) for _, row in events_df.iterrows()]

    seen_stages: List[str] = []
    for ev in events:
        if ev.stage not in seen_stages:
            seen_stages.append(ev.stage)
    stages_recorded = sorted(seen_stages, key=lambda s: AUDIT_STAGE_ORDER.get(s, len(AUDIT_STAGES)))

    analyst_events = [ev for ev in events if ev.stage == "ANALYST_REVIEW" and ev.analyst_decision]
    latest_decision = analyst_events[-1].analyst_decision if analyst_events else None

    return IncidentAuditSummary(
        incident_id=incident_id,
        total_events=len(events),
        stages_recorded=stages_recorded,
        first_event_timestamp=events[0].timestamp,
        last_event_timestamp=events[-1].timestamp,
        latest_analyst_decision=latest_decision,
        events=events,
    )
