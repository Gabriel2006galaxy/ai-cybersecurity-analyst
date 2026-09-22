"""
config.py
---------
Central configuration for the AI-Based Cybersecurity Analyst Assistant.

Keeping all paths, constants, and tunable settings in one place makes the
project easier to maintain as more phases (anomaly detection, rule engine,
classification, GenAI summary, response planning, feedback) are added.
"""

import os

# ---------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_FILE_PATH = os.path.join(DATA_DIR, "synthetic_security_logs.csv")

# Early Phase 0 placeholder - superseded by FEEDBACK_DB_PATH below (Phase 7
# uses a small local SQLite database instead of a CSV, so feedback records
# stay structured and safely queryable). Left in place, unused, so nothing
# that referenced it earlier ever breaks.
FEEDBACK_FILE_PATH = os.path.join(DATA_DIR, "analyst_feedback.csv")

# Phase 7 - the actual persistent store for analyst feedback (see
# src/feedback.py). SQLite via Python's built-in sqlite3 module - no
# external database dependency. Lives in data/ alongside the dataset, but
# is a completely separate file, so it never touches - or is touched by -
# the checksummed synthetic_security_logs.csv.
FEEDBACK_DB_PATH = os.path.join(DATA_DIR, "analyst_feedback.db")

# Phase 8 - local defensive knowledge base used by the RAG retriever (see
# src/retriever.py). A directory of small, curated, hand-written JSON
# reference documents - not part of the generated/checksummed dataset and
# never modified by the running application.
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")

# ---------------------------------------------------------------------
# App-level settings
# ---------------------------------------------------------------------
APP_TITLE = "AI-Based Cybersecurity Analyst Assistant"
APP_ICON = "🛡️"
APP_LAYOUT = "wide"

# Shown in the dashboard header / README so it's always clear this is a
# defensive-only, academic demonstration project.
PROJECT_TAGLINE = (
    "A defensive, student-built assistant that helps a SOC analyst "
    "triage security logs — detect anomalies, apply rules, classify "
    "incidents, and get an AI-generated summary with a response plan."
)

# Fixed status colors used consistently across every anomaly-detection
# chart (never remapped by filters - "color follows the entity").
STATUS_COLOR_MAP = {"NORMAL": "#0ca30c", "ANOMALOUS": "#d03b3b"}

# Fixed status colors for Phase 4's confidence LEVEL (display only - plays
# no role in the confidence calculation itself). Deliberately different
# hues from STATUS_COLOR_MAP above so a chart can never make "low
# confidence" look like "anomalous" - severity/status and confidence are
# different concepts (see Phase 4 README section) and are never colored
# the same way. Values are the "good" / "warning" / "serious" steps of the
# dataviz skill's validated status palette.
CONFIDENCE_COLOR_MAP = {"LOW": "#ec835a", "MEDIUM": "#fab219", "HIGH": "#0ca30c"}

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# Dataset generation settings (used by generate_data.py)
# ---------------------------------------------------------------------
NUM_RECORDS = 350          # total synthetic log records to generate
SUSPICIOUS_RATIO = 0.22    # approx. share of records that look suspicious

# ---------------------------------------------------------------------
# Phase 1 - Data preprocessing & feature engineering
# ---------------------------------------------------------------------
TIMESTAMP_COLUMN = "timestamp"

# Ground-truth column. Used ONLY to evaluate later phases (anomaly
# detection, classification) - it must never be used as an input feature.
LABEL_COLUMN = "true_label"

REQUIRED_NUMERICAL_COLUMNS = [
    "failed_logins",
    "request_count",
    "port",
    "data_transferred_mb",
    "session_duration",
    "device_count",
]

REQUIRED_CATEGORICAL_COLUMNS = [
    "user",
    "source_ip",
    "destination_ip",
    "event_type",
    "login_location",
]

# Low-cardinality categorical column(s) that are safe to one-hot encode.
LOW_CARDINALITY_CATEGORICAL_COLUMNS = ["event_type"]

# Identifier-style columns: too high-cardinality (or too identifying) for
# one-hot encoding. Their useful signal is instead captured with small
# derived features in engineer_features() (e.g. is_external_source_ip,
# location_mismatch), and the raw columns are kept only for display.
IDENTIFIER_COLUMNS = ["user", "source_ip", "destination_ip", "login_location"]

# Off-hours definition (documented, simple): 22:00-23:59 or 00:00-05:59
OFF_HOURS_START_HOUR = 22   # hour >= this is off-hours (evening/night)
OFF_HOURS_END_HOUR = 6      # hour < this is off-hours (early morning)

# ---------------------------------------------------------------------
# Phase 2 - AI Anomaly Detection (Isolation Forest)
# ---------------------------------------------------------------------
# Primary, required parameters for sklearn.ensemble.IsolationForest.
ISOLATION_FOREST_N_ESTIMATORS = 100

# Expected proportion of anomalies in the data. Chosen close to (but not
# exactly equal to) the ~22% suspicious rate used to *generate* the
# synthetic dataset (see SUSPICIOUS_RATIO in Phase 0) - a realistic
# stand-in for the rough estimate a SOC analyst might give in practice.
# This is a modelling hyperparameter set once, up front; it is NOT read
# from true_label at runtime and is not tuned by looking at evaluation
# results. Fully configurable from the Streamlit UI.
ISOLATION_FOREST_CONTAMINATION = 0.2

ISOLATION_FOREST_RANDOM_STATE = RANDOM_SEED

# ---------------------------------------------------------------------
# Phase 2 (optional) - Self-Organizing Map
# Exploratory clustering visualization only - never a second anomaly
# detector and never required for the app to work.
# ---------------------------------------------------------------------
SOM_GRID_SIZE = None       # None -> auto-sized from the number of events
SOM_NUM_ITERATIONS = 200

# ---------------------------------------------------------------------
# Phase 3 - Knowledge-Based Rule Engine & Incident Classification
# ---------------------------------------------------------------------
# All thresholds below are explainable, fixed rule thresholds chosen from
# how the synthetic dataset was constructed (Phase 0) and general
# defensive-security reasoning - NOT learned parameters, and never tuned
# by looking at true_label / evaluation metrics.

# "high_failed_logins" fact: normal activity has 0-2 failed logins;
# the brute-force generator pattern uses 6-20. 5 sits cleanly between them.
FAILED_LOGIN_THRESHOLD = 5

# "high_request_count" fact: normal activity (incl. brute-force attempts)
# tops out around 120 requests; the port-scan pattern uses 200-2000.
# 150 is chosen above the normal/brute-force range so this fact stays a
# clean signal of scan-like volume rather than overlapping brute force.
REQUEST_COUNT_THRESHOLD = 150

# "high_data_transfer" fact, in MB: normal transfers rarely exceed ~50MB
# (exponential distribution, scale=8); the exfiltration pattern uses
# 500-5000MB. 100MB sits well above normal with a comfortable margin.
DATA_TRANSFER_THRESHOLD = 100

# Secondary, higher threshold used only to escalate an already-triggered
# suspicious-data-transfer rule to HIGH severity ("severe data-transfer
# behavior") - roughly the midpoint of the 500-5000MB exfiltration range.
SEVERE_DATA_TRANSFER_THRESHOLD = 1000

# "multiple_devices" fact: normal activity uses 1-2 devices; the
# many-devices generator pattern uses 4-9. 3 sits cleanly between them.
DEVICE_COUNT_THRESHOLD = 3

# Ports treated as ordinary/expected traffic for the "unusual_port" fact.
# Single source of truth - generate_data.py imports this list too, so the
# rule engine's notion of "common" always matches how the dataset was built.
COMMON_PORTS = [80, 443, 22, 25, 53, 3306, 3389, 8443]

# Exact event_type values that count as privilege-escalation / network-scan
# indicators. Explicit membership checks (not substring matching) keep
# rule conditions precise and unambiguous.
PRIVILEGE_ESCALATION_EVENT_TYPES = {"privilege_escalation_attempt"}
NETWORK_SCAN_EVENT_TYPES = {"network_scan_detected"}

# ---------------------------------------------------------------------
# Phase 4 - Uncertainty / Confidence Scoring
# ---------------------------------------------------------------------
# Confidence-level thresholds (unchanged from their Phase 0 placeholder
# definitions - Requirement #9 says not to silently redefine them):
#   score <  LOW_CONFIDENCE_THRESHOLD                              -> LOW
#   LOW_CONFIDENCE_THRESHOLD <= score < HIGH_CONFIDENCE_THRESHOLD  -> MEDIUM
#   score >= HIGH_CONFIDENCE_THRESHOLD                             -> HIGH
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Component weights for the evidence-based confidence formula
# (src/confidence.py). Deterministic, hand-picked, and documented here -
# never tuned by looking at true_label or any evaluation metric. They must
# sum to 1.0 so the final blended score stays naturally within [0, 1].
#
#   - ANOMALY and RULES get the largest, roughly-equal shares because
#     they are this project's two primary interpretation signals: the
#     Phase 2 AI model and the Phase 3 knowledge base, respectively.
#   - SUPPORTING gets a smaller, independent share since most of its
#     signal is already folded into whichever rules fired.
#   - CONSISTENCY gets a moderate share so that agreement/disagreement
#     across the other three sources meaningfully moves the score, without
#     being able to dominate it on its own.
CONFIDENCE_WEIGHT_ANOMALY = 0.35
CONFIDENCE_WEIGHT_RULES = 0.30
CONFIDENCE_WEIGHT_SUPPORTING = 0.15
CONFIDENCE_WEIGHT_CONSISTENCY = 0.20

# --- Rule-evidence component tiers ---
# How strongly the NUMBER of triggered Phase 3 rules supports the current
# interpretation. Chosen ordinally (0 rules < 1 rule < 2+ rules) from plain
# reasoning about corroborating evidence, not fit to any metric.
RULE_EVIDENCE_ZERO_RULES = 0.2        # anomalous, but no rule matched ("bare" anomaly)
RULE_EVIDENCE_ONE_RULE = 0.6          # anomalous, exactly one rule matched
RULE_EVIDENCE_MULTIPLE_RULES = 0.9    # anomalous, two or more rules matched
# Rules never evaluate for a Normal Activity classification (every rule's
# condition starts with `anomalous_event AND ...`), so this component is
# structurally inapplicable there. A fixed neutral baseline is used instead
# of 0.0 or 1.0, so it neither drags down nor artificially inflates
# confidence for normal events - the real signal for those comes from the
# anomaly, supporting-indicator, and consistency components instead.
RULE_EVIDENCE_NORMAL_BASELINE = 0.6
# Small bonus applied when a triggered rule's conclusion is one of
# rule_engine.HIGH_RISK_CONCLUSIONS (currently: off-hours privilege
# activity) - reflects Requirement #6 ("consider whether the triggered
# rule is a stronger security indicator"), capped so it can never push the
# component above 1.0.
STRONG_RULE_EVIDENCE_BONUS = 0.1

# --- Evidence-consistency component tiers ---
# Same three numeric tiers are reused for both the anomalous and normal
# branches of src.confidence.compute_consistency() - only the conditions
# that select a tier differ (see that function's docstring).
CONSISTENCY_HIGH = 1.0
CONSISTENCY_MEDIUM = 0.6
CONSISTENCY_LOW = 0.25

# Phase 5 - Generative AI (Ollama)
OLLAMA_MODEL_NAME = "qwen3:1.7b"   # change to whichever local model is installed

# ---------------------------------------------------------------------
# Phase 6 - Defensive Response Planning
# ---------------------------------------------------------------------
# Ordered low -> high. A plan/step's priority is always one of these -
# never a numeric score, to keep it clearly distinct from Phase 4's
# confidence_score (Requirement: don't merge severity/confidence/priority
# concepts).
PLAN_PRIORITY_LEVELS = ["ROUTINE", "STANDARD", "HIGH", "URGENT"]

# Defense-in-depth safety net (see src/response_planner.py's
# validate_plan()): a generated step's action text must never contain any
# of these - the planner is only ever allowed to recommend advisory,
# policy-based defensive steps, never a specific offensive/automated
# action against a real system. This is checked even though the planner's
# own step library never produces such text, exactly so a future bug or
# careless edit is caught immediately instead of silently shipping.
PROHIBITED_PLAN_ACTION_KEYWORDS = [
    "block ip", "block the ip", "disable account", "disable the account",
    "disable the user", "terminate session", "terminate the session",
    "isolate machine", "isolate the machine", "isolate the host",
    "modify firewall", "modify the firewall", "reconfigure firewall",
    "exploit", "attack the", "launch an attack", "hack", "run command",
    "execute payload", "delete user", "delete account", "wipe", "shut down",
    "kill process", "revoke all access",
]

# ---------------------------------------------------------------------
# Phase 7 - Human Analyst Review & Feedback Loop
# ---------------------------------------------------------------------
# The three decisions an analyst can make on a reviewed incident. Fixed,
# closed set - `src/feedback.py` rejects anything else rather than
# guessing at an unrecognized decision.
ANALYST_DECISIONS = ["APPROVE", "REJECT", "MODIFY"]

# A reviewable incident's lifecycle. PENDING is never stored in the
# feedback database itself - it is the derived state of any incident
# that has no feedback record yet (see src/feedback.get_review_status_map).
# The other three are stored, one-to-one with ANALYST_DECISIONS above.
REVIEW_STATUS_PENDING = "PENDING"
REVIEW_STATUSES = ["PENDING", "APPROVED", "REJECTED", "MODIFIED"]
DECISION_TO_REVIEW_STATUS = {
    "APPROVE": "APPROVED",
    "REJECT": "REJECTED",
    "MODIFY": "MODIFIED",
}

# Simple, common English words excluded when summarizing analyst notes
# into keyword counts (src/feedback.summarize_feedback) - keeps the
# "common analyst note keywords" insight meaningful (content words only)
# rather than dominated by "the", "and", "this", etc. Deliberately small
# and hand-written - no NLP dependency for an academic project.
FEEDBACK_NOTE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "for", "with", "this", "that", "it",
    "as", "at", "by", "from", "has", "have", "had", "not", "no", "so",
    "if", "than", "then", "there", "their", "its", "did", "does", "do",
    "will", "would", "should", "could", "may", "might", "can", "still",
}

# ---------------------------------------------------------------------
# Phase 8 - RAG / Knowledge Base Integration
# ---------------------------------------------------------------------
# Lightweight retrieval tuning for src/retriever.py. Both are fixed,
# hand-picked constants - never learned, never tuned by looking at
# true_label or any evaluation metric, and never touched by the retriever
# itself at runtime.
#
#   RAG_TOP_K          - at most this many knowledge-base entries are
#                         attached to a single incident's GenAI context.
#   RAG_MIN_SIMILARITY - a candidate entry below this cosine-similarity
#                         score (TF-IDF vector space) is treated as "not a
#                         useful match" and dropped, so an incident with no
#                         genuinely relevant reference material simply
#                         gets an empty retrieval result instead of a
#                         forced, weakly-related one.
RAG_TOP_K = 3
RAG_MIN_SIMILARITY = 0.05

# ---------------------------------------------------------------------
# Phase 9 - Continuous Improvement from Analyst Feedback
# ---------------------------------------------------------------------
# All read-only reporting knobs for src/continuous_improvement.py. None
# of these are learned or tuned against true_label/any evaluation metric,
# and changing them can never change the anomaly detector, rule engine,
# confidence model, RAG retriever, or response planner - this module has
# no write path into any of them.
#
#   CI_MIN_PATTERN_OCCURRENCES - how many times something (a REJECT/
#                                 MODIFY incident type, a modified plan
#                                 step, an analyst-note keyword) must
#                                 recur across DISTINCT feedback records
#                                 before it counts as a "repeated"/
#                                 "frequent" pattern worth surfacing. A
#                                 single one-off review never generates a
#                                 suggestion on its own.
#   CI_TOP_N_MODIFIED_STEPS    - how many frequently-modified-step
#                                 patterns to keep/display, most-frequent
#                                 first.
#   CI_TOP_N_NOTE_THEMES       - how many recurring analyst-note keyword
#                                 themes to keep/display, most-frequent
#                                 first.
#   CI_TOP_N_SUGGESTIONS       - a display cap on the number of
#                                 improvement suggestions returned, most
#                                 significant first - never a limit on
#                                 what's detected, just on how much is
#                                 shown at once.
CI_MIN_PATTERN_OCCURRENCES = 2
CI_TOP_N_MODIFIED_STEPS = 5
CI_TOP_N_NOTE_THEMES = 10
CI_TOP_N_SUGGESTIONS = 15

# ---------------------------------------------------------------------
# Phase 10 - Explainability & Audit Trail
# ---------------------------------------------------------------------
# `src/audit_trail.py` uses a SEPARATE SQLite database from Phase 7's
# feedback store (FEEDBACK_DB_PATH above) - the audit trail is an
# observational log of what the pipeline did, never analyst decisions
# themselves (those still live only in FEEDBACK_DB_PATH).
AUDIT_DB_PATH = os.path.join(DATA_DIR, "audit_trail.db")

# The fixed set of pipeline stages an audit event can belong to, in their
# natural chronological order for one incident's trace. `src/audit_trail.py`
# validates every write against this list so a typo can never silently
# create an untracked "stage". `src/explainability.py` and the
# Explainability & Audit Trail page reuse this same list/order so the
# on-screen timeline is never independently re-ordered.
AUDIT_STAGES = [
    "INCIDENT_SELECTED",
    "ANOMALY_DETECTION",
    "RULE_CLASSIFICATION",
    "CONFIDENCE",
    "RAG_RETRIEVAL",
    "GENAI_ANALYSIS",
    "RESPONSE_PLAN",
    "ANALYST_REVIEW",
]
AUDIT_STAGE_ORDER = {stage: i for i, stage in enumerate(AUDIT_STAGES)}
