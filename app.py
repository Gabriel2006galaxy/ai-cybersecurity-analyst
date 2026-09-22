"""
app.py
------
Streamlit entry point for the AI-Based Cybersecurity Analyst Assistant.

UI STRUCTURE (a presentation-only redesign - no detection, classification,
confidence, RAG, GenAI, response-planning, feedback, continuous-improvement,
explainability, audit, or evaluation LOGIC was changed by this redesign;
every one of those modules is still called exactly as it always was):

    - "Case Analysis" - the main screen. An analyst selects one security
      case and sees, in one continuous flow: case details, an AI-generated
      case summary, why the case was flagged (in plain English, with a
      technical-details expander for the underlying evidence), the
      knowledge-base sources the AI used, the recommended next steps (the
      advisory response plan), and the final analyst decision (APPROVE /
      REJECT / MODIFY). This consolidates what used to be seven separate
      pages into one case-centered workflow.

    - "Review & Insights" - everything an analyst needs to look back on:
      Case History, Analyst Decisions, Feedback, Continuous Improvement,
      Explainability, Audit Trail, and Evaluation / Performance, organized
      as tabs on one page instead of six separate top-level pages.

Every computation in this file comes from the same, unmodified src/
modules and the same config.py constants as always - this file remains a
thin orchestrator that never contains pipeline logic itself. `true_label`
is still never used anywhere in this file except being passed through to
the strictly offline, post-hoc evaluation report built in
src/evaluation.py (Review & Insights -> Evaluation / Performance).
"""

import dataclasses
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    ANALYST_DECISIONS,
    APP_ICON,
    APP_LAYOUT,
    APP_TITLE,
    AUDIT_STAGES,
    CI_MIN_PATTERN_OCCURRENCES,
    CONFIDENCE_COLOR_MAP,
    DECISION_TO_REVIEW_STATUS,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE,
    KNOWLEDGE_BASE_DIR,
    OLLAMA_MODEL_NAME,
    PLAN_PRIORITY_LEVELS,
    PROJECT_TAGLINE,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUSES,
    STATUS_COLOR_MAP,
)
from src.anomaly_detector import (
    DISPLAY_COLUMNS,
    evaluate_against_ground_truth,
    is_som_available,
    run_anomaly_detection,
    run_som_analysis,
)
from src.audit_trail import (
    get_events_by_incident,
    initialize_audit_database,
    summarize_incident_audit_history,
    write_audit_event,
)
from src.confidence import run_confidence_scoring
from src.continuous_improvement import generate_continuous_improvement_report
from src.data_loader import get_basic_stats, load_logs
from src.evaluation import (
    build_evaluation_report,
    measure_anomaly_detection_timing,
    measure_genai_generation_timing,
    measure_rag_retrieval_timing,
)
from src.explainability import build_incident_explanation
from src.feedback import (
    FeedbackRecord,
    build_incident_id,
    dict_to_plan,
    generate_feedback_id,
    get_feedback_by_incident,
    get_review_status_map,
    initialize_feedback_database,
    load_feedback,
    plan_to_dict,
    save_feedback,
    summarize_feedback,
    validate_feedback,
)
from src.genai_analyzer import build_incident_context, generate_ai_case_summary, get_ollama_status
from src.preprocessor import run_preprocessing_pipeline
from src.retriever import (
    load_knowledge_base,
    retrieve_knowledge_for_incident,
    retrieved_items_to_dicts,
)
from src.response_planner import (
    PLAN_ORDERING_RATIONALE,
    PLAN_STAGE_LABELS,
    PLAN_STAGE_NAMES,
    PlanStep,
    ResponsePlanResult,
    generate_response_plan,
    run_response_planning,
    validate_plan,
)
from src.rule_engine import FACT_DESCRIPTIONS, INCIDENT_TYPES, RULES, run_rule_engine

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)


# ---------------------------------------------------------------------
# Light, modern styling. Presentation only - every selector below is a
# cosmetic (color/spacing/radius) tweak to Streamlit's own DOM; it changes
# no widget behavior and is invisible to streamlit.testing.v1.AppTest,
# which inspects the widget tree, not rendered CSS.
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --brand-blue: #2563EB;
    --brand-blue-dark: #1D4ED8;
    --brand-cyan: #06B6D4;
    --brand-cyan-dark: #0891B2;
    --brand-purple: #7C3AED;
    --brand-purple-dark: #6D28D9;
    --brand-green: #10B981;
    --brand-green-dark: #059669;
    --brand-red: #EF4444;
    --brand-red-dark: #DC2626;
    --brand-orange: #F59E0B;
    --brand-orange-dark: #D97706;
    --deep-navy: #172554;
    --ink: #1F2430;
    /* One consistent medium-light blue/lavender tone for the whole app
       shell (sidebar + header + main background) so it reads as a single
       colored canvas, with light/white cards standing out on top of it. */
    --shell-base: #CDD4EC;
}

/* ---- Page background: a unified medium-light blue/lavender shell tone,
   with soft blue/cyan/purple highlight glows and a very faint
   cybersecurity-style grid/node motif layered on top. Still reads as
   light, colorful, and professional - not dark, not heavily saturated. ---- */
.stApp {
    background:
        radial-gradient(1200px 620px at 10% -8%, rgba(37, 99, 235, 0.10), transparent 58%),
        radial-gradient(1000px 520px at 82% 6%, rgba(6, 182, 212, 0.08), transparent 55%),
        radial-gradient(1300px 700px at 92% 96%, rgba(124, 58, 237, 0.10), transparent 60%),
        radial-gradient(1500px 480px at 50% 0%, rgba(99, 102, 241, 0.08), transparent 62%),
        repeating-linear-gradient(90deg, rgba(23, 37, 84, 0.035) 0px, rgba(23, 37, 84, 0.035) 1px, transparent 1px, transparent 96px),
        repeating-linear-gradient(0deg, rgba(23, 37, 84, 0.035) 0px, rgba(23, 37, 84, 0.035) 1px, transparent 1px, transparent 96px),
        var(--shell-base);
}

/* Streamlit's own top toolbar (Deploy / menu) - match the same shell
   tone so the header no longer reads as a separate white strip. */
header[data-testid="stHeader"] {
    background: var(--shell-base) !important;
    background-image: none !important;
}

/* ---- Use more of the available width, less dead margin ---- */
div[data-testid="stAppViewBlockContainer"] {
    max-width: 1360px !important;
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* ---- Typography: premium but restrained ---- */
h1, h2, h3 { color: var(--ink); }
h1 { color: var(--deep-navy); font-weight: 800; letter-spacing: -0.01em; }
h2, h3 { font-weight: 700; }
div[data-testid="stMetricValue"], div.stButton > button p { font-weight: 700; }

/* ---- Sidebar: simple, compact, still on-brand, with a soft glow on
   the active nav item. Same shell tone as the rest of the app so the
   whole application feels like one consistent colored canvas. ---- */
section[data-testid="stSidebar"] {
    background: var(--shell-base);
    border-right: 1px solid rgba(23, 37, 84, 0.10);
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    width: 100%;
    transition: box-shadow 0.18s ease, background 0.18s ease;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, var(--brand-blue) 0%, var(--brand-purple) 100%);
    border-color: transparent;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.30), 0 0 18px rgba(124, 58, 237, 0.28);
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* ---- Cards / bordered containers - soft depth + gentle hover lift.
   Kept light so they still clearly stand out from the #CDD4EC app-shell
   background, but with a soft cool blue-white tint (~#F2F5FC) instead
   of pure white, so the softer contrast reads as one harmonious system
   with the blue/cyan section headings instead of washing them out.

   IMPORTANT: Streamlit reuses the SAME data-testid
   ("stVerticalBlockBorderWrapper") for every vertical-block wrapper in
   the DOM - including the single giant wrapper around the entire page
   and the wrapper around each st.columns() column - not only for real
   st.container(border=True) cards. Targeting the testid alone was
   painting that page-wide wrapper (and every column) solid white,
   which hid the colored app-shell background. Real bordered
   containers are reliably distinguished (empirically verified against
   this app's Streamlit 1.38.0) by an additional emotion-cache class
   Streamlit only applies when border=True; plain/implicit wrappers get
   "st-emotion-cache-0" (no styles) instead. Scoping to that class keeps
   this rule on genuine cards only. */
div[data-testid="stVerticalBlockBorderWrapper"].st-emotion-cache-1gp6n1k {
    background: linear-gradient(180deg, #F6F9FD 0%, #F2F5FC 100%);
    border-radius: 16px;
    border: 1px solid #DEE4F5;
    box-shadow: 0 4px 18px rgba(23, 37, 84, 0.07);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"].st-emotion-cache-1gp6n1k:hover {
    box-shadow: 0 8px 26px rgba(23, 37, 84, 0.11);
    transform: translateY(-1px);
}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #F6F9FD 0%, #F1F4FC 100%);
    border-radius: 14px;
    padding: 1rem 1.1rem 0.85rem 1.1rem;
    border: 1px solid #DEE4F5;
    box-shadow: 0 2px 10px rgba(23, 37, 84, 0.05);
    transition: box-shadow 0.2s ease;
}
div[data-testid="stMetricLabel"] { color: #6B7280; font-weight: 600; }
div[data-testid="stMetricValue"] {
    white-space: normal;
    overflow-wrap: break-word;
    line-height: 1.2;
    font-size: 1.6rem;
    color: var(--ink);
}

/* Small colored icon-circle badges placed just above a metric card
   (added via st.markdown immediately before st.metric) - purely
   decorative, the metric widget itself is untouched. Each uses a
   two-tone gradient per the approved palette, with a soft matching
   glow so the four cards read as visually distinct but professional. */
.status-badge {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; margin-bottom: 6px;
}
.badge-blue {
    background: linear-gradient(135deg, #DCE7FF 0%, #D5F4FA 100%);
    box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.12), 0 0 14px rgba(6, 182, 212, 0.25);
}
.badge-orange {
    background: linear-gradient(135deg, #FDECD3 0%, #FBDCDC 100%);
    box-shadow: 0 0 0 1px rgba(217, 119, 6, 0.14), 0 0 14px rgba(239, 68, 68, 0.20);
}
.badge-green {
    background: linear-gradient(135deg, #D7F5E9 0%, #D5F4FA 100%);
    box-shadow: 0 0 0 1px rgba(5, 150, 105, 0.14), 0 0 14px rgba(6, 182, 212, 0.22);
}
.badge-purple {
    background: linear-gradient(135deg, #EAE0FC 0%, #E3E8FE 100%);
    box-shadow: 0 0 0 1px rgba(124, 58, 237, 0.14), 0 0 14px rgba(124, 58, 237, 0.22);
}
div.element-container:has(.badge-blue)   + div.element-container div[data-testid="stMetric"] {
    border-top: 4px solid var(--brand-blue);
    box-shadow: 0 2px 10px rgba(23, 37, 84, 0.05), 0 0 20px rgba(6, 182, 212, 0.10);
}
div.element-container:has(.badge-orange) + div.element-container div[data-testid="stMetric"] {
    border-top: 4px solid var(--brand-orange);
    box-shadow: 0 2px 10px rgba(23, 37, 84, 0.05), 0 0 20px rgba(239, 68, 68, 0.09);
}
div.element-container:has(.badge-green)  + div.element-container div[data-testid="stMetric"] {
    border-top: 4px solid var(--brand-green);
    box-shadow: 0 2px 10px rgba(23, 37, 84, 0.05), 0 0 20px rgba(6, 182, 212, 0.10);
}
div.element-container:has(.badge-purple) + div.element-container div[data-testid="stMetric"] {
    border-top: 4px solid var(--brand-purple);
    box-shadow: 0 2px 10px rgba(23, 37, 84, 0.05), 0 0 20px rgba(124, 58, 237, 0.10);
}

/* Section-heading accent icons (added via a tiny marker immediately
   before the st.subheader() call - the heading TEXT is never touched,
   only its color/icon are styled). */
div.element-container:has(.marker-ai-heading) + div.element-container h3 {
    color: var(--brand-purple-dark);
    text-shadow: 0 0 24px rgba(124, 58, 237, 0.18);
}
div.element-container:has(.marker-ai-heading) + div.element-container h3::before {
    content: "\\2728  ";
}
div.element-container:has(.marker-kb-heading) + div.element-container h3 {
    color: var(--brand-blue-dark);
}
div.element-container:has(.marker-kb-heading) + div.element-container h3::before {
    content: "\\1F4DA  ";
}
div.element-container:has(.marker-flagged-heading) + div.element-container h3 {
    color: var(--brand-cyan-dark);
}
div.element-container:has(.marker-flagged-heading) + div.element-container h3::before {
    content: "\\1F50E  ";
}
div.element-container:has(.marker-next-steps-heading) + div.element-container h3 {
    color: var(--brand-blue-dark);
}
div.element-container:has(.marker-next-steps-heading) + div.element-container h3::before {
    content: "\\1F6E1\\FE0F  ";
}

/* AI Case Summary result card - purple -> blue gradient accent with a
   soft neon glow around the card (subtle, not the whole card lit up).
   A subtle lavender tint throughout (never pure white) for the AI/RAG
   surface, per the card-tint pass. */
div.element-container:has(.marker-ai-summary) + div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(155deg, #F3EEFC 0%, #F1F6FC 55%, #F1EFFA 100%);
    border: 1px solid #E4D9FB;
    box-shadow: 0 4px 18px rgba(23, 37, 84, 0.07), 0 0 34px rgba(124, 58, 237, 0.14);
}

/* "Generated by / Automatic summary" line inside the AI Case Summary
   card, turned into a small polished AI-model badge pill (same text,
   only its container is styled). */
div.element-container:has(.marker-ai-badge) + div.element-container {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.10) 0%, rgba(37, 99, 235, 0.10) 100%);
    border: 1px solid rgba(124, 58, 237, 0.22);
    border-radius: 999px;
    padding: 3px 14px;
    margin-bottom: 10px;
    box-shadow: 0 0 12px rgba(124, 58, 237, 0.12);
}

/* Knowledge Base item cards - blue/purple RAG accent, lavender tint
   throughout (never pure white), per the card-tint pass. */
div.element-container:has(.marker-kb-item) + div[data-testid="stVerticalBlockBorderWrapper"] {
    border-left: 4px solid var(--brand-purple);
    background: linear-gradient(155deg, #F5F3FE 0%, #F2F5FC 62%);
    box-shadow: 0 4px 18px rgba(23, 37, 84, 0.06), 0 0 16px rgba(124, 58, 237, 0.06);
}

/* Bulleted evidence rows ("Why Was This Flagged?" reasons, technical
   evidence lists) - small cyan/blue accent marker instead of a plain
   dot, for clearer visual hierarchy. Purely a list-marker style. */
div[data-testid="stMarkdownContainer"] ul { margin-top: 0.2rem; margin-bottom: 0.2rem; }
div[data-testid="stMarkdownContainer"] ul li {
    list-style: none;
    position: relative;
    padding-left: 1.3rem;
    margin-bottom: 0.35rem;
}
div[data-testid="stMarkdownContainer"] ul li::before {
    content: "";
    position: absolute;
    left: 0.15rem;
    top: 0.55rem;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-blue) 0%, var(--brand-cyan) 100%);
    box-shadow: 0 0 6px rgba(6, 182, 212, 0.55);
}

/* ---- Buttons ---- */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--brand-blue) 0%, var(--brand-purple) 100%);
    border-radius: 10px;
    border: none;
    box-shadow: 0 3px 10px rgba(37, 99, 235, 0.28), 0 0 16px rgba(124, 58, 237, 0.20);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.34), 0 0 22px rgba(124, 58, 237, 0.30);
    transform: translateY(-1px);
}
div.stButton > button[kind="secondary"] {
    border-radius: 10px;
    border: 1px solid #D9DEF5;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: var(--brand-blue);
    box-shadow: 0 0 12px rgba(37, 99, 235, 0.16);
}

/* ---- Radios (nav pills / Review & Insights sections / plain radios) ---- */
div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    border: 1px solid #E7EAF5;
    border-radius: 10px;
    padding: 4px 14px;
    background: #FAFBFF;
    transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--brand-blue);
    box-shadow: 0 0 10px rgba(37, 99, 235, 0.18);
}

/* Analyst Decision control - APPROVE / REJECT / MODIFY as premium,
   colored product buttons (a marker placed immediately before the
   st.radio("Decision", ...) call scopes this to ONLY that widget).
   Solid gradient fill + soft glow when selected, gentle hover lift
   always available. */
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label {
    padding: 9px 22px;
    border-radius: 10px;
    font-weight: 600;
    transition: box-shadow 0.18s ease, transform 0.18s ease, background 0.18s ease;
}
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:hover {
    transform: translateY(-1px);
}

div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(1) {
    background: #E7F9F1; border: 1.5px solid var(--brand-green);
}
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(1) p { color: var(--brand-green-dark); }
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(1):has(input:checked) {
    background: linear-gradient(135deg, var(--brand-green) 0%, var(--brand-cyan-dark) 130%);
    border-color: var(--brand-green);
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.32), 0 0 18px rgba(16, 185, 129, 0.38);
}
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(1):has(input:checked) p { color: #FFFFFF; }

div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(2) {
    background: #FDEBEB; border: 1.5px solid var(--brand-red);
}
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(2) p { color: var(--brand-red-dark); }
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(2):has(input:checked) {
    background: linear-gradient(135deg, var(--brand-red) 0%, var(--brand-red-dark) 130%);
    border-color: var(--brand-red);
    box-shadow: 0 2px 10px rgba(239, 68, 68, 0.32), 0 0 18px rgba(239, 68, 68, 0.38);
}
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(2):has(input:checked) p { color: #FFFFFF; }

div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(3) {
    background: #F1EAFC; border: 1.5px solid var(--brand-purple);
}
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(3) p { color: var(--brand-purple-dark); }
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(3):has(input:checked) {
    background: linear-gradient(135deg, var(--brand-purple) 0%, var(--brand-blue) 130%);
    border-color: var(--brand-purple);
    box-shadow: 0 2px 10px rgba(124, 58, 237, 0.32), 0 0 18px rgba(124, 58, 237, 0.38);
}
div.element-container:has(.marker-decision-radio) + div.element-container div[role="radiogroup"] > label:nth-of-type(3):has(input:checked) p { color: #FFFFFF; }

div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #E7EAF5;
    background: #FFFFFF;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Plain-English presentation layer. These map REAL, already-computed
# technical identifiers (rule-engine fact IDs, knowledge-base IDs) to
# simple wording for the main screen. They never change what fact fired,
# what rule matched, or which knowledge entry was retrieved - only how
# its label is displayed. The underlying technical text (FACT_DESCRIPTIONS,
# each knowledge entry's real title) always remains visible in a
# "Technical details" expander alongside the plain version.
# ---------------------------------------------------------------------
FACT_PLAIN_TEXT = {
    "anomalous_event": "This event's overall pattern looks statistically unusual compared to normal activity.",
    "high_failed_logins": "There were an unusually high number of failed login attempts.",
    "high_request_count": "This user made an unusually high number of requests.",
    "unusual_port": "Network traffic used an uncommon network port.",
    "high_data_transfer": "A large amount of data was transferred.",
    "off_hours_activity": "This activity happened outside normal working hours.",
    "location_mismatch": "The login location is different from where this user normally logs in.",
    "multiple_devices": "This user was active from an unusually high number of devices.",
    "privilege_escalation_attempt": "An attempt to gain higher account privileges was detected.",
    "network_scan_indicator": "Activity matching a network scan was detected.",
}

KB_PLAIN_TITLES = {
    "KB-001": "How to Handle Suspicious Login Attempts",
    "KB-002": "Unusual Network Activity Guide",
    "KB-003": "Suspicious Network Scanning Guide",
    "KB-004": "How to Handle Large Data Transfers",
    "KB-005": "Unusual Login Location Guide",
    "KB-006": "Unusual After-Hours Account Activity Guide",
    "KB-007": "Unusual Multi-Device Activity Guide",
    "KB-008": "How Analysts Review Security Cases",
    "KB-009": "Steps to Respond to a Security Incident",
    "KB-010": "How Security Activity Is Monitored",
}


# ---------------------------------------------------------------------
# Sidebar - branding + the two-area navigation. No development/phase
# UI, no per-page list - see the module docstring above.
# ---------------------------------------------------------------------
NAV_AREAS = ["Case Analysis", "Review & Insights"]

with st.sidebar:
    st.markdown("## 🛡️ CyberAnalyst AI")
    st.divider()

    selected_page = st.radio("Navigate", NAV_AREAS, index=0, label_visibility="collapsed")


# ---------------------------------------------------------------------
# Shared data loading (used by every page) - unchanged.
# ---------------------------------------------------------------------
try:
    logs_df = load_logs()
except FileNotFoundError as e:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.error(str(e))
    st.stop()

if "feedback_db_ready" not in st.session_state:
    st.session_state["feedback_db_ready"] = initialize_feedback_database()

if "audit_db_ready" not in st.session_state:
    st.session_state["audit_db_ready"] = initialize_audit_database()


# ---------------------------------------------------------------------
# Record the deterministic pipeline stages already computed for one
# incident into the audit trail. OBSERVATIONAL ONLY - every value passed
# in was already computed elsewhere; this function never runs the anomaly
# detector, the rule engine, confidence scoring, RAG retrieval, or the
# response planner itself, and it never changes anything it records.
# Unchanged from the prior UI.
# ---------------------------------------------------------------------
def _log_case_processing_audit(incident_id, inf, conf, plan=None, retrieval_result=None, genai_result=None) -> None:
    write_audit_event(
        incident_id,
        "INCIDENT_SELECTED",
        "INCIDENT_SELECTED_FOR_PROCESSING",
        incident_type=inf.incident_type,
        severity=inf.severity,
        descriptive_details=f"Incident {incident_id} selected/processed for review and explanation.",
        skip_if_exists=True,
    )
    write_audit_event(
        incident_id,
        "ANOMALY_DETECTION",
        "ANOMALY_DETECTION_RESULT",
        incident_type=inf.incident_type,
        severity=inf.severity,
        descriptive_details=(
            f"anomaly_status={inf.evidence.get('anomaly_status', 'NORMAL')}, "
            f"anomaly_score={float(inf.evidence.get('anomaly_score', 0.0) or 0.0):.4f}."
        ),
        skip_if_exists=True,
    )
    write_audit_event(
        incident_id,
        "RULE_CLASSIFICATION",
        "RULE_CLASSIFICATION_RESULT",
        incident_type=inf.incident_type,
        severity=inf.severity,
        relevant_rule_ids=[tr.rule_id for tr in inf.triggered_rules],
        descriptive_details=inf.reason,
        skip_if_exists=True,
    )
    write_audit_event(
        incident_id,
        "CONFIDENCE",
        "CONFIDENCE_RESULT",
        incident_type=inf.incident_type,
        severity=inf.severity,
        confidence_level=conf.confidence_level,
        confidence_score=conf.confidence_score,
        supporting_indicators=list(conf.supporting_indicators),
        human_review_recommended=conf.human_review_recommended,
        descriptive_details=conf.uncertainty_reason,
        skip_if_exists=True,
    )

    retrieved_ids, rag_details = None, ""
    if retrieval_result is not None:
        retrieved_ids = [item.knowledge_id for item in retrieval_result.items]
        rag_details = f"{len(retrieval_result.items)} knowledge source(s) retrieved for this incident."
    elif genai_result is not None and genai_result.retrieved_knowledge_count > 0:
        retrieved_ids = list(genai_result.retrieved_knowledge_ids)
        rag_details = f"{genai_result.retrieved_knowledge_count} knowledge source(s) were supplied to the AI analysis."
    if retrieved_ids is not None:
        write_audit_event(
            incident_id,
            "RAG_RETRIEVAL",
            "RAG_RETRIEVAL_RESULT",
            incident_type=inf.incident_type,
            severity=inf.severity,
            retrieved_knowledge_ids=retrieved_ids,
            descriptive_details=rag_details,
            skip_if_exists=True,
        )

    if genai_result is not None:
        write_audit_event(
            incident_id,
            "GENAI_ANALYSIS",
            "GENAI_ANALYSIS_RESULT",
            incident_type=inf.incident_type,
            severity=inf.severity,
            confidence_level=conf.confidence_level,
            confidence_score=conf.confidence_score,
            retrieved_knowledge_ids=list(genai_result.retrieved_knowledge_ids),
            ai_model_name=genai_result.model_used,
            descriptive_details=f"source={genai_result.source}, used_fallback={genai_result.used_fallback}.",
            skip_if_exists=True,
        )

    if plan is not None:
        write_audit_event(
            incident_id,
            "RESPONSE_PLAN",
            "RESPONSE_PLAN_RESULT",
            incident_type=inf.incident_type,
            severity=inf.severity,
            human_review_recommended=plan.review_required,
            descriptive_details=f"{plan.plan_title} ({plan.priority} priority, {len(plan.steps)} step(s)).",
            skip_if_exists=True,
        )


RESULT_TABLE_COLUMNS = DISPLAY_COLUMNS + ["anomaly_score", "anomaly_status"]


def _style_chart(fig):
    """Shared, minimal styling so every chart reads as one clean, light system."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Status",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#1F2430"),
    )
    return fig


def _get_pipeline_results(df: pd.DataFrame):
    """
    Shared helper reused by both Case Analysis and Review & Insights - the
    exact same Phase 1-4 chain every page in this app has always run
    (preprocessing -> anomaly detection -> rule engine -> confidence
    scoring), using the same st.session_state["anomaly_result"] caching
    key so results stay consistent across the whole app in one session.
    Returns (pre, anomaly_result, rule_result, confidence_result), or
    None if the dataset could not be prepared - callers must check for
    None and stop.
    """
    try:
        pre = run_preprocessing_pipeline(df)
    except Exception as exc:  # noqa: BLE001 - never crash the app
        st.error("Something went wrong while preparing the security logs.")
        st.exception(exc)
        return None
    if not pre.is_usable:
        st.error("The security log dataset could not be prepared for analysis.")
        for err in pre.validation.errors:
            st.error(err)
        return None

    anomaly_result = st.session_state.get("anomaly_result")
    if anomaly_result is None:
        try:
            anomaly_result = run_anomaly_detection(pre)
            st.session_state["anomaly_result"] = anomaly_result
        except Exception as exc:  # noqa: BLE001
            st.error("Anomaly detection could not be run on this dataset.")
            st.exception(exc)
            return None

    try:
        rule_result = run_rule_engine(pre, anomaly_result)
    except Exception as exc:  # noqa: BLE001 - never crash the app
        st.error("Something went wrong while classifying events.")
        st.exception(exc)
        return None
    if rule_result.results_df.empty:
        st.warning("No events are available to analyze in this dataset.")
        return None

    try:
        confidence_result = run_confidence_scoring(rule_result)
    except Exception as exc:  # noqa: BLE001 - never crash the app
        st.error("Something went wrong while scoring confidence.")
        st.exception(exc)
        return None

    return pre, anomaly_result, rule_result, confidence_result


# ---------------------------------------------------------------------
# Shared helpers - unchanged from the prior UI, reused verbatim by Case
# Analysis (the safety-critical decision logic in particular is copied
# here exactly, not rewritten).
# ---------------------------------------------------------------------
def _incident_universe(confidence_result) -> pd.DataFrame:
    """
    Every event that actually has a response plan (Normal Activity is
    excluded - the response planner never generates a plan for it, so
    there is nothing for an analyst to review), tagged with its stable,
    human-readable `incident_id` (see `src.feedback.build_incident_id`).
    """
    rows = []
    for idx, row in confidence_result.results_df.iterrows():
        inf = row["_inference"]
        if inf.incident_type == "Normal Activity":
            continue
        conf = row["_confidence"]
        rows.append(
            {
                "event_index": idx,
                "incident_id": build_incident_id(idx),
                "event_timestamp": row.get("timestamp", ""),
                "incident_type": inf.incident_type,
                "severity": inf.severity,
                "confidence_level": conf.confidence_level,
                "confidence_score": conf.confidence_score,
            }
        )
    return pd.DataFrame(rows)


def _latest_feedback_per_incident(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """The single most recent feedback record per incident_id, indexed by
    incident_id - used to derive each incident's current review status and
    when it was last reviewed."""
    if feedback_df.empty:
        return feedback_df
    return (
        feedback_df.sort_values("timestamp", ascending=False)
        .drop_duplicates("incident_id", keep="first")
        .set_index("incident_id")
    )


def _render_plan_editor(incident_id: str, plan: ResponsePlanResult) -> ResponsePlanResult:
    """
    The safe MODIFY interface: lets the analyst edit a step's wording/
    priority/review flag, remove a step, reorder steps (via an editable
    "Order #"), and add a new investigative step - using only plain,
    individually-testable Streamlit widgets. `incident_type`, `severity`,
    and `confidence` are never shown as editable here at all - they are
    simply not part of this widget set, so there is no way to change them
    from this screen.

    Returns a CANDIDATE modified `ResponsePlanResult`, already re-validated
    with the exact same `src.response_planner.validate_plan()` the system
    itself uses - the original, system-generated `plan` object passed in
    is never mutated.
    """
    steps_key = f"modify_steps_{incident_id}"
    counter_key = f"modify_uid_counter_{incident_id}"

    if steps_key not in st.session_state:
        st.session_state[steps_key] = [
            {
                "uid": i,
                "action": s.action,
                "purpose": s.purpose,
                "reason": s.reason,
                "priority": s.priority,
                "requires_human_review": s.requires_human_review,
                "stage": s.stage,
                "order": i + 1,
                "remove": False,
            }
            for i, s in enumerate(plan.steps)
        ]
        st.session_state[counter_key] = len(plan.steps)

    st.caption(
        "Edit a step's action/reason/priority/review flag below, check "
        "'Remove' to drop a step, or change 'Order #' to reorder (steps "
        "are renumbered automatically before saving). `incident_type`, "
        "`severity`, and `confidence` are never editable here - they stay "
        "exactly as computed by the system."
    )
    if st.button("↺ Reset to system-generated plan", key=f"reset_{incident_id}"):
        del st.session_state[steps_key]
        del st.session_state[counter_key]
        st.rerun()

    steps_state = st.session_state[steps_key]
    for step in steps_state:
        uid = step["uid"]
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            step["action"] = c1.text_input("Action", value=step["action"], key=f"action_{incident_id}_{uid}")
            step["order"] = c2.number_input(
                "Order #", min_value=1, value=int(step["order"]), step=1, key=f"order_{incident_id}_{uid}"
            )
            step["reason"] = st.text_input("Reason", value=step["reason"], key=f"reason_{incident_id}_{uid}")
            c3, c4, c5 = st.columns(3)
            current_priority = step["priority"] if step["priority"] in PLAN_PRIORITY_LEVELS else PLAN_PRIORITY_LEVELS[0]
            step["priority"] = c3.selectbox(
                "Priority", PLAN_PRIORITY_LEVELS, index=PLAN_PRIORITY_LEVELS.index(current_priority),
                key=f"priority_{incident_id}_{uid}",
            )
            step["requires_human_review"] = c4.checkbox(
                "Requires human review", value=step["requires_human_review"], key=f"review_{incident_id}_{uid}"
            )
            step["remove"] = c5.checkbox(
                "Remove this step", value=step.get("remove", False), key=f"remove_{incident_id}_{uid}"
            )
            st.caption(f"Purpose (from the system plan): {step['purpose']}")

    with st.expander("➕ Add a defensive investigation step"):
        new_action = st.text_input("New step - action", key=f"new_action_{incident_id}")
        new_reason = st.text_input("New step - reason", key=f"new_reason_{incident_id}")
        new_priority = st.selectbox("New step - priority", PLAN_PRIORITY_LEVELS, key=f"new_priority_{incident_id}")
        new_review = st.checkbox("New step requires human review", key=f"new_review_{incident_id}")
        if st.button("Add Step", key=f"add_step_{incident_id}"):
            if new_action.strip():
                new_uid = st.session_state[counter_key]
                st.session_state[counter_key] += 1
                steps_state.append(
                    {
                        "uid": new_uid,
                        "action": new_action,
                        "purpose": "Analyst-added investigative step.",
                        "reason": new_reason,
                        "priority": new_priority,
                        "requires_human_review": new_review,
                        "stage": "TRIAGED",
                        "order": max((s["order"] for s in steps_state), default=0) + 1,
                        "remove": False,
                    }
                )
                st.rerun()
            else:
                st.warning("Enter an action before adding a step.")

    # Build the candidate modified plan from the current widget state -
    # kept steps, sorted by the analyst's chosen order, renumbered 1..N.
    kept = sorted((s for s in steps_state if not s.get("remove")), key=lambda s: s["order"])
    new_steps = [
        PlanStep(
            step_number=i + 1,
            action=s["action"],
            purpose=s["purpose"],
            reason=s["reason"],
            priority=s["priority"],
            requires_human_review=s["requires_human_review"],
            stage=s["stage"],
        )
        for i, s in enumerate(kept)
    ]
    modified_plan = dataclasses.replace(
        plan, steps=new_steps, review_required=any(s.requires_human_review for s in new_steps)
    )
    modified_plan.validation_status = validate_plan(modified_plan)  # reuse the system's own validator - never duplicated

    st.markdown("##### Live Preview — Modified Plan")
    if new_steps:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": s.step_number, "Action": s.action, "Reason": s.reason,
                        "Priority": s.priority, "Human Review": "Required" if s.requires_human_review else "—",
                    }
                    for s in new_steps
                ]
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.warning("This modified plan has no steps left - add at least one before submitting.")

    if modified_plan.validation_status.is_valid:
        st.success("✅ Modified plan passes validation.")
    else:
        st.error("⚠️ Modified plan FAILS validation - fix the issues below before submitting. It will not be saved as-is:")
        for err in modified_plan.validation_status.errors:
            st.markdown(f"- {err}")

    return modified_plan


# ---------------------------------------------------------------------
# Page: Case Analysis - the main screen. Select a case, then one
# continuous flow: case details -> AI case summary -> why it was flagged
# -> knowledge base used by the AI -> recommended next steps -> the
# analyst's decision. Consolidates what used to be seven separate pages
# (Incident Classification, Uncertainty & Confidence, Generative AI
# Analysis, Response Planning, Analyst Review, plus the RAG section that
# used to live on Generative AI Analysis) into one case-centered
# workflow. Every function called below is the exact same function the
# old pages called - only the presentation is new.
# ---------------------------------------------------------------------
def render_case_analysis_page(df: pd.DataFrame) -> None:
    st.title("Case Analysis")
    st.caption("Select a security case to review its AI-assisted analysis and recommended next steps.")

    pipeline = _get_pipeline_results(df)
    if pipeline is None:
        return
    pre, anomaly_result, rule_result, confidence_result = pipeline

    incidents_df = _incident_universe(confidence_result)
    if incidents_df.empty:
        st.info("No security cases need review right now — every event in this dataset looks like normal activity.")
        return

    feedback_df = load_feedback()
    status_map = get_review_status_map(feedback_df)
    incidents_df = incidents_df.copy()
    incidents_df["review_status"] = incidents_df["incident_id"].map(status_map).fillna(REVIEW_STATUS_PENDING)

    # --- Select Security Case ---
    st.subheader("Select Security Case")
    options = {
        row["incident_id"]: (
            f"{row['incident_id']} — {row['incident_type']} ({row['severity']}) — "
            f"{row['review_status'].title()}"
        )
        for _, row in incidents_df.iterrows()
    }
    option_ids = list(options.keys())
    default_incident = st.session_state.get("last_selected_incident_id")
    default_index = option_ids.index(default_incident) if default_incident in option_ids else 0
    selected_incident_id = st.selectbox(
        "Security case", options=option_ids, index=default_index, format_func=lambda i: options[i],
        label_visibility="collapsed",
    )
    st.session_state["last_selected_incident_id"] = selected_incident_id

    event_index = int(incidents_df.loc[incidents_df["incident_id"] == selected_incident_id, "event_index"].iloc[0])
    inf = confidence_result.results_df.loc[event_index, "_inference"]
    conf = confidence_result.results_df.loc[event_index, "_confidence"]
    review_status = str(incidents_df.loc[incidents_df["incident_id"] == selected_incident_id, "review_status"].iloc[0])

    st.divider()

    # --- Four info cards ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="status-badge badge-blue">🧩</div>', unsafe_allow_html=True)
        st.metric("Incident Type", inf.incident_type)
    with c2:
        st.markdown('<div class="status-badge badge-orange">⚠️</div>', unsafe_allow_html=True)
        st.metric("Severity", inf.severity.title() if inf.severity != "N/A" else "N/A")
    with c3:
        st.markdown('<div class="status-badge badge-green">📊</div>', unsafe_allow_html=True)
        st.metric(
            "Confidence", f"{conf.confidence_score:.0%}",
            help=f"{conf.confidence_level.title()} confidence level. This is not a probability.",
        )
    with c4:
        st.markdown('<div class="status-badge badge-purple">🗂️</div>', unsafe_allow_html=True)
        st.metric("Review Status", review_status.title())

    st.divider()

    # --- AI Case Summary (shown once) ---
    st.markdown('<div class="marker-ai-heading"></div>', unsafe_allow_html=True)
    st.subheader("AI Case Summary")

    retrieval_result = retrieve_knowledge_for_incident(inf, conf)
    st.session_state.setdefault("retrieval_results", {})[event_index] = retrieval_result

    with st.expander("⚙️ AI model settings", expanded=False):
        model_name = st.text_input(
            "Local AI model name",
            value=st.session_state.get("genai_model_name", OLLAMA_MODEL_NAME),
            help="Change this if a different local Ollama model is installed. No source-code edits needed.",
        )
        st.session_state["genai_model_name"] = model_name
        try:
            ollama_status = get_ollama_status(model_name)
            icon = "✅" if (ollama_status.ollama_available and ollama_status.model_available) else "ℹ️"
            st.caption(f"{icon} {ollama_status.message}")
        except Exception:  # noqa: BLE001 - a status check must never break the page
            pass
    model_name = st.session_state.get("genai_model_name", OLLAMA_MODEL_NAME)

    genai_cache = st.session_state.setdefault("genai_results", {})
    cache_key = (event_index, model_name)
    cached_genai = genai_cache.get(cache_key)

    if cached_genai is None:
        if st.button("🧠 Generate AI Case Summary", type="primary", key=f"gen_summary_{selected_incident_id}"):
            context = build_incident_context(
                inf, conf,
                retrieved_knowledge=[item.content for item in retrieval_result.items],
                retrieved_knowledge_items=retrieved_items_to_dicts(retrieval_result.items),
            )
            try:
                genai_cache[cache_key] = generate_ai_case_summary(context, model_name=model_name)
                cached_genai = genai_cache.get(cache_key)
            except Exception as exc:  # noqa: BLE001 - the module itself should never raise, but stay safe
                st.error("Something went wrong while generating the AI case summary.")
                st.exception(exc)

    if cached_genai is None:
        st.info(
            "Click **Generate AI Case Summary** to produce an analyst-style explanation for this case.",
            icon="👆",
        )
    else:
        st.markdown('<div class="marker-ai-summary"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="marker-ai-badge"></div>', unsafe_allow_html=True)
            if cached_genai.used_fallback:
                st.caption(
                    "⚠️ Automatic summary — the local AI model was unavailable, "
                    "so a rule-based summary was used instead."
                )
            else:
                st.caption(f"✅ Generated by the local AI model ({cached_genai.model_used}).")
            st.write(cached_genai.summary)
        if cached_genai.retrieved_knowledge_count > 0:
            st.caption(
                "📚 Informed by: "
                + ", ".join(
                    KB_PLAIN_TITLES.get(kid, title)
                    for kid, title in zip(cached_genai.retrieved_knowledge_ids, cached_genai.retrieved_knowledge_titles)
                )
            )

    st.divider()

    # --- Why Was This Flagged? ---
    st.markdown('<div class="marker-flagged-heading"></div>', unsafe_allow_html=True)
    st.subheader("Why Was This Flagged?")
    plain_reasons = [FACT_PLAIN_TEXT.get(fact, FACT_DESCRIPTIONS.get(fact, fact)) for fact in inf.derived_facts]
    if plain_reasons:
        for reason in plain_reasons:
            st.markdown(f"- {reason}")
    else:
        st.caption("No specific risk indicators were recorded beyond the anomaly flag itself.")

    if cached_genai is not None and cached_genai.why_suspicious:
        st.write(cached_genai.why_suspicious)

    with st.expander("🔧 Technical details"):
        st.markdown("**Observed evidence**")
        st.json(inf.evidence, expanded=False)
        st.markdown("**Derived facts**")
        if inf.derived_facts:
            for fact in inf.derived_facts:
                st.markdown(f"- **{fact}** — {FACT_DESCRIPTIONS.get(fact, '')}")
        else:
            st.caption("None.")
        st.markdown("**Rules triggered**")
        if inf.triggered_rules:
            for tr in inf.triggered_rules:
                st.markdown(f"- **{tr.rule_id}** ({tr.name}): {tr.explanation}")
        else:
            st.caption("No rules matched this event.")
        st.markdown("**Anomaly score (Isolation Forest)**")
        st.write(
            f"{float(inf.evidence.get('anomaly_score', 0.0) or 0.0):.4f} "
            "(higher = more unusual; this is not a probability)"
        )
        st.markdown("**Confidence evidence breakdown**")
        breakdown_df = pd.DataFrame(
            [
                {
                    "Component": name.replace("_", " ").title(),
                    "Value (0-1)": comp.value, "Weight": comp.weight, "Contribution": comp.contribution,
                }
                for name, comp in conf.components.items()
            ]
        )
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        for reason in conf.confidence_reasons:
            st.markdown(f"- {reason}")
        st.markdown("**Uncertainty / limitations**")
        st.write(conf.uncertainty_reason)
        if cached_genai is not None:
            st.markdown("**AI severity explanation**")
            st.write(cached_genai.severity_explanation)
            st.markdown("**AI confidence explanation**")
            st.write(cached_genai.confidence_explanation)
            st.markdown("**AI supporting evidence**")
            st.write(cached_genai.supporting_evidence)
            st.markdown("**AI recommendations (from the generative-AI summary)**")
            for rec in cached_genai.recommendations:
                st.markdown(f"- {rec}")
            if cached_genai.warnings:
                st.markdown("**Notes / warnings about the AI result**")
                for w in cached_genai.warnings:
                    st.caption(f"- {w}")
            st.markdown("**Raw AI response (debugging)**")
            st.code(cached_genai.raw_response or "(empty — a deterministic fallback was used)")

    st.divider()

    # --- Knowledge Base Used by AI ---
    st.markdown('<div class="marker-kb-heading"></div>', unsafe_allow_html=True)
    st.subheader("Knowledge Base Used by AI")
    st.caption(
        "The AI used the following security information to understand this case "
        "and prepare its summary and recommendation."
    )
    if not retrieval_result.items:
        st.info("No closely related reference material was found for this case.", icon="ℹ️")
    else:
        for item in retrieval_result.items:
            plain_title = KB_PLAIN_TITLES.get(item.knowledge_id, item.title)
            st.markdown('<div class="marker-kb-item"></div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"**{plain_title}**")
                excerpt = item.content if len(item.content) <= 260 else item.content[:260].rsplit(" ", 1)[0] + "…"
                st.write(excerpt)
                with st.expander("Read full guidance"):
                    st.write(item.content)
                    st.caption(
                        f"Source: {item.knowledge_id} · {item.title} · Topic: {item.topic} · "
                        f"Relevance score: {item.similarity_score:.2f}"
                    )

    st.divider()

    # --- Recommended Next Steps ---
    st.markdown('<div class="marker-next-steps-heading"></div>', unsafe_allow_html=True)
    st.subheader("Recommended Next Steps")
    st.warning(
        "These steps are advisory only. The system recommends a defensive response - "
        "it does not execute any action against a real system.",
        icon="🛡️",
    )

    plan = generate_response_plan(inf, conf, genai_result=cached_genai)
    st.session_state.setdefault("response_plans", {})[event_index] = plan
    _log_case_processing_audit(
        selected_incident_id, inf, conf,
        plan=plan, retrieval_result=retrieval_result, genai_result=cached_genai,
    )

    st.markdown(f"**{plan.plan_title}**")
    pc1, pc2 = st.columns(2)
    pc1.metric("Priority", plan.priority)
    pc2.metric("Human Review", "Required" if plan.review_required else "Not required")
    st.write(plan.plan_summary)

    if plan.steps:
        steps_df = pd.DataFrame(
            [
                {
                    "#": s.step_number, "Action": s.action, "Why": s.reason,
                    "Priority": s.priority, "Human Review": "Required" if s.requires_human_review else "—",
                }
                for s in plan.steps
            ]
        )
        st.dataframe(steps_df, use_container_width=True, hide_index=True)
    else:
        st.info("No response steps are needed for this case.", icon="✅")

    if not plan.validation_status.is_valid:
        st.error("This plan needs attention before it can be used:")
        for err in plan.validation_status.errors:
            st.markdown(f"- {err}")

    with st.expander("🔧 Technical plan details"):
        st.write(plan.planning_reason)
        st.caption(PLAN_ORDERING_RATIONALE)
        if plan.steps:
            full_steps_df = pd.DataFrame(
                [
                    {
                        "#": s.step_number, "Action": s.action, "Purpose": s.purpose, "Reason": s.reason,
                        "Priority": s.priority, "Human Review": "Required" if s.requires_human_review else "—",
                        "Stage": PLAN_STAGE_LABELS.get(s.stage, s.stage),
                    }
                    for s in plan.steps
                ]
            )
            st.dataframe(full_steps_df, use_container_width=True, hide_index=True)
        if plan.validation_status.warnings:
            st.markdown("**Validation warnings**")
            for w in plan.validation_status.warnings:
                st.markdown(f"- {w}")
        st.markdown("**Plan flow stages**")
        stage_cols = st.columns(len(PLAN_STAGE_NAMES))
        for col, stage_name in zip(stage_cols, PLAN_STAGE_NAMES):
            label = PLAN_STAGE_LABELS[stage_name]
            included = stage_name in plan.included_stages
            with col:
                st.markdown(f"**{'✅' if included else '⬜'}**")
                st.caption(label)
        if plan.supplementary_ai_context:
            st.markdown("**Supplementary AI context** (advisory only, never used to build this plan)")
            for rec in plan.supplementary_ai_context:
                st.markdown(f"- {rec}")

    st.divider()

    # --- Past reviews (context ahead of a new decision) ---
    past_reviews = get_feedback_by_incident(selected_incident_id)
    if not past_reviews.empty:
        with st.expander(f"📜 Past reviews for this case ({len(past_reviews)})"):
            st.dataframe(
                past_reviews[["analyst_decision", "review_status", "analyst_notes", "timestamp"]].rename(
                    columns={
                        "analyst_decision": "Decision", "review_status": "Status",
                        "analyst_notes": "Notes", "timestamp": "Timestamp",
                    }
                ),
                use_container_width=True, hide_index=True,
            )

    # --- Analyst Decision ---
    # Safety-critical: reused exactly from the prior UI. incident_type,
    # severity, and confidence are never editable inputs anywhere in this
    # section - they are only ever displayed. original_plan is always
    # preserved; modified_plan is only ever set for a MODIFY decision.
    st.subheader("Analyst Decision")
    st.info("**Your decision is the final decision for this case.**", icon="✅")
    st.markdown('<div class="marker-decision-radio"></div>', unsafe_allow_html=True)
    decision = st.radio(
        "Decision", ANALYST_DECISIONS, key=f"decision_{selected_incident_id}", horizontal=True
    )
    notes = st.text_area(
        "Analyst Notes",
        key=f"notes_{selected_incident_id}",
        placeholder="e.g. Reviewed with authentication logs; activity appears expected.",
    )

    modified_plan = None
    if decision == "MODIFY":
        st.markdown("#### Modify the Response Plan")
        modified_plan = _render_plan_editor(selected_incident_id, plan)

    submit_blocked = decision == "MODIFY" and (
        modified_plan is None or not modified_plan.steps or not modified_plan.validation_status.is_valid
    )
    if st.button("✅ Submit Decision", type="primary", disabled=submit_blocked, key=f"submit_{selected_incident_id}"):
        record = FeedbackRecord(
            feedback_id=generate_feedback_id(selected_incident_id),
            incident_id=selected_incident_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            incident_type=inf.incident_type,
            severity=inf.severity,
            confidence_level=conf.confidence_level,
            confidence_score=conf.confidence_score,
            human_review_recommended=conf.human_review_recommended,
            analyst_decision=decision,
            review_status=DECISION_TO_REVIEW_STATUS[decision],
            original_plan=plan_to_dict(plan),
            modified_plan=plan_to_dict(modified_plan) if decision == "MODIFY" else None,
            analyst_notes=notes or "",
            triggered_rule_ids=[tr.rule_id for tr in inf.triggered_rules],
            supporting_indicators=list(conf.supporting_indicators),
            ai_model_name=cached_genai.model_used if cached_genai else None,
            ai_summary_source=cached_genai.source if cached_genai else None,
        )
        success, errors = save_feedback(record)
        if success:
            write_audit_event(
                selected_incident_id,
                "ANALYST_REVIEW",
                "FEEDBACK_DECISION",
                incident_type=inf.incident_type,
                severity=inf.severity,
                confidence_level=conf.confidence_level,
                confidence_score=conf.confidence_score,
                human_review_recommended=conf.human_review_recommended,
                analyst_decision=decision,
                descriptive_details=f"Analyst recorded {decision} (feedback_id={record.feedback_id}).",
                skip_if_exists=False,
            )
            st.success(f"Decision saved: **{record.review_status}**.")
            st.caption(f"Feedback ID: `{record.feedback_id}` · Saved at {record.timestamp} (UTC)")
            for k in (f"modify_steps_{selected_incident_id}", f"modify_uid_counter_{selected_incident_id}"):
                st.session_state.pop(k, None)
            with st.expander("View saved decision record"):
                st.json(dataclasses.asdict(record), expanded=False)
        else:
            st.error("The decision could not be saved:")
            for err in errors:
                st.markdown(f"- {err}")


# ---------------------------------------------------------------------
# Page: Review & Insights - Case History, Analyst Decisions, Feedback,
# Continuous Improvement, Explainability, Audit Trail, and Evaluation /
# Performance, as tabs on one page. Every tab below reuses the exact
# same functions the old, separate pages called - content only moved,
# nothing was rewritten or recomputed differently.
# ---------------------------------------------------------------------
def _render_case_history_tab(incidents_df: pd.DataFrame, feedback_df: pd.DataFrame) -> None:
    st.subheader("Case History")
    if incidents_df.empty:
        st.info("No security cases have been classified yet.")
        return

    status_map = get_review_status_map(feedback_df)
    latest = _latest_feedback_per_incident(feedback_df)
    incidents_df = incidents_df.copy()
    incidents_df["review_status"] = incidents_df["incident_id"].map(status_map).fillna(REVIEW_STATUS_PENDING)
    incidents_df["reviewed_at"] = (
        incidents_df["incident_id"].map(latest["timestamp"]) if not latest.empty else "—"
    )
    incidents_df["reviewed_at"] = incidents_df["reviewed_at"].fillna("—")

    status_counts = incidents_df["review_status"].value_counts()
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Cases", len(incidents_df))
    d2.metric("Pending", int(status_counts.get(REVIEW_STATUS_PENDING, 0)))
    d3.metric("Approved", int(status_counts.get("APPROVED", 0)))
    d4.metric("Rejected", int(status_counts.get("REJECTED", 0)))
    d5.metric("Modified", int(status_counts.get("MODIFIED", 0)))

    f1, f2, f3 = st.columns(3)
    with f1:
        status_filter = st.multiselect("Review Status", REVIEW_STATUSES, default=REVIEW_STATUSES, key="ch_status_filter")
    with f2:
        severity_options = sorted(incidents_df["severity"].unique().tolist())
        severity_filter = st.multiselect("Severity", severity_options, default=severity_options, key="ch_severity_filter")
    with f3:
        type_options = sorted(incidents_df["incident_type"].unique().tolist())
        type_filter = st.multiselect("Incident Type", type_options, default=type_options, key="ch_type_filter")

    filtered = incidents_df[
        incidents_df["review_status"].isin(status_filter)
        & incidents_df["severity"].isin(severity_filter)
        & incidents_df["incident_type"].isin(type_filter)
    ]
    st.dataframe(
        filtered[
            ["incident_id", "incident_type", "severity", "confidence_level", "review_status", "reviewed_at"]
        ].rename(
            columns={
                "incident_id": "Case ID", "incident_type": "Incident Type", "severity": "Severity",
                "confidence_level": "Confidence", "review_status": "Review Status", "reviewed_at": "Last Reviewed",
            }
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Showing {len(filtered)} of {len(incidents_df)} cases.")


def _render_analyst_decisions_tab(feedback_df: pd.DataFrame) -> None:
    st.subheader("Analyst Decisions")
    if feedback_df.empty:
        st.info("No analyst decisions have been recorded yet. Visit **Case Analysis** to review a case.")
        return

    st.dataframe(
        feedback_df[
            ["incident_id", "analyst_decision", "review_status", "severity", "confidence_level", "analyst_notes", "timestamp"]
        ].rename(
            columns={
                "incident_id": "Case", "analyst_decision": "Decision", "review_status": "Review Status",
                "severity": "Severity", "confidence_level": "Confidence", "analyst_notes": "Notes",
                "timestamp": "Timestamp",
            }
        ),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.markdown("#### Inspect a Decision")
    options = {
        row["feedback_id"]: f"[{row['incident_id']}] {row['analyst_decision']} · {row['timestamp']}"
        for _, row in feedback_df.iterrows()
    }
    selected_fb_id = st.selectbox("Decision record", options=list(options.keys()), format_func=lambda i: options[i])
    record_row = feedback_df[feedback_df["feedback_id"] == selected_fb_id].iloc[0]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Case", record_row["incident_id"])
    r2.metric("Decision", record_row["analyst_decision"])
    r3.metric("Severity", record_row["severity"])
    r4.metric("Confidence", record_row["confidence_level"])

    st.markdown("**Analyst Notes**")
    st.write(record_row["analyst_notes"] or "_(no notes entered)_")

    def _steps_table(plan_dict):
        steps = (plan_dict or {}).get("steps") or []
        if not steps:
            return None
        return pd.DataFrame(
            [
                {
                    "#": s["step_number"], "Action": s["action"], "Priority": s["priority"],
                    "Human Review": "Required" if s["requires_human_review"] else "—",
                }
                for s in steps
            ]
        )

    st.markdown("**Original Response Plan** (system-generated, always preserved unmodified)")
    original_table = _steps_table(record_row["original_plan"])
    if original_table is not None:
        st.dataframe(original_table, use_container_width=True, hide_index=True)
    else:
        st.caption("No response steps were part of the original plan.")

    if record_row["analyst_decision"] == "MODIFY" and record_row["modified_plan"]:
        st.markdown("**Analyst-Modified Response Plan**")
        modified_table = _steps_table(record_row["modified_plan"])
        if modified_table is not None:
            st.dataframe(modified_table, use_container_width=True, hide_index=True)
        st.caption(
            "This decision's final plan was analyst-modified — it replaced the "
            "system-generated plan above for this decision only. The original "
            "system plan is preserved above, unmodified, for comparison."
        )


def _render_feedback_insights_tab(feedback_df: pd.DataFrame) -> None:
    st.subheader("Feedback")
    if feedback_df.empty:
        st.info("No feedback has been recorded yet. Visit **Case Analysis** to review a case.")
        return

    summary = summarize_feedback(feedback_df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Reviews", summary.total_reviews)
    m2.metric("Approved", summary.approved_count, f"{summary.approval_rate:.0%} approval rate")
    m3.metric("Rejected", summary.rejected_count, f"{summary.rejection_rate:.0%} rejection rate")
    m4.metric("Modified", summary.modified_count, f"{summary.modification_rate:.0%} modification rate")
    m5.metric("Incident Types Reviewed", len(summary.incident_type_counts))

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Most common incident types reviewed**")
        if summary.incident_type_counts:
            st.dataframe(
                pd.DataFrame(
                    sorted(summary.incident_type_counts.items(), key=lambda kv: -kv[1]),
                    columns=["Incident Type", "Reviews"],
                ),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No data yet.")
    with ic2:
        st.markdown("**Common analyst-note keywords**")
        if summary.note_keyword_counts:
            st.dataframe(
                pd.DataFrame(list(summary.note_keyword_counts.items()), columns=["Keyword", "Occurrences"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No repeated keywords found in analyst notes yet.")


def _render_continuous_improvement_tab(feedback_df: pd.DataFrame) -> None:
    st.subheader("Continuous Improvement")
    st.caption(
        "Offline, human-supervised analysis of analyst feedback: Analyst Feedback → "
        "Pattern Detection → Improvement Suggestion → Human/Developer Review → Manual Future Change."
    )
    st.warning(
        "**These insights are recommendations for future system improvement. "
        "They do not automatically modify the system.** No threshold, rule, model, "
        "RAG configuration, or response plan is changed by this tab — a developer "
        "reviews these suggestions and decides, by hand, whether to act on them.",
        icon="🛠️",
    )
    if feedback_df.empty:
        st.info("No feedback has been recorded yet, so there is nothing to analyze.")
        return

    ci_report = generate_continuous_improvement_report(feedback_df)

    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric("Total Feedback Records", ci_report.total_feedback_records)
    ci2.metric("Approval Rate", f"{ci_report.approval_rate:.0%}", f"{ci_report.approved_count} approved")
    ci3.metric("Rejection Rate", f"{ci_report.rejection_rate:.0%}", f"{ci_report.rejected_count} rejected")
    ci4.metric("Modification Rate", f"{ci_report.modification_rate:.0%}", f"{ci_report.modified_count} modified")

    cic1, cic2 = st.columns(2)
    with cic1:
        st.markdown("**Most reviewed incident types**")
        if ci_report.most_reviewed_incident_types:
            st.dataframe(
                pd.DataFrame(ci_report.most_reviewed_incident_types, columns=["Incident Type", "Reviews"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No data yet.")
    with cic2:
        st.markdown("**Incident types with repeated REJECT/MODIFY (disagreement)**")
        disagreement_rows = [
            {
                "Incident Type": b.incident_type, "Reject": b.reject_count, "Modify": b.modify_count,
                "Disagreement Rate": f"{b.disagreement_rate:.0%}",
            }
            for b in ci_report.incident_type_breakdowns
            if b.disagreement_count >= 1
        ]
        if disagreement_rows:
            st.dataframe(pd.DataFrame(disagreement_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No REJECT/MODIFY decisions recorded yet.")

    st.markdown("**Frequently modified response-plan steps**")
    if ci_report.frequent_step_modifications:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Incident Type": f.incident_type, "Original Step Wording": f.original_action,
                        "Times Modified": f.occurrence_count, "Supporting Incidents": ", ".join(f.supporting_incident_ids),
                    }
                    for f in ci_report.frequent_step_modifications
                ]
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption(
            f"No response-plan step has been modified in {CI_MIN_PATTERN_OCCURRENCES}+ "
            "separate reviews yet."
        )

    st.markdown("**Common analyst-note themes**")
    if ci_report.note_themes:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Theme": t.keyword, "Occurrences": t.occurrence_count,
                        "Supporting Incidents": ", ".join(t.supporting_incident_ids),
                    }
                    for t in ci_report.note_themes
                ]
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("No recurring analyst-note theme found yet.")

    st.markdown("**Suggested Improvements**")
    if not ci_report.suggestions:
        st.caption(
            "No improvement suggestions yet - patterns need to repeat across "
            f"at least {CI_MIN_PATTERN_OCCURRENCES} separate feedback "
            "records before a suggestion is generated."
        )
    else:
        for suggestion in ci_report.suggestions:
            with st.container(border=True):
                st.markdown(f"**{suggestion.suggestion_id} · {suggestion.title}**")
                st.write(suggestion.description)
                st.caption(
                    "Supporting incidents: " + ", ".join(suggestion.supporting_incident_ids)
                    + " · Supporting feedback IDs: " + ", ".join(suggestion.supporting_feedback_ids)
                )


def _render_explainability_tab(confidence_result) -> None:
    st.subheader("Explainability")
    st.caption("See why the system reached its conclusion for one case — the full evidence chain.")
    st.warning(
        "**This tab only explains and records what already happened. It cannot "
        "change any anomaly result, classification, severity, confidence score, "
        "RAG retrieval, GenAI output, response plan, or analyst decision.**",
        icon="🔒",
    )

    results_df = confidence_result.results_df
    options = {
        idx: (
            f"[{build_incident_id(idx)}] {row['timestamp']} · {row['incident_type']} "
            f"({row['severity']}) · confidence {row['confidence_score']:.2f} ({row['confidence_level']})"
        )
        for idx, row in results_df.iterrows()
    }
    option_idx = list(options.keys())
    default_incident = st.session_state.get("last_selected_incident_id")
    default_index = 0
    if default_incident:
        for i, idx in enumerate(option_idx):
            if build_incident_id(idx) == default_incident:
                default_index = i
                break
    selected_idx = st.selectbox(
        "Case", options=option_idx, index=default_index, format_func=lambda i: options[i], key="explain_case_select"
    )
    incident_id = build_incident_id(selected_idx)
    inf = results_df.loc[selected_idx, "_inference"]
    conf = results_df.loc[selected_idx, "_confidence"]

    st.markdown("#### Incident Overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Case ID", incident_id)
    o2.metric("Incident Type", inf.incident_type)
    o3.metric("Severity", inf.severity)
    o4.metric("Confidence", f"{conf.confidence_score:.2f} ({conf.confidence_level})")

    model_name = st.session_state.get("genai_model_name", OLLAMA_MODEL_NAME)
    genai_cache = st.session_state.get("genai_results", {})
    cached_genai = genai_cache.get((selected_idx, model_name))
    cached_retrieval = st.session_state.get("retrieval_results", {}).get(selected_idx)
    cached_plan = st.session_state.get("response_plans", {}).get(selected_idx)

    incident_feedback = get_feedback_by_incident(incident_id)
    latest_feedback_row = incident_feedback.iloc[0].to_dict() if not incident_feedback.empty else None

    explanation = build_incident_explanation(
        incident_id, inf, conf,
        retrieval_result=cached_retrieval, genai_result=cached_genai,
        response_plan=cached_plan, latest_feedback_row=latest_feedback_row,
    )

    _log_case_processing_audit(
        incident_id, inf, conf, plan=cached_plan, retrieval_result=cached_retrieval, genai_result=cached_genai
    )

    st.divider()
    st.markdown("#### Anomaly Detection")
    st.write(explanation.anomaly.explanation)
    a1, a2 = st.columns(2)
    a1.metric("Status", explanation.anomaly.anomaly_status)
    a2.metric("Anomaly Score", f"{explanation.anomaly.anomaly_score:.4f}")

    st.markdown("#### Rule / Knowledge-Based Explanation")
    st.write(explanation.classification.reason)
    if explanation.classification.triggered_rules:
        for r in explanation.classification.triggered_rules:
            st.markdown(f"- **{r.rule_id}** ({r.name}): {r.explanation}")
    else:
        st.caption("No rules matched this event.")

    st.markdown("#### Severity + Confidence Explanation")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Severity**")
        st.write(explanation.severity.explanation)
    with sc2:
        st.markdown("**Confidence**")
        st.write(explanation.confidence.summary)
    with st.expander("Confidence components (weighted evidence breakdown)"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Component": name, "Value": c["value"], "Weight": c["weight"], "Contribution": c["contribution"]}
                    for name, c in explanation.confidence.components.items()
                ]
            ),
            use_container_width=True, hide_index=True,
        )
    st.markdown("**Supporting Indicators**")
    st.write(explanation.supporting_indicators.indicators or "None recorded.")
    st.markdown("**Uncertainty**")
    st.write(explanation.uncertainty.uncertainty_reason)
    st.caption(
        f"Human review recommended: {'Yes' if explanation.uncertainty.human_review_recommended else 'No'}"
    )

    st.markdown("#### RAG Sources")
    st.write(explanation.rag.summary)
    if explanation.rag.items:
        st.dataframe(pd.DataFrame(explanation.rag.items), use_container_width=True, hide_index=True)
    if not explanation.rag.ran:
        st.info("Visit **Case Analysis** for this case to see which knowledge sources were retrieved.", icon="ℹ️")

    st.markdown("#### Generative AI Explanation")
    st.write(explanation.genai.summary)
    if explanation.genai.ran:
        g1, g2, g3 = st.columns(3)
        g1.metric("Model", explanation.genai.model_used or "n/a")
        g2.metric("Source", explanation.genai.source)
        g3.metric("Used Fallback", "Yes" if explanation.genai.used_fallback else "No")
    else:
        st.info("Visit **Case Analysis** for this case to generate an AI summary.", icon="ℹ️")

    st.markdown("#### Response Plan Explanation")
    st.write(explanation.response_plan.summary)
    if not explanation.response_plan.ran:
        st.info("Visit **Case Analysis** for this case to generate a response plan.", icon="ℹ️")

    st.markdown("#### Human Analyst Review")
    st.write(explanation.human_review.summary)

    with st.expander(f"Knowledge base: {len(RULES)} rules"):
        rules_overview = pd.DataFrame(
            [
                {"Rule ID": r.rule_id, "Name": r.name, "Condition": r.condition_text, "Incident Type": r.incident_type}
                for r in RULES
            ]
        )
        st.dataframe(rules_overview, use_container_width=True, hide_index=True)


def _render_audit_trail_tab(confidence_result) -> None:
    st.subheader("Audit Trail")
    st.caption(
        "A persistent, append-only record of when each pipeline stage ran for a "
        "case. Observational only — nothing here can change a prior audit event."
    )

    results_df = confidence_result.results_df
    options = {
        idx: f"[{build_incident_id(idx)}] {row['timestamp']} · {row['incident_type']} ({row['severity']})"
        for idx, row in results_df.iterrows()
    }
    option_idx = list(options.keys())
    default_incident = st.session_state.get("last_selected_incident_id")
    default_index = 0
    if default_incident:
        for i, idx in enumerate(option_idx):
            if build_incident_id(idx) == default_incident:
                default_index = i
                break
    selected_idx = st.selectbox(
        "Case", options=option_idx, index=default_index, format_func=lambda i: options[i], key="audit_case_select"
    )
    incident_id = build_incident_id(selected_idx)

    audit_summary = summarize_incident_audit_history(incident_id)
    if audit_summary.total_events == 0:
        st.info("No audit events recorded yet for this case.")
        return

    st.caption(
        f"{audit_summary.total_events} event(s) recorded · first at "
        f"{audit_summary.first_event_timestamp} · latest at {audit_summary.last_event_timestamp}."
    )
    stage_cols = st.columns(len(AUDIT_STAGES))
    for col, stage in zip(stage_cols, AUDIT_STAGES):
        with col:
            done = stage in audit_summary.stages_recorded
            st.markdown(f"**{'✅' if done else '⬜'}**")
            st.caption(stage.replace("_", " ").title())
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Timestamp": ev.timestamp, "Stage": ev.stage, "Event Type": ev.event_type,
                    "Incident Type": ev.incident_type, "Severity": ev.severity, "Confidence": ev.confidence_level,
                    "Analyst Decision": ev.analyst_decision or "—", "Details": ev.descriptive_details,
                }
                for ev in audit_summary.events
            ]
        ),
        use_container_width=True, hide_index=True,
    )


def _render_evaluation_tab(pre, anomaly_result, rule_result, confidence_result) -> None:
    st.subheader("Evaluation / Performance")
    st.caption(
        "Descriptive counts, rates, and (where a ground-truth label exists) "
        "offline evaluation metrics, drawn from what the system has already produced."
    )
    st.warning(
        "**This tab is EVALUATION, not automatic optimization.** It cannot change "
        "the anomaly detector, rule thresholds, confidence thresholds, RAG settings, "
        "the AI model, response-planning logic, or any analyst decision. `true_label` "
        "is read here ONLY to calculate offline evaluation metrics for this synthetic "
        "dataset — every metric that uses it is labeled **Offline Evaluation**.",
        icon="🔒",
    )

    try:
        report = build_evaluation_report(anomaly_result, rule_result, confidence_result)
    except Exception as exc:  # noqa: BLE001 - never crash the app
        st.error("Something went wrong while building the evaluation report.")
        st.exception(exc)
        return

    st.markdown("#### A. Dataset Overview")
    ov = report.dataset_overview
    o1, o2, o3 = st.columns(3)
    o1.metric("Total Events", ov.total_events)
    o2.metric("Predicted Normal", ov.predicted_normal_count)
    o3.metric("Predicted Anomalous", ov.predicted_anomalous_count)
    if ov.ground_truth_available:
        st.caption("Offline Evaluation — ground-truth split (true_label), for comparison only:")
        g1, g2 = st.columns(2)
        g1.metric("Ground Truth Normal", ov.ground_truth_normal_count)
        g2.metric("Ground Truth Anomalous", ov.ground_truth_anomalous_count)
    else:
        st.info("No `true_label` column is available, so the ground-truth distribution is not shown.", icon="ℹ️")

    st.divider()

    st.markdown("#### B. Anomaly Detection Performance")
    am = report.anomaly_metrics
    st.caption(am.note)
    if am.measured:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Accuracy", f"{am.accuracy:.2%}")
        b2.metric("Precision", f"{am.precision:.2%}")
        b3.metric("Recall", f"{am.recall:.2%}")
        b4.metric("F1-score", f"{am.f1_score:.2%}")
        cm_df = pd.DataFrame(
            am.confusion_matrix,
            index=[f"Actual: {lbl.title()}" for lbl in am.confusion_matrix_labels],
            columns=[f"Predicted: {lbl.title()}" for lbl in am.confusion_matrix_labels],
        )
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Accuracy/precision/recall/F1 are unavailable/not measured (no ground truth).", icon="ℹ️")

    with st.expander("📈 Anomaly detection charts & advanced options"):
        results_df = anomaly_result.results_df
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.caption("Normal vs. Anomalous Count")
            counts = results_df["anomaly_status"].value_counts().rename_axis("status").reset_index(name="count")
            fig_a = px.bar(counts, x="status", y="count", color="status", color_discrete_map=STATUS_COLOR_MAP, text="count")
            fig_a.update_layout(showlegend=False)
            st.plotly_chart(_style_chart(fig_a), use_container_width=True)
        with viz_col2:
            st.caption("Anomaly Score Distribution")
            fig_b = px.histogram(
                results_df, x="anomaly_score", color="anomaly_status", nbins=30,
                color_discrete_map=STATUS_COLOR_MAP, barmode="overlay", opacity=0.75,
                labels={"anomaly_score": "Anomaly score (higher = more anomalous)"},
            )
            st.plotly_chart(_style_chart(fig_b), use_container_width=True)

        viz_col3, viz_col4 = st.columns(2)
        with viz_col3:
            st.caption("Timeline of Events")
            fig_c = px.scatter(
                results_df, x="timestamp", y="anomaly_score", color="anomaly_status",
                color_discrete_map=STATUS_COLOR_MAP, hover_data=["user", "event_type"],
                labels={"anomaly_score": "Anomaly score"},
            )
            fig_c.update_traces(marker=dict(size=8, line=dict(width=0)))
            st.plotly_chart(_style_chart(fig_c), use_container_width=True)
        with viz_col4:
            st.caption("Request Count vs. Data Transferred")
            fig_d = px.scatter(
                results_df, x="request_count", y="data_transferred_mb", color="anomaly_status",
                color_discrete_map=STATUS_COLOR_MAP, hover_data=["user", "event_type"],
                labels={"data_transferred_mb": "Data transferred (MB)"},
            )
            fig_d.update_traces(marker=dict(size=8, line=dict(width=0)))
            st.plotly_chart(_style_chart(fig_d), use_container_width=True)

        st.markdown("**Full anomaly-scored event table**")
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            filter_choice = st.selectbox("Filter", ["All Events", "Anomalous Only", "Normal Only"], key="eval_anomaly_filter")
        with col_f2:
            sort_desc = st.checkbox("Sort by anomaly score (most unusual first)", value=True, key="eval_anomaly_sort")
        table_df = results_df
        if filter_choice == "Anomalous Only":
            table_df = table_df[table_df["anomaly_status"] == "ANOMALOUS"]
        elif filter_choice == "Normal Only":
            table_df = table_df[table_df["anomaly_status"] == "NORMAL"]
        table_df = table_df.sort_values("anomaly_score", ascending=not sort_desc)
        st.dataframe(
            table_df[RESULT_TABLE_COLUMNS], use_container_width=True, hide_index=True,
            column_config={"anomaly_score": st.column_config.NumberColumn(format="%.4f")},
        )
        st.caption(f"Showing {len(table_df)} of {anomaly_result.total_events} total events.")

        st.markdown("**Optional: Self-Organizing Map (exploratory clustering)**")
        st.caption(
            "Purely exploratory - shows how events cluster in feature space. Isolation "
            "Forest above remains the only anomaly detector; the SOM does not produce "
            "its own anomaly predictions and never replaces it."
        )
        if not is_som_available():
            st.info("MiniSom is not installed, so this optional section is skipped.", icon="ℹ️")
        else:
            if st.checkbox("Run SOM clustering visualization", key="eval_som_checkbox"):
                try:
                    som_result = run_som_analysis(pre.ml_df)
                    som_df = som_result.winner_coords.copy()
                    som_df["anomaly_status"] = results_df.loc[som_df.index, "anomaly_status"]
                    rng = np.random.default_rng(ISOLATION_FOREST_RANDOM_STATE)
                    som_df["som_x_jitter"] = som_df["som_x"] + rng.uniform(-0.3, 0.3, len(som_df))
                    som_df["som_y_jitter"] = som_df["som_y"] + rng.uniform(-0.3, 0.3, len(som_df))
                    fig_som = px.scatter(
                        som_df, x="som_x_jitter", y="som_y_jitter", color="anomaly_status",
                        color_discrete_map=STATUS_COLOR_MAP,
                        labels={"som_x_jitter": "SOM grid X", "som_y_jitter": "SOM grid Y"},
                    )
                    fig_som.update_traces(marker=dict(size=8, line=dict(width=0)))
                    st.plotly_chart(_style_chart(fig_som), use_container_width=True)
                    st.caption(
                        f"Grid: {som_result.grid_x}×{som_result.grid_y} neurons · "
                        f"quantization error: {som_result.quantization_error:.4f}"
                    )
                except Exception as exc:  # noqa: BLE001 - optional feature must never break the page
                    st.warning("SOM analysis failed, but this does not affect the Isolation Forest results above.")
                    st.exception(exc)

        st.markdown("**Advanced: re-run anomaly detection with different settings**")
        st.metric("Algorithm", "Isolation Forest")
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            n_estimators = st.number_input(
                "n_estimators", min_value=10, max_value=500, value=ISOLATION_FOREST_N_ESTIMATORS, step=10,
                key="eval_n_estimators",
            )
        with col_cfg2:
            contamination = st.slider(
                "contamination", min_value=0.01, max_value=0.5, value=float(ISOLATION_FOREST_CONTAMINATION), step=0.01,
                key="eval_contamination",
            )
        with col_cfg3:
            st.metric("random_state (fixed)", ISOLATION_FOREST_RANDOM_STATE)
        if st.button("Re-run Anomaly Detection", key="eval_rerun_anomaly"):
            try:
                st.session_state["anomaly_result"] = run_anomaly_detection(
                    pre, n_estimators=int(n_estimators), contamination=float(contamination)
                )
                st.success("Anomaly detection re-run. This new result is now used everywhere in the app for this session.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error("Anomaly detection could not be run with these settings.")
                st.exception(exc)

    st.divider()

    st.markdown("#### C. Incident Analysis")
    clf_m = report.classification_metrics
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Incident Type Distribution**")
        if clf_m.incident_type_counts:
            it_df = pd.DataFrame(
                {"Incident Type": list(clf_m.incident_type_counts.keys()), "Count": list(clf_m.incident_type_counts.values())}
            )
            fig = px.bar(it_df, x="Incident Type", y="Count", text="Count")
            st.plotly_chart(_style_chart(fig), use_container_width=True)
    with c2:
        st.markdown("**Severity Distribution**")
        if clf_m.severity_counts:
            sev_df = pd.DataFrame(
                {"Severity": list(clf_m.severity_counts.keys()), "Count": list(clf_m.severity_counts.values())}
            )
            fig = px.bar(sev_df, x="Severity", y="Count", text="Count")
            st.plotly_chart(_style_chart(fig), use_container_width=True)

    st.markdown("**Rule Firing Counts**")
    if clf_m.rule_firing_counts:
        rule_df = pd.DataFrame(
            {"Rule ID": list(clf_m.rule_firing_counts.keys()), "Times Fired": list(clf_m.rule_firing_counts.values())}
        )
        fig = px.bar(rule_df, x="Rule ID", y="Times Fired", text="Times Fired")
        st.plotly_chart(_style_chart(fig), use_container_width=True)
    else:
        st.caption("No rules have fired.")
    st.caption(
        f"{clf_m.multi_rule_event_count} event(s) triggered 2+ rules · "
        f"{clf_m.no_rule_anomaly_count} anomalous event(s) triggered no rule "
        "(classified as Other Anomaly)."
    )

    st.divider()

    st.markdown("#### D. Confidence & Human Review")
    conf_m = report.confidence_metrics
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("HIGH", conf_m.high_count)
    d2.metric("MEDIUM", conf_m.medium_count)
    d3.metric("LOW", conf_m.low_count)
    d4.metric("Human Review Recommended", conf_m.human_review_count)
    conf_df = pd.DataFrame(
        {"Confidence Level": ["HIGH", "MEDIUM", "LOW"], "Count": [conf_m.high_count, conf_m.medium_count, conf_m.low_count]}
    )
    fig = px.bar(
        conf_df, x="Confidence Level", y="Count", color="Confidence Level",
        color_discrete_map=CONFIDENCE_COLOR_MAP, text="Count",
    )
    st.plotly_chart(_style_chart(fig), use_container_width=True)

    st.divider()

    st.markdown("#### E. RAG Performance")
    rag_m = report.rag_metrics
    st.caption(rag_m.note)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Knowledge Base Size", rag_m.knowledge_base_size)
    e2.metric("Retrievals Attempted", rag_m.retrieval_attempted_count)
    e3.metric("Retrievals w/ a Match", rag_m.retrieval_hit_count)
    e4.metric("Retrievals w/ No Match", rag_m.retrieval_empty_count)
    if rag_m.top_match_counts_by_incident_type:
        st.markdown("**Top-Match Distribution by Incident Type**")
        top_df = pd.DataFrame(
            {
                "Incident Type": list(rag_m.top_match_counts_by_incident_type.keys()),
                "Top-Match Count": list(rag_m.top_match_counts_by_incident_type.values()),
            }
        )
        fig = px.bar(top_df, x="Incident Type", y="Top-Match Count", text="Top-Match Count")
        st.plotly_chart(_style_chart(fig), use_container_width=True)
    if rag_m.controlled_success_rate is not None:
        st.metric(
            "Controlled Retrieval Success Rate", f"{rag_m.controlled_success_rate:.2%}",
            help=(
                "Share of top retrieval matches whose knowledge-base entry is "
                "hand-tagged for the incident's own classified incident_type."
            ),
        )
        st.caption(f"{rag_m.controlled_success_count} of {rag_m.controlled_eligible_count} eligible top matches.")
    else:
        st.info("Controlled retrieval success rate is unavailable/not measured for this run.", icon="ℹ️")

    st.divider()

    st.markdown("#### F. Generative AI Performance")
    genai_m = report.genai_metrics
    st.caption(genai_m.note)
    f1, f2, f3 = st.columns(3)
    f1.metric("Model", genai_m.model_name)
    f2.metric("Ollama Available", "Yes" if genai_m.ollama_available else "No")
    f3.metric("Model Available", "Yes" if genai_m.model_available else "No")
    st.caption(genai_m.ollama_status_message)
    if genai_m.measured:
        gg1, gg2, gg3 = st.columns(3)
        gg1.metric("Generations Attempted", genai_m.generations_attempted)
        gg2.metric("Successful (Local AI)", genai_m.successful_generations)
        gg3.metric("Fallback", genai_m.fallback_generations)
        if genai_m.source_breakdown:
            src_df = pd.DataFrame(
                {"Source": list(genai_m.source_breakdown.keys()), "Count": list(genai_m.source_breakdown.values())}
            )
            fig = px.bar(src_df, x="Source", y="Count", text="Count")
            st.plotly_chart(_style_chart(fig), use_container_width=True)
    else:
        st.info("No AI generations are recorded in the audit trail yet.", icon="ℹ️")

    with st.expander("Measure a sample AI generation time (local machine, optional)"):
        st.caption(
            "Runs ONE real AI case-summary call for the first anomalous event in "
            "this dataset and times it. This is a local-machine sample only, not a "
            "production benchmark, and it does not change any stored result."
        )
        if st.button("Run timing sample", key="eval_timing_sample"):
            anomalous_rows = confidence_result.results_df[
                confidence_result.results_df["incident_type"] != "Normal Activity"
            ]
            if anomalous_rows.empty:
                st.warning("No anomalous events available to time.")
            else:
                sample_row = anomalous_rows.iloc[0]
                sample_retrieval = retrieve_knowledge_for_incident(
                    sample_row["_inference"], sample_row["_confidence"]
                )
                sample_context = build_incident_context(
                    sample_row["_inference"], sample_row["_confidence"],
                    retrieved_knowledge=[item.content for item in sample_retrieval.items],
                    retrieved_knowledge_items=retrieved_items_to_dicts(sample_retrieval.items),
                )
                timing = measure_genai_generation_timing(sample_context)
                if timing.measured:
                    st.success(f"{timing.elapsed_seconds:.2f} seconds")
                    st.caption(timing.note)
                else:
                    st.warning(timing.note)

    st.divider()

    st.markdown("#### G. Analyst Feedback")
    fb_m = report.feedback_metrics
    if fb_m.total_reviews == 0:
        st.info("No analyst feedback is recorded yet. Visit **Case Analysis** to record a decision.", icon="ℹ️")
    else:
        gg1, gg2, gg3, gg4 = st.columns(4)
        gg1.metric("Total Reviews", fb_m.total_reviews)
        gg2.metric("Approved", f"{fb_m.approved_count} ({fb_m.approval_rate:.0%})")
        gg3.metric("Rejected", f"{fb_m.rejected_count} ({fb_m.rejection_rate:.0%})")
        gg4.metric("Modified", f"{fb_m.modified_count} ({fb_m.modification_rate:.0%})")
        fb_df = pd.DataFrame(
            {"Decision": ["Approved", "Rejected", "Modified"], "Count": [fb_m.approved_count, fb_m.rejected_count, fb_m.modified_count]}
        )
        fig = px.bar(fb_df, x="Decision", y="Count", text="Count")
        st.plotly_chart(_style_chart(fig), use_container_width=True)

    st.divider()

    st.markdown("#### H. Audit Activity")
    audit_m = report.audit_metrics
    if audit_m.total_events == 0:
        st.info("No audit events are recorded yet.", icon="ℹ️")
    else:
        st.metric("Total Audit Events", audit_m.total_events)
        stage_df = pd.DataFrame(
            {"Stage": list(audit_m.events_per_stage.keys()), "Count": list(audit_m.events_per_stage.values())}
        )
        fig = px.bar(stage_df, x="Stage", y="Count", text="Count")
        st.plotly_chart(_style_chart(fig), use_container_width=True)

    st.divider()

    st.markdown("#### Local Performance Timing")
    st.caption(
        "Descriptive, local-machine timing only - never a production benchmark. "
        "Anomaly detection and RAG retrieval are timed fresh each time this tab "
        "loads; AI generation timing is opt-in above since it may call a live "
        "local AI server."
    )
    anomaly_timing = measure_anomaly_detection_timing(pre)
    rag_timing = measure_rag_retrieval_timing(confidence_result)
    t1, t2 = st.columns(2)
    with t1:
        if anomaly_timing.measured:
            st.metric("Anomaly Detection (train + predict)", f"{anomaly_timing.elapsed_seconds:.3f}s")
            st.caption(f"{anomaly_timing.note} (n={anomaly_timing.sample_size})")
        else:
            st.info(anomaly_timing.note, icon="ℹ️")
    with t2:
        if rag_timing.measured:
            st.metric("RAG Retrieval (all anomalous events)", f"{rag_timing.elapsed_seconds:.3f}s")
            st.caption(f"{rag_timing.note} (n={rag_timing.sample_size})")
        else:
            st.info(rag_timing.note, icon="ℹ️")


REVIEW_INSIGHTS_SECTIONS = [
    "Case History", "Analyst Decisions", "Feedback", "Continuous Improvement",
    "Explainability", "Audit Trail", "Evaluation / Performance",
]


def render_review_insights_page(df: pd.DataFrame) -> None:
    st.title("Review & Insights")
    st.caption(
        "Review previous cases, analyst decisions, feedback, system reasoning, "
        "audit history, and performance."
    )

    pipeline = _get_pipeline_results(df)
    if pipeline is None:
        return
    pre, anomaly_result, rule_result, confidence_result = pipeline

    incidents_df = _incident_universe(confidence_result)
    feedback_df = load_feedback()

    # A tab-styled section switcher (not st.tabs()) - only the selected
    # section's content is ever rendered. st.tabs() would render all seven
    # sections' dataframes and charts into the page at once (Streamlit
    # keeps inactive tab panels mounted, just visually hidden), which is
    # far more chart/grid content than one page needs at a time and made
    # the real rendered page unstable. Rendering exactly one section keeps
    # the same seven-section grouping the analyst sees, real data only,
    # nothing recomputed differently.
    selected_section = st.radio(
        "Review & Insights section", REVIEW_INSIGHTS_SECTIONS, index=0,
        horizontal=True, label_visibility="collapsed", key="ri_section",
    )
    st.divider()

    if selected_section == "Case History":
        _render_case_history_tab(incidents_df, feedback_df)
    elif selected_section == "Analyst Decisions":
        _render_analyst_decisions_tab(feedback_df)
    elif selected_section == "Feedback":
        _render_feedback_insights_tab(feedback_df)
    elif selected_section == "Continuous Improvement":
        _render_continuous_improvement_tab(feedback_df)
    elif selected_section == "Explainability":
        _render_explainability_tab(confidence_result)
    elif selected_section == "Audit Trail":
        _render_audit_trail_tab(confidence_result)
    elif selected_section == "Evaluation / Performance":
        _render_evaluation_tab(pre, anomaly_result, rule_result, confidence_result)


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------
if selected_page == "Case Analysis":
    render_case_analysis_page(logs_df)
else:
    render_review_insights_page(logs_df)
