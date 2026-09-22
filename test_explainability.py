"""
test_explainability.py
------------------------
Phase 10 tests for `src/explainability.py` ONLY.

This file deliberately does NOT chain into the Phase 1-9 regression
cascade (unlike `test_retriever.py`, which re-runs `test_feedback.main()`
and so on down the chain). Run it directly:

    python test_explainability.py

A full project regression (`python test_retriever.py`, 1,196+ checks) can
still be run separately whenever a complete check is actually needed -
this file is intentionally scoped to Phase 10's explainability module and
its Streamlit page only, per the Phase 10 task's own instruction not to
repeatedly run the full regression during development.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

import config
from src.anomaly_detector import run_anomaly_detection
from src.confidence import ConfidenceRecord, run_confidence_scoring, score_confidence
from src.explainability import (
    AnomalyExplanation,
    ClassificationExplanation,
    ConfidenceExplanation,
    GenAIExplanation,
    HumanReviewExplanation,
    IncidentExplanation,
    RagExplanation,
    ResponsePlanExplanation,
    RuleExplanation,
    SeverityExplanation,
    SupportingIndicatorsExplanation,
    UncertaintyExplanation,
    build_incident_explanation,
    explain_anomaly_detection,
    explain_classification,
    explain_confidence,
    explain_genai,
    explain_human_review,
    explain_rag,
    explain_response_plan,
    explain_rule_evidence,
    explain_severity,
    explain_supporting_indicators,
    explain_uncertainty,
)
import src.explainability as explainability_module
from src.data_loader import load_logs
from src.genai_analyzer import GenAIResult, build_incident_context, generate_ai_case_summary
from src.preprocessor import run_preprocessing_pipeline
from src.response_planner import generate_response_plan
from src.retriever import RetrievalResult, RetrievedKnowledgeItem, retrieve_knowledge_for_incident
from src.rule_engine import InferenceRecord, run_rule_engine

PASSED = 0


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1


# =======================================================================
# Shared real-pipeline fixture (computed once, reused by every test that
# needs it, so 14 test functions don't each retrain Isolation Forest).
# =======================================================================
_PIPELINE_CACHE = {}


def _pipeline():
    if "confidence_result" not in _PIPELINE_CACHE:
        df = load_logs()
        pre = run_preprocessing_pipeline(df)
        anomaly_result = run_anomaly_detection(pre)
        rule_result = run_rule_engine(pre, anomaly_result)
        confidence_result = run_confidence_scoring(rule_result)
        _PIPELINE_CACHE["confidence_result"] = confidence_result
    return _PIPELINE_CACHE["confidence_result"]


def _find_row(predicate):
    """First (inf, conf) pair in the real dataset matching `predicate(inf)`."""
    results_df = _pipeline().results_df
    for _, row in results_df.iterrows():
        inf = row["_inference"]
        if predicate(inf):
            return inf, row["_confidence"]
    raise AssertionError("No matching row found in the real dataset for this test.")


# =======================================================================
# 1. Normal-incident explanation
# =======================================================================
def test_normal_incident_explanation():
    print("\n1. A Normal Activity event is explained correctly")
    inf, conf = _find_row(lambda i: i.incident_type == "Normal Activity")

    anomaly = explain_anomaly_detection(inf)
    check(isinstance(anomaly, AnomalyExplanation), "explain_anomaly_detection returns an AnomalyExplanation")
    check(anomaly.anomaly_status == "NORMAL", "a Normal Activity event's anomaly_status is NORMAL")
    check("NORMAL" in anomaly.explanation, "the anomaly explanation text mentions NORMAL")

    classification = explain_classification(inf)
    check(classification.incident_type == "Normal Activity", "classification.incident_type is Normal Activity")
    check(classification.triggered_rules == [], "a Normal Activity event has no triggered rules")

    severity = explain_severity(inf)
    check(severity.severity == "N/A", "a Normal Activity event's severity is N/A")
    check("not classified as an incident" in severity.explanation, "severity explanation states why N/A applies")

    explanation = build_incident_explanation("INC-TEST-NORMAL", inf, conf)
    check(explanation.response_plan.ran is False, "no response plan was supplied, so response_plan.ran is False")
    check(explanation.human_review.reviewed is False, "no feedback was supplied, so human_review.reviewed is False")
    check(explanation.rag.ran is False, "no retrieval result was supplied, so rag.ran is False")
    check(explanation.genai.ran is False, "no GenAI result was supplied, so genai.ran is False")


# =======================================================================
# 2. Anomalous-incident explanation
# =======================================================================
def test_anomalous_incident_explanation():
    print("\n2. An anomalous/incident event is explained correctly")
    inf, conf = _find_row(lambda i: i.incident_type != "Normal Activity")

    anomaly = explain_anomaly_detection(inf)
    check(anomaly.anomaly_status == "ANOMALOUS", "an incident event's anomaly_status is ANOMALOUS")
    check("ANOMALOUS" in anomaly.explanation, "the anomaly explanation text mentions ANOMALOUS")

    classification = explain_classification(inf)
    check(classification.incident_type != "Normal Activity", "classification.incident_type is a real incident type")
    check(classification.incident_type == inf.incident_type, "classification.incident_type matches the source record exactly")

    severity = explain_severity(inf)
    check(severity.severity in ("LOW", "MEDIUM", "HIGH"), "an incident's severity is LOW/MEDIUM/HIGH")
    check(severity.severity == inf.severity, "severity.severity matches the source record exactly")


# =======================================================================
# 3. Rule evidence explanation
# =======================================================================
def test_rule_evidence_explanation():
    print("\n3. Rule evidence is explained using the rule engine's own text, never reworded")
    inf, conf = _find_row(lambda i: len(i.triggered_rules) >= 1)

    explanations = explain_rule_evidence(inf)
    check(len(explanations) == len(inf.triggered_rules), "one RuleExplanation per triggered rule")
    for exp, tr in zip(explanations, inf.triggered_rules):
        check(isinstance(exp, RuleExplanation), "each item is a RuleExplanation")
        check(exp.rule_id == tr.rule_id, "rule_id matches the source TriggeredRule exactly")
        check(exp.name == tr.name, "name matches the source TriggeredRule exactly")
        check(exp.conclusion == tr.conclusion, "conclusion matches the source TriggeredRule exactly")
        check(tr.explanation in exp.explanation, "the rule engine's own explanation text is reused verbatim, not reworded")


# =======================================================================
# 4. Severity explanation
# =======================================================================
def test_severity_explanation():
    print("\n4. Severity explanation names exactly the rules that actually fired")
    inf, conf = _find_row(lambda i: len(i.triggered_rules) >= 1)
    severity = explain_severity(inf)
    for tr in inf.triggered_rules:
        check(tr.rule_id in severity.explanation, f"severity explanation names triggered rule {tr.rule_id}")
    check(str(len(inf.triggered_rules)) in severity.explanation, "severity explanation states the correct rule count")


# =======================================================================
# 5. Confidence explanation
# =======================================================================
def test_confidence_explanation():
    print("\n5. Confidence explanation reuses the confidence record's own values exactly")
    inf, conf = _find_row(lambda i: True)
    explanation = explain_confidence(conf)
    check(isinstance(explanation, ConfidenceExplanation), "explain_confidence returns a ConfidenceExplanation")
    check(explanation.confidence_score == conf.confidence_score, "confidence_score matches the source record exactly")
    check(explanation.confidence_level == conf.confidence_level, "confidence_level matches the source record exactly")
    check(explanation.evidence_strength == conf.evidence_strength, "evidence_strength matches the source record exactly")
    check(explanation.reasons == list(conf.confidence_reasons), "reasons list matches the source record exactly, not invented")
    check(set(explanation.components.keys()) == set(conf.components.keys()), "confidence components are the same 4 named components")
    for name, comp in conf.components.items():
        check(explanation.components[name]["value"] == comp.value, f"{name} value matches the source component exactly")
        check(explanation.components[name]["weight"] == comp.weight, f"{name} weight matches the source component exactly")
    check(explanation.confidence_level in explanation.summary, "the summary text states the confidence level")


# =======================================================================
# 6. Uncertainty + supporting-indicators explanation
# =======================================================================
def test_uncertainty_and_supporting_indicators_explanation():
    print("\n6. Uncertainty and supporting-indicator explanations reuse the confidence record exactly")
    inf, conf = _find_row(lambda i: True)

    uncertainty = explain_uncertainty(conf)
    check(isinstance(uncertainty, UncertaintyExplanation), "explain_uncertainty returns an UncertaintyExplanation")
    check(uncertainty.uncertainty_reason == conf.uncertainty_reason, "uncertainty_reason matches the source record exactly")
    check(
        uncertainty.human_review_recommended == conf.human_review_recommended,
        "human_review_recommended matches the source record exactly",
    )

    indicators = explain_supporting_indicators(conf)
    check(isinstance(indicators, SupportingIndicatorsExplanation), "explain_supporting_indicators returns a SupportingIndicatorsExplanation")
    check(indicators.indicators == list(conf.supporting_indicators), "indicators list matches the source record exactly, not invented")


# =======================================================================
# 7. RAG explanation (present and absent)
# =======================================================================
def test_rag_explanation():
    print("\n7. RAG explanation correctly distinguishes 'not run yet' from real retrieved sources")
    absent = explain_rag(None)
    check(isinstance(absent, RagExplanation), "explain_rag returns a RagExplanation")
    check(absent.ran is False, "explain_rag(None) reports ran=False")
    check(absent.items == [] and absent.count == 0, "explain_rag(None) has no items")

    inf, conf = _find_row(lambda i: i.incident_type != "Normal Activity")
    retrieval = retrieve_knowledge_for_incident(inf, conf)
    present = explain_rag(retrieval)
    check(present.ran is True, "explain_rag(a real RetrievalResult) reports ran=True")
    check(present.count == len(retrieval.items), "item count matches the real retrieval result exactly")
    retrieved_ids = {item.knowledge_id for item in retrieval.items}
    explained_ids = {item["knowledge_id"] for item in present.items}
    check(explained_ids == retrieved_ids, "every explained knowledge_id is exactly one that was actually retrieved - none invented")
    for item, source in zip(present.items, retrieval.items):
        check(item["similarity_score"] == source.similarity_score, "similarity_score matches the real retrieval result exactly")


# =======================================================================
# 8. GenAI explanation (present and absent, authoritative values preserved)
# =======================================================================
def test_genai_explanation():
    print("\n8. GenAI explanation never lets AI-generated content redefine the authoritative SYSTEM values")
    inf, conf = _find_row(lambda i: i.incident_type != "Normal Activity")

    absent = explain_genai(None, inf, conf)
    check(isinstance(absent, GenAIExplanation), "explain_genai returns a GenAIExplanation")
    check(absent.ran is False, "explain_genai(None, ...) reports ran=False")
    check(absent.authoritative_incident_type == inf.incident_type, "authoritative_incident_type still comes from the InferenceRecord, even with no GenAI result")
    check(absent.authoritative_severity == inf.severity, "authoritative_severity still comes from the InferenceRecord")
    check(absent.authoritative_confidence_level == conf.confidence_level, "authoritative_confidence_level still comes from the ConfidenceRecord")

    context = build_incident_context(inf, conf)
    genai_result = generate_ai_case_summary(context)  # sandbox has no Ollama server -> deterministic fallback
    present = explain_genai(genai_result, inf, conf)
    check(present.ran is True, "explain_genai(a real GenAIResult, ...) reports ran=True")
    check(present.model_used == genai_result.model_used, "model_used matches the real GenAIResult exactly")
    check(present.source == genai_result.source, "source matches the real GenAIResult exactly")
    check(present.used_fallback == genai_result.used_fallback, "used_fallback matches the real GenAIResult exactly")
    # The authoritative values are STILL taken from inf/conf, never from genai_result
    # (GenAIResult has no incident_type/severity field at all to read from).
    check(present.authoritative_incident_type == inf.incident_type, "authoritative_incident_type still comes from the InferenceRecord, not the AI text")
    check(present.authoritative_severity == inf.severity, "authoritative_severity still comes from the InferenceRecord, not the AI text")


# =======================================================================
# 9. Response-plan explanation (present and absent)
# =======================================================================
def test_response_plan_explanation():
    print("\n9. Response-plan explanation reuses an already-generated plan, never builds its own")
    inf, conf = _find_row(lambda i: i.incident_type != "Normal Activity")

    absent = explain_response_plan(None)
    check(isinstance(absent, ResponsePlanExplanation), "explain_response_plan returns a ResponsePlanExplanation")
    check(absent.ran is False, "explain_response_plan(None) reports ran=False")

    plan = generate_response_plan(inf, conf)
    present = explain_response_plan(plan)
    check(present.ran is True, "explain_response_plan(a real plan) reports ran=True")
    check(present.plan_title == plan.plan_title, "plan_title matches the real plan exactly")
    check(present.priority == plan.priority, "priority matches the real plan exactly")
    check(present.step_count == len(plan.steps), "step_count matches the real plan exactly")
    check(plan.planning_reason in present.summary or present.planning_reason == plan.planning_reason, "the plan's own planning_reason is reused, not reinvented")


# =======================================================================
# 10. Human-review explanation (present and absent, dict and Series)
# =======================================================================
def test_human_review_explanation():
    print("\n10. Human-review explanation reuses a real feedback row, never invents a decision")
    absent = explain_human_review(None)
    check(isinstance(absent, HumanReviewExplanation), "explain_human_review returns a HumanReviewExplanation")
    check(absent.reviewed is False, "explain_human_review(None) reports reviewed=False")

    row_dict = {
        "analyst_decision": "REJECT",
        "review_status": "REJECTED",
        "analyst_notes": "Confirmed false positive after review.",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    present = explain_human_review(row_dict)
    check(present.reviewed is True, "a real feedback row reports reviewed=True")
    check(present.analyst_decision == "REJECT", "analyst_decision matches the real feedback row exactly")
    check("REJECT" in present.summary, "the summary mentions the real recorded decision")
    check("Confirmed false positive after review." in present.summary, "the summary includes the real analyst notes, not invented ones")

    row_series = pd.Series(row_dict)
    present_series = explain_human_review(row_series)
    check(present_series.analyst_decision == "REJECT", "explain_human_review also accepts a pandas Series, not just a dict")


# =======================================================================
# 11. build_incident_explanation() end-to-end consistency
# =======================================================================
def test_build_incident_explanation_end_to_end():
    print("\n11. build_incident_explanation() assembles a fully consistent bundle from real objects")
    inf, conf = _find_row(lambda i: i.incident_type != "Normal Activity" and len(i.triggered_rules) >= 1)
    retrieval = retrieve_knowledge_for_incident(inf, conf)
    context = build_incident_context(inf, conf, retrieved_knowledge=[i.content for i in retrieval.items])
    genai_result = generate_ai_case_summary(context)
    plan = generate_response_plan(inf, conf, genai_result=genai_result)
    feedback_row = {"analyst_decision": "APPROVE", "review_status": "APPROVED", "analyst_notes": "", "timestamp": "2026-01-01T00:00:00+00:00"}

    explanation = build_incident_explanation(
        "INC-TEST-0001", inf, conf,
        retrieval_result=retrieval, genai_result=genai_result, response_plan=plan, latest_feedback_row=feedback_row,
    )
    check(isinstance(explanation, IncidentExplanation), "build_incident_explanation returns an IncidentExplanation")
    check(explanation.incident_id == "INC-TEST-0001", "incident_id is passed through unchanged")
    check(explanation.classification.incident_type == inf.incident_type, "bundled classification matches the source InferenceRecord")
    check(explanation.confidence.confidence_score == conf.confidence_score, "bundled confidence matches the source ConfidenceRecord")
    check(explanation.rag.ran is True and explanation.rag.count == len(retrieval.items), "bundled RAG explanation matches the real retrieval result")
    check(explanation.genai.ran is True, "bundled GenAI explanation reflects that a real GenAIResult was supplied")
    check(explanation.response_plan.ran is True and explanation.response_plan.step_count == len(plan.steps), "bundled response-plan explanation matches the real plan")
    check(explanation.human_review.reviewed is True and explanation.human_review.analyst_decision == "APPROVE", "bundled human-review explanation matches the supplied feedback row")


# =======================================================================
# 12. No unsupported / invented evidence
# =======================================================================
def test_no_unsupported_or_invented_evidence():
    print("\n12. No explanation ever names a rule, knowledge source, or indicator that wasn't actually present")
    checked_any = False
    results_df = _pipeline().results_df
    for _, row in results_df.head(60).iterrows():
        inf = row["_inference"]
        conf = row["_confidence"]
        if inf.incident_type == "Normal Activity":
            continue
        checked_any = True

        classification = explain_classification(inf)
        real_rule_ids = {tr.rule_id for tr in inf.triggered_rules}
        explained_rule_ids = {r.rule_id for r in classification.triggered_rules}
        check(explained_rule_ids == real_rule_ids, f"{inf.incident_type}: explained rule IDs exactly match the real triggered rules, no invention")

        indicators = explain_supporting_indicators(conf)
        check(indicators.indicators == list(conf.supporting_indicators), f"{inf.incident_type}: supporting indicators exactly match the real confidence record")

        retrieval = retrieve_knowledge_for_incident(inf, conf)
        rag = explain_rag(retrieval)
        real_kb_ids = {item.knowledge_id for item in retrieval.items}
        explained_kb_ids = {item["knowledge_id"] for item in rag.items}
        check(explained_kb_ids == real_kb_ids, f"{inf.incident_type}: explained RAG knowledge IDs exactly match what was actually retrieved")

    check(checked_any, "at least one real incident was checked for unsupported/invented evidence")


# =======================================================================
# 13. true_label is never used, and the module never recomputes anything
# =======================================================================
def test_true_label_not_used_and_module_never_recomputes():
    print("\n13. true_label is never used, and no earlier-phase compute function is imported")
    module_vars = vars(explainability_module)
    check("LABEL_COLUMN" not in module_vars, "config.LABEL_COLUMN is not imported into explainability.py's namespace")

    for dc in (
        AnomalyExplanation, ClassificationExplanation, SeverityExplanation, ConfidenceExplanation,
        SupportingIndicatorsExplanation, UncertaintyExplanation, RagExplanation, GenAIExplanation,
        ResponsePlanExplanation, HumanReviewExplanation, IncidentExplanation, RuleExplanation,
    ):
        field_names = {f.name for f in dataclasses.fields(dc)}
        check("true_label" not in field_names, f"{dc.__name__} has no true_label field")

    # This module must EXPLAIN existing decisions, never recompute them -
    # none of the actual compute/training/retrieval/generation functions
    # from earlier phases should be importable from this module's namespace.
    never_imported = [
        "run_inference", "run_rule_engine", "score_confidence", "run_confidence_scoring",
        "retrieve", "retrieve_knowledge_for_incident", "generate_ai_case_summary",
        "generate_response_plan", "train_isolation_forest", "run_anomaly_detection",
    ]
    for name in never_imported:
        check(name not in module_vars, f"'{name}' is not imported into explainability.py - it never recomputes an earlier phase's decision")


# =======================================================================
# 14. Streamlit UI - the Explainability & Audit Trail page
# =======================================================================
def test_streamlit_explainability_page_renders():
    print("\n14. The Explainability & Audit Trail page renders with no exceptions")
    from contextlib import contextmanager
    import os
    import tempfile

    import config as config_module
    import src.audit_trail as audit_trail_module
    import src.feedback as feedback_module

    @contextmanager
    def isolated_app_databases():
        feedback_fd, feedback_path = tempfile.mkstemp(suffix="_feedback.db")
        audit_fd, audit_path = tempfile.mkstemp(suffix="_audit.db")
        os.close(feedback_fd)
        os.close(audit_fd)
        os.remove(feedback_path)
        os.remove(audit_path)
        original_feedback_path = feedback_module.FEEDBACK_DB_PATH
        original_audit_path = audit_trail_module.AUDIT_DB_PATH
        feedback_module.FEEDBACK_DB_PATH = feedback_path
        audit_trail_module.AUDIT_DB_PATH = audit_path
        try:
            yield
        finally:
            feedback_module.FEEDBACK_DB_PATH = original_feedback_path
            audit_trail_module.AUDIT_DB_PATH = original_audit_path
            for p in (feedback_path, audit_path):
                if os.path.exists(p):
                    os.remove(p)

    from streamlit.testing.v1 import AppTest

    with isolated_app_databases():
        at = AppTest.from_file("app.py")
        at.run(timeout=120)
        check(len(at.exception) == 0, "the app loads on the Case Analysis (landing) page with no exception")

        # UI redesign note: "Explainability & Audit Trail" is no longer a
        # standalone page - it was split into an "Explainability" section
        # and a separate "Audit Trail" section, both under "Review &
        # Insights". Review & Insights renders exactly ONE section per run
        # (a tab-styled section switcher, not st.tabs() - st.tabs() would
        # mount all seven sections' dataframes/charts into the page at
        # once, which in real-browser testing crashed the page), so each
        # section is selected and checked on its own. The Explainability
        # section's own sub-sections are one heading level below its
        # "Explainability" subheader (st.markdown("#### ..."), not
        # st.subheader) - the same "check markdown, not subheader"
        # adaptation already established in test_continuous_improvement.py
        # when a heading's widget type changed under this project.
        at.sidebar.radio[0].set_value("Review & Insights").run(timeout=120)
        check(len(at.exception) == 0, "the Review & Insights page loads with no exception")
        ri_section = [r for r in at.radio if r.key == "ri_section"][0]

        ri_section.set_value("Explainability").run(timeout=120)
        check(len(at.exception) == 0, "the Explainability section loads with no exception")
        subheaders = [s.value for s in at.subheader]
        check("Explainability" in subheaders, "the Explainability section is present")

        headings = [m.value for m in at.markdown]
        for expected in (
            "#### Incident Overview", "#### Anomaly Detection", "#### Rule / Knowledge-Based Explanation",
            "#### Severity + Confidence Explanation", "#### RAG Sources", "#### Generative AI Explanation",
            "#### Response Plan Explanation", "#### Human Analyst Review",
        ):
            check(expected in headings, f"the '{expected.lstrip('# ')}' section is present on the page")

        # Incident selection works: change the selected event (in the
        # Explainability section's own case selector) and rerun.
        selectboxes = [s for s in at.selectbox if s.key == "explain_case_select"]
        check(len(selectboxes) == 1, "the Explainability section's incident-selection selectbox is present")
        selectboxes[0].set_value(7).run(timeout=120)
        check(len(at.exception) == 0, "selecting a different incident re-renders with no exception")

    with isolated_app_databases():
        at2 = AppTest.from_file("app.py")
        at2.run(timeout=120)
        at2.sidebar.radio[0].set_value("Review & Insights").run(timeout=120)
        ri_section2 = [r for r in at2.radio if r.key == "ri_section"][0]
        ri_section2.set_value("Audit Trail").run(timeout=120)
        check(len(at2.exception) == 0, "the Audit Trail section loads with no exception")
        subheaders2 = [s.value for s in at2.subheader]
        check("Audit Trail" in subheaders2, "the Audit Trail section (successor to 'Audit Timeline') is present")


# =======================================================================
def main():
    test_normal_incident_explanation()
    test_anomalous_incident_explanation()
    test_rule_evidence_explanation()
    test_severity_explanation()
    test_confidence_explanation()
    test_uncertainty_and_supporting_indicators_explanation()
    test_rag_explanation()
    test_genai_explanation()
    test_response_plan_explanation()
    test_human_review_explanation()
    test_build_incident_explanation_end_to_end()
    test_no_unsupported_or_invented_evidence()
    test_true_label_not_used_and_module_never_recomputes()
    test_streamlit_explainability_page_renders()

    print("\n" + "=" * 70)
    print(f"PHASE 10 EXPLAINABILITY CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)


if __name__ == "__main__":
    main()
