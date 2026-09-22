"""
test_genai.py
---------------
A small, dependency-free validation script for the Phase 5 Generative AI
analyst layer (src/genai_analyzer.py). Same style as test_preprocessing.py,
test_anomaly_detection.py, test_rule_engine.py, and test_confidence.py:
plain asserts, no pytest.

This file also folds in a full regression run of the Phase 1, Phase 2,
Phase 3, and Phase 4 test suites (test_confidence.py already re-runs all
three of those itself), so a single command verifies the whole pipeline
still works after adding Phase 5:

    python test_genai.py

IMPORTANT: this sandboxed test environment does NOT run a local Ollama
server, so `is_ollama_available()` is expected to return False here and
every generation test below exercises the DETERMINISTIC FALLBACK path.
Per the Phase 5 spec, that is not treated as a project failure - the
whole point of the fallback is that the application stays fully useful
without Ollama. The "structured output parsing" and "malformed response"
tests call `parse_ai_response()` directly with synthetic text, so they
fully exercise the Ollama-response-handling code paths without needing a
real model.

Never modifies data/synthetic_security_logs.csv.
"""

import pandas as pd

from src.anomaly_detector import run_anomaly_detection
from src.confidence import run_confidence_scoring, score_confidence
from src.data_loader import load_logs
from src.genai_analyzer import (
    GenAIResult,
    IncidentContext,
    OllamaStatus,
    build_incident_context,
    build_prompt,
    generate_ai_case_summary,
    generate_fallback_summary,
    get_ollama_status,
    is_model_available,
    is_ollama_available,
    parse_ai_response,
)
from src.preprocessor import run_preprocessing_pipeline
from src.rule_engine import run_inference, run_rule_engine

PASSED = 0


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


# Same baseline event used by test_rule_engine.py / test_confidence.py.
BASE_ROW = {
    "failed_logins": 0,
    "request_count": 10,
    "port": 443,
    "data_transferred_mb": 5.0,
    "session_duration": 300,
    "device_count": 1,
    "event_type": "login_success",
    "login_location": "Mumbai, IN",
    "hour": 12,
    "is_off_hours": 0,
    "location_mismatch": 0,
    "anomaly_score": 0.0,
    "anomaly_status": "NORMAL",
    "true_label": "normal",
}


def make_row(**overrides) -> pd.Series:
    data = dict(BASE_ROW)
    data.update(overrides)
    return pd.Series(data)


def make_context(row: pd.Series, anomaly_score_normalized: float = 0.6) -> IncidentContext:
    """Build a Phase 5 IncidentContext straight from a crafted row, exactly
    the way the app does: Phase 3 InferenceRecord + Phase 4 ConfidenceRecord."""
    inference_record = run_inference(row)
    confidence_record = score_confidence(inference_record, anomaly_score_normalized=anomaly_score_normalized)
    return build_incident_context(inference_record, confidence_record)


def test_ollama_availability_check():
    print("\n1. Ollama availability check never raises and returns a plain bool")
    available = is_ollama_available()
    check(isinstance(available, bool), "is_ollama_available() returns a bool")
    print(f"     (this environment reports Ollama available = {available})")


def test_model_unavailable_handling():
    print("\n2. Model-unavailable handling is safe and clearly reported")
    fake_model = "definitely-not-a-real-model-xyz"
    result = is_model_available(fake_model)
    check(isinstance(result, bool), "is_model_available() returns a bool, never raises")

    status = get_ollama_status(fake_model)
    check(isinstance(status, OllamaStatus), "get_ollama_status() returns an OllamaStatus")
    check(status.model_name == fake_model, "status reports back the requested model name")
    check(not status.model_available, "a made-up model name is correctly reported as unavailable")
    check(len(status.message) > 0, "status includes a clear, non-empty message")


def test_fallback_summary_generation():
    print("\n3. Fallback summary generation produces a complete, valid result")
    context = make_context(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    result = generate_fallback_summary(context)
    check(isinstance(result, GenAIResult), "generate_fallback_summary() returns a GenAIResult")
    check(result.used_fallback is True, "used_fallback is True for the deterministic fallback")
    check(result.source == "fallback", "source is 'fallback'")
    for field_name in [
        "summary", "observed_evidence", "why_suspicious", "incident_classification",
        "severity_explanation", "confidence_explanation", "supporting_evidence", "human_review",
    ]:
        value = getattr(result, field_name)
        check(isinstance(value, str) and len(value) > 0, f"'{field_name}' is a non-empty string")
    check(isinstance(result.recommendations, list) and len(result.recommendations) > 0, "recommendations is a non-empty list")
    check(
        any("unavailable" in w.lower() for w in result.warnings),
        "the fallback clearly labels itself as a fallback, not a silent 'AI unavailable' message",
    )


def test_fallback_summary_contains_supplied_evidence():
    print("\n4. Fallback summary reflects the ACTUAL supplied evidence, not a generic template")
    context = make_context(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    result = generate_fallback_summary(context)
    check("RULE-BF-01" in result.severity_explanation, "the triggered rule ID (RULE-BF-01) appears in the severity explanation")
    check(
        "failed_logins = 23" in result.observed_evidence,
        "the actual observed value (failed_logins = 23) appears in the observed-evidence text",
    )
    check("Possible Brute Force" in result.incident_classification, "the actual incident_type appears in the classification text")

    # A different event's evidence should produce visibly different text - proves this
    # isn't a hard-coded example (Requirement #14).
    other_context = make_context(make_row(anomaly_status="ANOMALOUS", data_transferred_mb=900))
    other_result = generate_fallback_summary(other_context)
    check(other_result.summary != result.summary, "a different event produces a different summary, not a fixed template")


def test_fallback_does_not_use_true_label():
    print("\n5. Fallback (and the whole module) never uses true_label")
    row_with_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="suspicious")
    row_without_label = make_row(anomaly_status="ANOMALOUS", failed_logins=23)
    del row_without_label["true_label"]

    result_with = generate_fallback_summary(make_context(row_with_label))
    result_without = generate_fallback_summary(make_context(row_without_label))
    check(result_with.summary == result_without.summary, "presence/absence of true_label does not change the fallback summary")
    check(
        result_with.recommendations == result_without.recommendations,
        "presence/absence of true_label does not change the fallback recommendations",
    )

    # IncidentContext structurally has no field for it at all.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(IncidentContext)}
    check("true_label" not in field_names, "IncidentContext has no true_label field at all")

    # Static check, mirroring the other test suites' approach.
    import src.genai_analyzer as genai_module

    check(
        "LABEL_COLUMN" not in vars(genai_module),
        "config.LABEL_COLUMN is not imported into genai_analyzer.py's namespace",
    )


def test_prompt_includes_structured_evidence():
    print("\n6. Prompt generation includes the structured incident evidence")
    context = make_context(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    system_prompt, user_prompt = build_prompt(context)
    check(len(system_prompt) > 0, "a non-empty system prompt is produced")
    check("defensive" in system_prompt.lower(), "the system prompt defines a defensive-only role")
    check(context.incident_type in user_prompt, "the user prompt contains the actual incident_type")
    check(context.severity in user_prompt, "the user prompt contains the actual severity")
    check("RULE-BF-01" in user_prompt, "the user prompt contains the actual triggered rule ID")
    check("failed_logins" in user_prompt, "the user prompt contains the actual observed evidence field")
    check(str(context.human_review_recommended) in user_prompt, "the user prompt states the actual human_review_recommended value")


def test_prompt_excludes_true_label():
    print("\n7. Prompt generation excludes true_label")
    context = make_context(make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="suspicious"))
    system_prompt, user_prompt = build_prompt(context)
    check("true_label" not in system_prompt.lower(), "true_label does not appear in the system prompt")
    check("true_label" not in user_prompt.lower(), "true_label does not appear in the user prompt")

    # Confirm the true_label VALUE doesn't leak in either, using a value that
    # can't collide with legitimate prompt vocabulary (unlike "suspicious",
    # which is also a normal English word used in the AI's own output schema
    # field name "why_suspicious").
    distinctive_context = make_context(
        make_row(anomaly_status="ANOMALOUS", failed_logins=23, true_label="zzz_ground_truth_marker_zzz")
    )
    _, distinctive_user_prompt = build_prompt(distinctive_context)
    check(
        "zzz_ground_truth_marker_zzz" not in distinctive_user_prompt,
        "a distinctive true_label VALUE does not leak into the prompt",
    )


def test_structured_output_parsing():
    print("\n8. Well-formed structured (JSON) AI output is parsed correctly")
    context = make_context(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    raw = (
        '{"summary": "A likely brute-force attempt was observed.", '
        '"observed_evidence": "failed_logins = 23", '
        '"why_suspicious": "High failed login count on an anomalous event.", '
        '"incident_classification": "This looks like a brute-force attack.", '
        '"severity_explanation": "Escalated due to repeated failures.", '
        '"confidence_explanation": "Fairly confident given the evidence.", '
        '"supporting_evidence": "RULE-BF-01 triggered.", '
        '"recommendations": ["Review login logs", "Verify the account with the user"], '
        '"human_review": "Not required."}'
    )
    result = parse_ai_response(raw, context)
    check(result.source == "ollama", "a well-formed JSON response is attributed to source 'ollama'")
    check(result.used_fallback is False, "used_fallback is False for a successfully parsed response")
    check(len(result.warnings) == 0, "no warnings are recorded for a clean, well-formed response")
    check(
        f"Incident Type: {context.incident_type}. Severity: {context.severity}." in result.incident_classification,
        "the authoritative incident_type/severity is present in incident_classification (Requirement #13)",
    )
    check(
        f"Confidence Level: {context.confidence_level}" in result.confidence_explanation,
        "the authoritative confidence_level is present in confidence_explanation (Requirement #13)",
    )
    check("Review login logs" in result.recommendations, "the model's own defensive recommendation is preserved")


def test_malformed_response_does_not_crash():
    print("\n9. Malformed AI responses never crash the application")
    context = make_context(make_row(anomaly_status="ANOMALOUS", failed_logins=23))

    not_json = "I think this looks like a brute force attempt based on the evidence given."
    result1 = parse_ai_response(not_json, context)  # must not raise
    check(isinstance(result1, GenAIResult), "non-JSON free text still produces a valid GenAIResult")
    check(result1.used_fallback is True, "non-JSON text is treated as a fallback (text-fallback path)")
    check(len(result1.warnings) > 0, "a warning is recorded when JSON parsing fails")

    missing_fields_json = '{"summary": "Something happened."}'
    result2 = parse_ai_response(missing_fields_json, context)  # must not raise
    check(isinstance(result2, GenAIResult), "JSON missing required fields still produces a valid GenAIResult")
    check(result2.used_fallback is True, "incomplete JSON falls back rather than producing a half-built result")

    empty_response = ""
    result3 = parse_ai_response(empty_response, context)  # must not raise
    check(isinstance(result3, GenAIResult), "a completely empty AI response still produces a valid GenAIResult")
    check(result3.source == "fallback", "an empty response falls all the way through to the deterministic fallback")

    garbage = "{not valid json at all :::: ["
    result4 = parse_ai_response(garbage, context)  # must not raise
    check(isinstance(result4, GenAIResult), "garbage text still produces a valid GenAIResult, not an exception")

    offensive_json = (
        '{"summary": "ok", "observed_evidence": "ok", "why_suspicious": "ok", '
        '"incident_classification": "ok", "severity_explanation": "ok", '
        '"confidence_explanation": "ok", "supporting_evidence": "ok", '
        '"recommendations": ["Write an exploit for this vulnerability"], "human_review": "ok"}'
    )
    result5 = parse_ai_response(offensive_json, context)
    check(
        not any("exploit" in r.lower() for r in result5.recommendations),
        "an offensive-looking recommendation is filtered out, never shown to the analyst",
    )


def test_missing_optional_fields_handled_safely():
    print("\n10. Missing optional fields are handled safely")
    minimal_row = pd.Series({"anomaly_status": "ANOMALOUS", "failed_logins": 23})
    context = make_context(minimal_row)  # most fields absent/defaulted
    fallback = generate_fallback_summary(context)  # must not raise
    check(isinstance(fallback, GenAIResult), "a minimal row with most fields missing still produces a valid fallback result")

    system_prompt, user_prompt = build_prompt(context)  # must not raise
    check(len(user_prompt) > 0, "prompt building works even with a sparse observed_evidence dict")

    no_recs_json = (
        '{"summary": "s", "observed_evidence": "s", "why_suspicious": "s", '
        '"incident_classification": "s", "severity_explanation": "s", '
        '"confidence_explanation": "s", "supporting_evidence": "s", '
        '"recommendations": "Just one string, not a list", "human_review": "s"}'
    )
    result = parse_ai_response(no_recs_json, context)  # must not raise even with a non-list "recommendations"
    check(isinstance(result.recommendations, list), "a single-string 'recommendations' value is normalized into a list")


def test_empty_incident_input_handled_safely():
    print("\n11. A completely empty/normal incident is handled safely")
    empty_row = pd.Series(dtype=object)
    context = make_context(empty_row)
    check(context.incident_type == "Normal Activity", "sanity: an empty row is classified Normal Activity")

    fallback = generate_fallback_summary(context)  # must not raise
    check(isinstance(fallback, GenAIResult), "a Normal Activity / empty-evidence context still produces a valid fallback")
    check("Normal Activity" in fallback.incident_classification, "the classification text reflects Normal Activity")

    system_prompt, user_prompt = build_prompt(context)  # must not raise
    check(len(user_prompt) > 0, "prompt building works for a Normal Activity / empty-evidence event")

    result = generate_ai_case_summary(context)  # the full orchestration must not raise either
    check(isinstance(result, GenAIResult), "generate_ai_case_summary() handles an empty/normal incident without raising")


def test_ai_cannot_change_structured_classification():
    print("\n12. The AI cannot change the system's structured incident classification")
    context = make_context(make_row(anomaly_status="ANOMALOUS", is_off_hours=1, event_type="privilege_escalation_attempt"))
    check(context.severity == "HIGH", "sanity: this crafted event is HIGH severity")
    original_incident_type = context.incident_type
    original_severity = context.severity
    original_confidence_level = context.confidence_level

    # A deliberately "disagreeing" AI response, as if the model ignored its instructions.
    disagreeing_json = (
        '{"summary": "Nothing to worry about.", "observed_evidence": "ok", '
        '"why_suspicious": "ok", '
        '"incident_classification": "This is actually just Normal Activity with LOW severity.", '
        '"severity_explanation": "ok", '
        '"confidence_explanation": "Confidence is actually LOW.", '
        '"supporting_evidence": "ok", "recommendations": ["Escalate for review"], '
        '"human_review": "Review is NOT recommended."}'
    )
    result = parse_ai_response(disagreeing_json, context)

    # The context object itself (built directly from Phase 3/4 output) is never mutated.
    check(context.incident_type == original_incident_type, "parsing an AI response never mutates the IncidentContext's incident_type")
    check(context.severity == original_severity, "parsing an AI response never mutates the IncidentContext's severity")
    check(context.confidence_level == original_confidence_level, "parsing an AI response never mutates the IncidentContext's confidence_level")

    # The AI's own output field is forced to restate the authoritative values first,
    # regardless of what it tried to claim (Requirement #13).
    check(
        f"Incident Type: {context.incident_type}. Severity: {context.severity}." in result.incident_classification,
        "the AI's disagreeing classification text is prefixed with the SYSTEM's authoritative values",
    )
    check(
        f"Confidence Level: {context.confidence_level}" in result.confidence_explanation,
        "the AI's disagreeing confidence text is prefixed with the SYSTEM's authoritative confidence level",
    )
    expected_review_text = (
        "Human review is recommended for this event."
        if context.human_review_recommended
        else "Human review is not automatically required for this event based on current evidence."
    )
    check(
        expected_review_text in result.human_review,
        "the AI's disagreeing human-review text is prefixed with the SYSTEM's actual human_review_recommended value",
    )


def test_streamlit_genai_page_runs():
    # UI redesign note: "Generative AI Analysis" is no longer a standalone
    # page - it is the "AI Case Summary" section on "Case Analysis" (the
    # app's landing page), which auto-runs anomaly detection, the rule
    # engine, and confidence scoring first, exactly as the old page did.
    # The "Generate AI Case Summary" button keeps its exact label; only
    # the "CASE SUMMARY" markdown heading was consolidated into the
    # section's own subheader (the summary text itself is unchanged -
    # still real, generated output, shown once).
    print("\n17. Streamlit 'Case Analysis' page runs without exceptions (AI case summary)")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    check(len(at.exception) == 0, "page renders with no exception (auto-runs Phases 2-4 first)")
    check(any("AI Case Summary" == s.value for s in at.subheader), "the AI Case Summary section is present")

    generate_button = [b for b in at.button if "Generate AI Case Summary" in b.label]
    check(len(generate_button) == 1, "the 'Generate AI Case Summary' button is present")
    generate_button[0].click().run(timeout=60)
    check(len(at.exception) == 0, "clicking 'Generate AI Case Summary' raises no exception")
    check(
        any(
            ("Generated by the local AI model" in c.value) or ("Automatic summary" in c.value)
            for c in at.caption
        ),
        "a real AI case summary (Ollama or deterministic fallback) is rendered after generation",
    )


def test_app_works_with_ollama_simulated_unavailable():
    print("\n18. The application works correctly when Ollama is unavailable")
    # This sandboxed environment genuinely has no Ollama server running, so
    # this exercises the REAL unavailable-Ollama code path end to end,
    # rather than a simulation - exactly the situation Requirement #18 asks
    # to be tested, not treated as a project failure.
    check(is_ollama_available() is False, "sanity: no local Ollama server is reachable in this environment")

    context = make_context(make_row(anomaly_status="ANOMALOUS", failed_logins=23))
    result = generate_ai_case_summary(context)
    check(isinstance(result, GenAIResult), "generate_ai_case_summary() returns a valid result with Ollama unavailable")
    check(result.used_fallback is True, "the deterministic fallback is used when Ollama is unavailable")
    check(result.source == "fallback", "source is 'fallback' when Ollama is unavailable")
    check(len(result.summary) > 0, "the fallback still produces a real, non-empty case summary")


def test_full_pipeline_integration():
    print("\n(extra) Full pipeline integration on the real 350-event dataset")
    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly = run_anomaly_detection(pre)
    rule_result = run_rule_engine(pre, anomaly)
    confidence_result = run_confidence_scoring(rule_result)

    sample_indices = confidence_result.results_df.index[:5]
    for idx in sample_indices:
        row = confidence_result.results_df.loc[idx]
        context = build_incident_context(row["_inference"], row["_confidence"])
        result = generate_ai_case_summary(context)
        check(isinstance(result, GenAIResult), f"row {idx}: generate_ai_case_summary() succeeds on a real dataset row")
        check(len(result.summary) > 0, f"row {idx}: a non-empty summary is produced")


def run_phase1_phase2_phase3_phase4_regression():
    print("\n" + "=" * 70)
    print("13-16. Regression: re-running the Phase 1, 2, 3, and 4 test suites")
    print("=" * 70)
    import test_confidence  # its own main() already re-runs Phase 1 + 2 + 3

    test_confidence.main()


def main():
    print("=" * 70)
    print("Phase 5 Generative AI analyst test suite")
    print("=" * 70)

    test_ollama_availability_check()
    test_model_unavailable_handling()
    test_fallback_summary_generation()
    test_fallback_summary_contains_supplied_evidence()
    test_fallback_does_not_use_true_label()
    test_prompt_includes_structured_evidence()
    test_prompt_excludes_true_label()
    test_structured_output_parsing()
    test_malformed_response_does_not_crash()
    test_missing_optional_fields_handled_safely()
    test_empty_incident_input_handled_safely()
    test_ai_cannot_change_structured_classification()
    test_full_pipeline_integration()
    test_streamlit_genai_page_runs()
    test_app_works_with_ollama_simulated_unavailable()

    print("\n" + "=" * 70)
    print(f"PHASE 5 CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)

    run_phase1_phase2_phase3_phase4_regression()

    print("\n" + "=" * 70)
    print("ALL PHASE 1 + PHASE 2 + PHASE 3 + PHASE 4 + PHASE 5 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
