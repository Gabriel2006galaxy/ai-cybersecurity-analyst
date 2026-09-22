"""
test_retriever.py
------------------
A small, dependency-free validation script for the Phase 8 RAG / Knowledge
Base retrieval layer (src/retriever.py, plus its integration into
src/genai_analyzer.py). Same style as test_genai.py, test_feedback.py, and
every other test_*.py in this project: plain asserts, no pytest.

This file also folds in a full regression run of the Phase 1-7 test suites
(test_feedback.py's own main() already re-runs Phase 1 through 6), so a
single command verifies the whole pipeline still works after adding
Phase 8:

    python test_retriever.py

IMPORTANT: this sandboxed test environment does NOT run a local Ollama
server, so `generate_ai_case_summary()` below exercises the DETERMINISTIC
FALLBACK path - exactly the path Phase 8's Requirement #4 says must keep
working ("If Ollama is unavailable... retrieved knowledge should still be
displayed").

Never modifies data/synthetic_security_logs.csv, and never writes to
knowledge_base/ - every test either reads the real knowledge base
read-only or points the retriever at a throwaway temp directory.
"""

import inspect
import json
import os
import shutil
import tempfile

import pandas as pd

import config
from src.confidence import score_confidence
from src.genai_analyzer import (
    OFFENSIVE_KEYWORDS,
    build_incident_context,
    generate_ai_case_summary,
    generate_fallback_summary,
)
from src.rule_engine import run_inference
from src.retriever import (
    KnowledgeEntry,
    RetrievalResult,
    RetrievedKnowledgeItem,
    build_retrieval_query,
    load_knowledge_base,
    retrieve,
    retrieve_knowledge_for_incident,
    retrieved_items_to_dicts,
)

PASSED = 0
_TEMP_DIRS = []


def check(condition: bool, message: str) -> None:
    global PASSED
    assert condition, f"FAILED: {message}"
    PASSED += 1
    print(f"  [ok] {message}")


def fresh_temp_dir() -> str:
    """A brand-new empty temp directory, cleaned up at the end of main()."""
    path = tempfile.mkdtemp(prefix="phase8_test_kb_")
    _TEMP_DIRS.append(path)
    return path


def write_kb_entry(directory: str, filename: str, **fields) -> None:
    with open(os.path.join(directory, filename), "w", encoding="utf-8") as fh:
        json.dump(fields, fh)


def cleanup_temp_dirs() -> None:
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)


# Same baseline event style used by test_feedback.py / test_genai.py.
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


def make_pair(**overrides):
    data = dict(BASE_ROW)
    data.update(overrides)
    row = pd.Series(data)
    inf = run_inference(row)
    conf = score_confidence(inf, anomaly_score_normalized=0.6)
    return inf, conf


# =======================================================================
# 1. Knowledge base loading
# =======================================================================
def test_knowledge_base_loads_successfully():
    print("\n1. The real knowledge base loads successfully")
    entries = load_knowledge_base()
    check(isinstance(entries, list), "load_knowledge_base() returns a list")
    check(len(entries) >= 10, f"at least 10 knowledge entries are present (found {len(entries)})")
    for entry in entries:
        check(isinstance(entry, KnowledgeEntry), f"{entry.knowledge_id}: is a KnowledgeEntry")
        check(bool(entry.knowledge_id), f"{entry.knowledge_id}: has a non-empty knowledge_id")
        check(bool(entry.title), f"{entry.knowledge_id}: has a non-empty title")
        check(bool(entry.topic), f"{entry.knowledge_id}: has a non-empty topic")
        check(bool(entry.content) and len(entry.content) > 40, f"{entry.knowledge_id}: has substantial content")
        check(bool(entry.source_type), f"{entry.knowledge_id}: has a non-empty source_type")
        check(isinstance(entry.incident_types, list), f"{entry.knowledge_id}: incident_types is a list")


def test_expected_knowledge_entries_exist():
    print("\n2. Every required topic area is covered by the knowledge base")
    entries = load_knowledge_base()
    combined_text = " ".join(f"{e.title} {e.topic} {e.content}".lower() for e in entries)
    required_keywords = [
        "brute-force", "network activity", "port scanning", "data transfer",
        "login location", "privilege activity", "device", "triage", "incident response",
        "logging",
    ]
    for keyword in required_keywords:
        check(keyword in combined_text, f"a knowledge entry mentions '{keyword}'")
    ids = {e.knowledge_id for e in entries}
    check(len(ids) == len(entries), "every knowledge_id is unique")


# =======================================================================
# 2. Determinism
# =======================================================================
def test_retrieval_is_deterministic():
    print("\n3. Retrieval is fully deterministic across repeated calls")
    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    r1 = retrieve_knowledge_for_incident(inf, conf)
    r2 = retrieve_knowledge_for_incident(inf, conf)
    ids1 = [item.knowledge_id for item in r1.items]
    ids2 = [item.knowledge_id for item in r2.items]
    scores1 = [item.similarity_score for item in r1.items]
    scores2 = [item.similarity_score for item in r2.items]
    check(ids1 == ids2 and len(ids1) > 0, "the same incident retrieves the same knowledge IDs, in the same order, twice")
    check(scores1 == scores2, "similarity scores are bit-for-bit identical across repeated calls")

    entries1 = load_knowledge_base()
    entries2 = load_knowledge_base()
    check(
        [e.knowledge_id for e in entries1] == [e.knowledge_id for e in entries2],
        "load_knowledge_base() returns entries in the same order every time",
    )


# =======================================================================
# 3. Relevance
# =======================================================================
def test_relevant_query_returns_relevant_topic():
    print("\n4. A relevant query returns the topically-matching knowledge entry first")
    scenarios = [
        (dict(anomaly_status="ANOMALOUS", failed_logins=23), "KB-001"),
        (dict(anomaly_status="ANOMALOUS", event_type="network_scan_detected", request_count=500, port=31337), "KB-003"),
        (dict(anomaly_status="ANOMALOUS", data_transferred_mb=2000), "KB-004"),
        (dict(anomaly_status="ANOMALOUS", location_mismatch=1), "KB-005"),
        (dict(anomaly_status="ANOMALOUS", is_off_hours=1, event_type="privilege_escalation_attempt"), "KB-006"),
        (dict(anomaly_status="ANOMALOUS", device_count=6), "KB-007"),
    ]
    for overrides, expected_top_id in scenarios:
        inf, conf = make_pair(**overrides)
        result = retrieve_knowledge_for_incident(inf, conf)
        check(len(result.items) > 0, f"{inf.incident_type}: at least one knowledge entry is retrieved")
        check(
            result.items[0].knowledge_id == expected_top_id,
            f"{inf.incident_type}: top retrieved entry is {expected_top_id} (got {result.items[0].knowledge_id})",
        )


# =======================================================================
# 4. Top-K behavior
# =======================================================================
def test_top_k_behavior():
    print("\n5. top_k correctly limits the number of retrieved entries")
    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    query = build_retrieval_query(inf, conf)

    result_default = retrieve(query)
    check(len(result_default.items) <= config.RAG_TOP_K, f"default retrieve() respects config.RAG_TOP_K ({config.RAG_TOP_K})")

    result_1 = retrieve(query, top_k=1, min_similarity=0.0)
    check(len(result_1.items) == 1, "top_k=1 returns exactly one entry")

    result_all = retrieve(query, top_k=100, min_similarity=0.0)
    check(len(result_all.items) == result_all.total_candidates, "a large top_k returns every candidate entry")

    scores = [item.similarity_score for item in result_all.items]
    check(scores == sorted(scores, reverse=True), "retrieved items are sorted by similarity score, descending")


# =======================================================================
# 5. Minimum similarity threshold
# =======================================================================
def test_minimum_similarity_threshold():
    print("\n6. min_similarity correctly filters out weak matches")
    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    query = build_retrieval_query(inf, conf)

    result_strict = retrieve(query, min_similarity=0.999)
    check(len(result_strict.items) == 0, "an unreachably high min_similarity threshold yields zero results")
    check(len(result_strict.warnings) > 0, "a threshold that filters out every candidate produces an explanatory warning")

    result_lenient = retrieve(query, min_similarity=0.0)
    check(len(result_lenient.items) > 0, "a min_similarity of 0.0 yields at least one result")


# =======================================================================
# 6. Empty / invalid query handling
# =======================================================================
def test_empty_invalid_query_handling():
    print("\n7. Empty/invalid queries are handled safely, never crash")
    for bad_query in ["", "   ", None]:
        result = retrieve(bad_query)
        check(isinstance(result, RetrievalResult), f"retrieve({bad_query!r}) returns a RetrievalResult")
        check(result.items == [], f"retrieve({bad_query!r}) returns zero items")

    # A query made entirely of English stopwords also produces no usable
    # TF-IDF terms - must degrade gracefully, not raise.
    result = retrieve("the a an of and")
    check(isinstance(result, RetrievalResult), "an all-stopword query returns a RetrievalResult, not an exception")


# =======================================================================
# 7. Missing knowledge base handling
# =======================================================================
def test_missing_knowledge_base_handling():
    print("\n8. A missing or empty knowledge base is handled safely, never crashes")
    result = retrieve("brute force login activity", knowledge_base_dir="/tmp/definitely_missing_kb_dir_xyz")
    check(result.items == [], "a nonexistent knowledge base directory yields zero items")
    check(result.total_candidates == 0, "a nonexistent knowledge base directory reports zero candidates")

    empty_dir = fresh_temp_dir()
    result_empty = retrieve("brute force login activity", knowledge_base_dir=empty_dir)
    check(result_empty.items == [], "an empty knowledge base directory yields zero items")

    entries_from_file = load_knowledge_base(knowledge_base_dir=__file__)  # a file, not a directory
    check(entries_from_file == [], "passing a file path instead of a directory is handled safely")


def test_malformed_entries_are_skipped_not_fatal():
    print("\n8b. Malformed knowledge-base files are skipped without crashing the whole load")
    kb_dir = fresh_temp_dir()
    write_kb_entry(kb_dir, "good.json", knowledge_id="KB-T1", title="Test Entry", topic="Testing",
                   incident_types=["General"], content="A perfectly valid test knowledge entry about testing.",
                   source_type="test")
    with open(os.path.join(kb_dir, "broken.json"), "w", encoding="utf-8") as fh:
        fh.write("{ this is not valid json")
    write_kb_entry(kb_dir, "missing_field.json", knowledge_id="KB-T2", title="Incomplete")  # missing required fields

    entries = load_knowledge_base(knowledge_base_dir=kb_dir)
    check(len(entries) == 1, "only the well-formed entry is loaded; malformed/incomplete files are skipped")
    check(entries[0].knowledge_id == "KB-T1", "the correctly-loaded entry has the expected knowledge_id")


# =======================================================================
# 8. Similarity score validity
# =======================================================================
def test_similarity_scores_are_valid():
    print("\n9. Similarity scores are always valid numbers in [0, 1]")
    for overrides in [
        dict(anomaly_status="ANOMALOUS", failed_logins=23),
        dict(anomaly_status="ANOMALOUS", data_transferred_mb=2000),
        dict(anomaly_status="NORMAL"),
    ]:
        inf, conf = make_pair(**overrides)
        result = retrieve_knowledge_for_incident(inf, conf, min_similarity=0.0)
        for item in result.items:
            check(isinstance(item.similarity_score, float), f"{item.knowledge_id}: similarity_score is a float")
            check(0.0 <= item.similarity_score <= 1.0, f"{item.knowledge_id}: similarity_score {item.similarity_score} is within [0, 1]")


# =======================================================================
# 9. true_label is never used
# =======================================================================
def test_true_label_not_used_in_retrieval():
    print("\n10. true_label is never read or used anywhere in retrieval")
    import dataclasses

    import src.retriever as retriever_module

    check("LABEL_COLUMN" not in vars(retriever_module), "config.LABEL_COLUMN is not imported into retriever.py's namespace")

    entry_fields = {f.name for f in dataclasses.fields(KnowledgeEntry)}
    item_fields = {f.name for f in dataclasses.fields(RetrievedKnowledgeItem)}
    check("true_label" not in entry_fields, "KnowledgeEntry has no true_label field")
    check("true_label" not in item_fields, "RetrievedKnowledgeItem has no true_label field")

    query_params = list(inspect.signature(build_retrieval_query).parameters)
    check(
        query_params == ["inference_record", "confidence_record"],
        "build_retrieval_query() only accepts InferenceRecord/ConfidenceRecord - no raw row, no label",
    )

    # Behavioral check: two events identical except for true_label retrieve identically.
    inf_a, conf_a = make_pair(anomaly_status="ANOMALOUS", failed_logins=23, true_label="suspicious")
    inf_b, conf_b = make_pair(anomaly_status="ANOMALOUS", failed_logins=23, true_label="normal")
    result_a = retrieve_knowledge_for_incident(inf_a, conf_a)
    result_b = retrieve_knowledge_for_incident(inf_b, conf_b)
    ids_a = [item.knowledge_id for item in result_a.items]
    ids_b = [item.knowledge_id for item in result_b.items]
    check(ids_a == ids_b, "flipping true_label on an otherwise-identical event does not change retrieval results")


# =======================================================================
# 10. No offensive content in the knowledge base
# =======================================================================
def test_no_offensive_content_in_knowledge_base():
    print("\n11. No knowledge-base entry contains offensive/exploit/attack content")
    entries = load_knowledge_base()
    prohibited = list(config.PROHIBITED_PLAN_ACTION_KEYWORDS) + list(OFFENSIVE_KEYWORDS)
    for entry in entries:
        lowered = f"{entry.title} {entry.content}".lower()
        for keyword in prohibited:
            check(keyword not in lowered, f"{entry.knowledge_id}: does not contain prohibited keyword '{keyword}'")


# =======================================================================
# 11. GenAI context integration
# =======================================================================
def test_genai_context_receives_retrieved_knowledge():
    print("\n12. Retrieved knowledge correctly reaches IncidentContext and the GenAI result")
    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    retrieval = retrieve_knowledge_for_incident(inf, conf)
    check(len(retrieval.items) > 0, "the brute-force scenario retrieves at least one knowledge entry")

    context = build_incident_context(
        inf, conf,
        retrieved_knowledge=[item.content for item in retrieval.items],
        retrieved_knowledge_items=retrieved_items_to_dicts(retrieval.items),
    )
    check(len(context.retrieved_knowledge) == len(retrieval.items), "IncidentContext.retrieved_knowledge has one entry per retrieved item")
    check(len(context.retrieved_knowledge_items) == len(retrieval.items), "IncidentContext.retrieved_knowledge_items has one entry per retrieved item")
    check(context.retrieved_knowledge_items[0]["knowledge_id"] == retrieval.items[0].knowledge_id, "structured retrieval metadata is preserved on the context")

    result = generate_fallback_summary(context)
    check(result.retrieved_knowledge_count == len(retrieval.items), "GenAIResult.retrieved_knowledge_count matches what was retrieved")
    check(result.retrieved_knowledge_ids == [item.knowledge_id for item in retrieval.items], "GenAIResult.retrieved_knowledge_ids matches the retrieved IDs")
    check(result.retrieved_knowledge_titles == [item.title for item in retrieval.items], "GenAIResult.retrieved_knowledge_titles matches the retrieved titles")
    check(result.retrieved_knowledge_scores == [item.similarity_score for item in retrieval.items], "GenAIResult.retrieved_knowledge_scores matches the retrieved scores")
    check(result.retrieved_knowledge_snippets == [item.content for item in retrieval.items], "GenAIResult.retrieved_knowledge_snippets matches the retrieved content")


def test_context_without_retrieval_still_works():
    print("\n12b. A context with no retrieved knowledge attached still produces a valid, empty-RAG result")
    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    context = build_incident_context(inf, conf)  # no retrieval passed at all - Phase 5/7 call style
    check(context.retrieved_knowledge == [], "retrieved_knowledge defaults to an empty list")
    check(context.retrieved_knowledge_items == [], "retrieved_knowledge_items defaults to an empty list")
    result = generate_fallback_summary(context)
    check(result.retrieved_knowledge_count == 0, "GenAIResult.retrieved_knowledge_count is 0 when nothing was retrieved")
    check(result.retrieved_knowledge_ids == [], "GenAIResult.retrieved_knowledge_ids is empty when nothing was retrieved")


# =======================================================================
# 12. Fallback still works without Ollama
# =======================================================================
def test_existing_fallback_still_works_without_ollama():
    print("\n13. The deterministic fallback still works end-to-end, with retrieval attached")
    inf, conf = make_pair(anomaly_status="ANOMALOUS", failed_logins=23)
    retrieval = retrieve_knowledge_for_incident(inf, conf)
    context = build_incident_context(
        inf, conf,
        retrieved_knowledge=[item.content for item in retrieval.items],
        retrieved_knowledge_items=retrieved_items_to_dicts(retrieval.items),
    )
    result = generate_ai_case_summary(context, model_name="definitely-not-installed-model")
    check(result.used_fallback is True, "with no Ollama server running, the deterministic fallback is used")
    check(len(result.summary) > 0, "the fallback still produces a non-empty summary even with RAG context attached")
    check(result.retrieved_knowledge_count == len(retrieval.items), "the fallback path still carries retrieval metadata for the UI")


# =======================================================================
# 13-15. Authoritative values remain unchanged
# =======================================================================
def test_authoritative_values_remain_unchanged():
    print("\n14-16. Authoritative incident_type/severity/confidence are never altered by RAG")
    scenarios = [
        dict(anomaly_status="ANOMALOUS", failed_logins=23),
        dict(anomaly_status="ANOMALOUS", data_transferred_mb=2000),
        dict(anomaly_status="NORMAL"),
    ]
    for overrides in scenarios:
        inf, conf = make_pair(**overrides)
        retrieval = retrieve_knowledge_for_incident(inf, conf)
        context = build_incident_context(
            inf, conf,
            retrieved_knowledge=[item.content for item in retrieval.items],
            retrieved_knowledge_items=retrieved_items_to_dicts(retrieval.items),
        )
        result = generate_ai_case_summary(context)

        check(
            f"Incident Type: {inf.incident_type}." in result.incident_classification,
            f"{inf.incident_type}: authoritative incident_type is restated verbatim regardless of retrieved knowledge",
        )
        check(
            f"Severity: {inf.severity}." in result.incident_classification,
            f"{inf.incident_type}: authoritative severity is restated verbatim regardless of retrieved knowledge",
        )
        # The exact phrasing differs between the Ollama-finalized path
        # ("Confidence Level: MEDIUM (score 0.53).") and the deterministic
        # fallback ("Confidence level is 'MEDIUM' ...") - see
        # src/genai_analyzer.py's `_finalize_result()` vs.
        # `_fallback_confidence_explanation()`. Both always contain the
        # authoritative confidence_level string verbatim, which is the
        # actual invariant RAG must never break.
        check(
            conf.confidence_level in result.confidence_explanation,
            f"{inf.incident_type}: authoritative confidence level ({conf.confidence_level}) is present verbatim regardless of retrieved knowledge",
        )
        # The underlying Phase 3/4 records themselves must be completely untouched.
        check(inf.incident_type == run_inference(pd.Series({**BASE_ROW, **overrides})).incident_type,
              f"{inf.incident_type}: retrieval never mutates the InferenceRecord's incident_type")


# =======================================================================
# Extra: no heavy dependency, real dataset integration, UI smoke test
# =======================================================================
def test_no_heavy_dependency_introduced():
    print("\n17. No chromadb / heavy vector-database dependency is used by the retriever")
    import sys

    import src.retriever as retriever_module

    check("chromadb" not in sys.modules, "chromadb is not imported/loaded anywhere by this process")
    module_source_file = inspect.getsourcefile(retriever_module) or ""
    with open(module_source_file, encoding="utf-8") as fh:
        import_lines = [line for line in fh if line.strip().startswith(("import ", "from "))]
    check(
        not any("chromadb" in line.lower() or "faiss" in line.lower() for line in import_lines),
        "src/retriever.py's import statements reference neither chromadb nor faiss",
    )
    with open("requirements.txt", encoding="utf-8") as fh:
        requirements_text = fh.read()
    check("chromadb" not in requirements_text.lower(), "chromadb has been removed from requirements.txt")


def test_full_pipeline_integration():
    print("\n18. Full pipeline integration on the real 350-event dataset")
    from src.anomaly_detector import run_anomaly_detection
    from src.confidence import run_confidence_scoring
    from src.data_loader import load_logs
    from src.preprocessor import run_preprocessing_pipeline
    from src.rule_engine import run_rule_engine

    df = load_logs()
    pre = run_preprocessing_pipeline(df)
    anomaly = run_anomaly_detection(pre)
    rule_result = run_rule_engine(pre, anomaly)
    confidence_result = run_confidence_scoring(rule_result)

    sample_indices = confidence_result.results_df.index[:8]
    for idx in sample_indices:
        row = confidence_result.results_df.loc[idx]
        retrieval = retrieve_knowledge_for_incident(row["_inference"], row["_confidence"])
        check(isinstance(retrieval, RetrievalResult), f"row {idx}: retrieval succeeds on a real dataset row")
        for item in retrieval.items:
            check(isinstance(item, RetrievedKnowledgeItem), f"row {idx}: retrieved item is well-typed")


def test_streamlit_genai_page_shows_retrieved_knowledge():
    # UI redesign note: "Generative AI Analysis" is no longer a standalone
    # page - RAG retrieval now runs automatically for the selected case on
    # "Case Analysis" (the app's landing page), and its section is now
    # named "Knowledge Base Used by AI" (a plain-English label; the
    # underlying retrieval call and its real results are unchanged).
    print("\n19. The Case Analysis Streamlit page loads and shows the knowledge base used by the AI")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=90)
    check(not at.exception, "the app loads on the Case Analysis (landing) page with no exception")

    subheaders = [h.value for h in at.subheader]
    check(any("Knowledge Base Used by AI" in s for s in subheaders), "the 'Knowledge Base Used by AI' section is present on the page")

    generate_buttons = [b for b in at.button if "Generate AI Case Summary" in (b.label or "")]
    check(len(generate_buttons) == 1, "the Generate AI Case Summary button is present")
    generate_buttons[0].click().run(timeout=90)
    check(not at.exception, "clicking Generate AI Case Summary with retrieval attached raises no exception")


# =======================================================================
# Phase 1-7 regression
# =======================================================================
def run_phase1_through_phase7_regression():
    print("\n" + "=" * 70)
    print("20. Regression: re-running the Phase 1-7 test suites")
    print("=" * 70)
    import test_feedback  # its own main() already re-runs Phase 1 through 6

    test_feedback.main()


def main():
    print("=" * 70)
    print("Phase 8 RAG / Knowledge Base test suite")
    print("=" * 70)

    try:
        test_knowledge_base_loads_successfully()
        test_expected_knowledge_entries_exist()
        test_retrieval_is_deterministic()
        test_relevant_query_returns_relevant_topic()
        test_top_k_behavior()
        test_minimum_similarity_threshold()
        test_empty_invalid_query_handling()
        test_missing_knowledge_base_handling()
        test_malformed_entries_are_skipped_not_fatal()
        test_similarity_scores_are_valid()
        test_true_label_not_used_in_retrieval()
        test_no_offensive_content_in_knowledge_base()
        test_genai_context_receives_retrieved_knowledge()
        test_context_without_retrieval_still_works()
        test_existing_fallback_still_works_without_ollama()
        test_authoritative_values_remain_unchanged()
        test_no_heavy_dependency_introduced()
        test_full_pipeline_integration()
        test_streamlit_genai_page_shows_retrieved_knowledge()
    finally:
        cleanup_temp_dirs()

    print("\n" + "=" * 70)
    print(f"PHASE 8 CHECKS PASSED ({PASSED} checks)")
    print("=" * 70)

    run_phase1_through_phase7_regression()

    print("\n" + "=" * 70)
    print("ALL PHASE 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
