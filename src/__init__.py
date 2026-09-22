"""
src package
-----------
Core logic for the AI-Based Cybersecurity Analyst Assistant.

Modules so far:

    src/data_loader.py     (Phase 0) - CSV loading + basic stats
    src/preprocessor.py    (Phase 1) - validation, cleaning, feature
                                        engineering, encoding/scaling
    src/anomaly_detector.py (Phase 2) - Isolation Forest training,
                                        prediction, evaluation, and an
                                        optional SOM clustering visual
    src/rule_engine.py     (Phase 3) - facts, rules, inference, incident
                                        classification, initial severity
    src/confidence.py      (Phase 4) - evidence-based confidence scoring,
                                        confidence levels, uncertainty
                                        explanations, human-review flag
    src/genai_analyzer.py  (Phase 5) - local Ollama integration, prompt
                                        construction, response parsing,
                                        and a deterministic evidence-based
                                        fallback for the AI case summary
    src/response_planner.py (Phase 6) - conditional defensive response
                                        planning, plan validation - never
                                        executes any action against a
                                        real system
    src/feedback.py         (Phase 7) - human analyst review persistence
                                        (SQLite): approve/reject/modify
                                        decisions, stable incident IDs,
                                        and descriptive feedback insights
    src/retriever.py        (Phase 8) - lightweight local RAG: loads
                                        knowledge_base/, builds TF-IDF
                                        vectors, ranks entries by cosine
                                        similarity to the current incident -
                                        no vector database, deterministic

Phase 7 completed every stage of the originally-planned pipeline.
Phase 8 (src/retriever.py) is a later, optional extension that fills in
the RAG hook src/genai_analyzer.py always left in place - it adds no new
pipeline stage of its own, only enriches the existing Generative AI step.

Keeping one module per pipeline stage keeps app.py thin and each stage
independently testable.
"""
