"""
retriever.py
------------
Phase 8 - lightweight Retrieval-Augmented Generation (RAG) over a small,
local, defensive cybersecurity knowledge base.

    Incident Context (Phases 2-4)
            |
      Retrieval Query                (this module)
            |
    TF-IDF + Cosine Similarity        (this module, scikit-learn only)
            |
      Top-K Knowledge Entries
            |
      Generative AI Analysis         (src/genai_analyzer.py, Phase 5)

This module implements ONLY the retrieval layer that
`src/genai_analyzer.py`'s `IncidentContext.retrieved_knowledge` hook was
always intended for (see that module's docstring). It never talks to
Ollama, never builds a prompt, and never decides an incident's
classification, severity, or confidence - it only finds and ranks a
handful of short reference documents that MIGHT be useful background for
the analyst and the GenAI explanation step.

IMPORTANT - no heavy dependencies:
    Retrieval is implemented with `sklearn.feature_extraction.text.
    TfidfVectorizer` and `sklearn.metrics.pairwise.cosine_similarity` -
    both already required by Phase 2 (Isolation Forest / StandardScaler).
    No vector database, no embeddings model, and no new package is
    introduced. `chromadb` is deliberately NOT used here.

IMPORTANT - deterministic:
    TF-IDF term counts and cosine similarity involve no randomness at
    all, so retrieval is fully reproducible for the same knowledge base
    and the same query. Knowledge-base entries are additionally loaded in
    a fixed, sorted order (by filename, then by `knowledge_id`), and
    result ties are broken by `knowledge_id` ascending, so the output
    ordering can never depend on filesystem iteration order.

IMPORTANT - true_label is never used:
    The retrieval query is built exclusively from an already-computed
    `InferenceRecord` (Phase 3) and `ConfidenceRecord` (Phase 4) - the
    same two objects `src/genai_analyzer.build_incident_context()` reads
    from. Neither of those objects is ever built from `config.LABEL_COLUMN`
    / `true_label` (see `src/rule_engine.py` and `src/confidence.py`), and
    this module never touches the raw dataframe or that column directly.

IMPORTANT - defensive-only knowledge base:
    Every entry under `knowledge_base/` is a short, hand-written,
    academic-appropriate reference document. None of it contains
    offensive instructions, exploit steps, credential-attack techniques,
    or guidance for attacking a real system (see `test_retriever.py`,
    which checks every entry against the same offensive-keyword lists
    used elsewhere in the project).

IMPORTANT - never a hard dependency:
    Every public function here degrades gracefully: a missing knowledge
    base directory, an empty directory, a malformed entry, or an empty/
    blank query all safely produce an empty `RetrievalResult` rather than
    raising. `src/genai_analyzer.py`'s Ollama path and deterministic
    fallback both continue to work whether or not any knowledge was
    retrieved (Requirement #4 of Phase 8).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KNOWLEDGE_BASE_DIR, RAG_MIN_SIMILARITY, RAG_TOP_K
from src.confidence import ConfidenceRecord
from src.rule_engine import InferenceRecord

# =======================================================================
# Data structures
# =======================================================================
REQUIRED_ENTRY_FIELDS = ["knowledge_id", "title", "topic", "content", "source_type"]


@dataclass
class KnowledgeEntry:
    """One reference document loaded from `knowledge_base/`."""

    knowledge_id: str
    title: str
    topic: str
    incident_types: List[str] = field(default_factory=list)
    content: str = ""
    source_type: str = "defensive_reference"


@dataclass
class RetrievedKnowledgeItem:
    """One knowledge entry that matched a query, plus its similarity score."""

    knowledge_id: str
    title: str
    topic: str
    content: str
    source_type: str
    similarity_score: float


@dataclass
class RetrievalResult:
    """
    The full result of one retrieval call - always safe to inspect even
    when nothing matched (`items` is simply empty, never None).
    """

    query_text: str = ""
    items: List[RetrievedKnowledgeItem] = field(default_factory=list)
    total_candidates: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)


# =======================================================================
# 1. Loading the knowledge base (Requirement #1)
# =======================================================================
def load_knowledge_base(knowledge_base_dir: Optional[str] = None) -> List[KnowledgeEntry]:
    """
    Load every `*.json` reference document from `knowledge_base_dir`
    (defaults to `config.KNOWLEDGE_BASE_DIR`), sorted by filename for a
    fully deterministic load order.

    Never raises. A missing directory, an unreadable/malformed file, or a
    file missing a required field is skipped (not fatal) - the rest of
    the knowledge base still loads normally.
    """
    directory = knowledge_base_dir or KNOWLEDGE_BASE_DIR
    entries: List[KnowledgeEntry] = []

    if not directory or not os.path.isdir(directory):
        return entries

    try:
        filenames = sorted(f for f in os.listdir(directory) if f.lower().endswith(".json"))
    except OSError:
        return entries

    for filename in filenames:
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue  # skip a malformed/unreadable file rather than crash

        if not isinstance(data, dict):
            continue
        if any(field_name not in data for field_name in REQUIRED_ENTRY_FIELDS):
            continue

        incident_types = data.get("incident_types", [])
        if not isinstance(incident_types, list):
            incident_types = [str(incident_types)]

        entries.append(
            KnowledgeEntry(
                knowledge_id=str(data["knowledge_id"]),
                title=str(data["title"]),
                topic=str(data["topic"]),
                incident_types=[str(t) for t in incident_types],
                content=str(data["content"]),
                source_type=str(data["source_type"]),
            )
        )

    # Deterministic regardless of filename scheme - sort by knowledge_id.
    entries.sort(key=lambda e: e.knowledge_id)
    return entries


# =======================================================================
# 2. Building a query from the incident context (Requirement #2, step 3)
# =======================================================================
def _humanize(token: str) -> str:
    """'high_failed_logins' -> 'high failed logins' - improves TF-IDF term
    overlap with the plain-English wording used in the knowledge base."""
    return str(token).replace("_", " ")


def build_retrieval_query(inference_record: InferenceRecord, confidence_record: ConfidenceRecord) -> str:
    """
    Build a plain-text retrieval query from ONLY already-computed Phase 3/
    4 results - never from the raw dataframe, and never from `true_label`
    (neither object has such a field to begin with).
    """
    parts: List[str] = [inference_record.incident_type, inference_record.severity]
    parts.extend(_humanize(f) for f in inference_record.derived_facts)

    for triggered_rule in inference_record.triggered_rules:
        parts.append(triggered_rule.name)
        parts.append(_humanize(triggered_rule.conclusion))
        parts.append(triggered_rule.explanation)

    parts.extend(str(indicator) for indicator in confidence_record.supporting_indicators)
    parts.append(confidence_record.confidence_level)
    if confidence_record.uncertainty_reason:
        parts.append(confidence_record.uncertainty_reason)

    return " ".join(str(p) for p in parts if p)


# =======================================================================
# 3. Core retrieval - TF-IDF + cosine similarity (Requirement #2)
# =======================================================================
def retrieve(
    query_text: str,
    knowledge_base_dir: Optional[str] = None,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> RetrievalResult:
    """
    Deterministic TF-IDF + cosine-similarity retrieval over the local
    knowledge base.

    Safe on every edge case:
      - empty/blank query           -> empty result, no crash
      - missing/empty knowledge base -> empty result, no crash
      - all scores below threshold  -> empty result, no crash
    """
    top_k = RAG_TOP_K if top_k is None else top_k
    min_similarity = RAG_MIN_SIMILARITY if min_similarity is None else min_similarity

    entries = load_knowledge_base(knowledge_base_dir)
    result = RetrievalResult(query_text=query_text or "", total_candidates=len(entries))

    if not entries:
        result.warnings.append("Knowledge base is empty or unavailable - no knowledge retrieved.")
        return result

    if not query_text or not query_text.strip():
        result.warnings.append("Empty retrieval query - no knowledge retrieved.")
        return result

    corpus = [f"{e.title} {e.topic} {e.content}" for e in entries]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        entry_matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([query_text])
    except ValueError:
        # E.g. the query/corpus contains only stopwords/empty tokens after
        # vectorization - treat as "nothing useful to match", not a crash.
        result.warnings.append("Query produced no usable terms - no knowledge retrieved.")
        return result

    similarities = cosine_similarity(query_vector, entry_matrix).flatten()

    scored = [
        (float(score), entry)
        for score, entry in zip(similarities, entries)
        if float(score) >= min_similarity
    ]
    # Deterministic ordering: highest similarity first, ties broken by
    # knowledge_id ascending so output never depends on incidental order.
    scored.sort(key=lambda pair: (-pair[0], pair[1].knowledge_id))

    top_matches = scored[: max(top_k, 0)]
    result.items = [
        RetrievedKnowledgeItem(
            knowledge_id=entry.knowledge_id,
            title=entry.title,
            topic=entry.topic,
            content=entry.content,
            source_type=entry.source_type,
            similarity_score=round(score, 4),
        )
        for score, entry in top_matches
    ]
    if not result.items:
        result.warnings.append("No knowledge-base entry met the minimum similarity threshold.")
    return result


def retrieve_knowledge_for_incident(
    inference_record: InferenceRecord,
    confidence_record: ConfidenceRecord,
    knowledge_base_dir: Optional[str] = None,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> RetrievalResult:
    """
    Convenience wrapper used by `app.py`: builds the query from the
    incident context and retrieves in one call. `Normal Activity` events
    still retrieve normally (e.g. against the general/monitoring entries)
    - retrieval never special-cases severity or incident type beyond what
    naturally falls out of the query text.
    """
    query_text = build_retrieval_query(inference_record, confidence_record)
    return retrieve(
        query_text,
        knowledge_base_dir=knowledge_base_dir,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def retrieved_items_to_dicts(items: List[RetrievedKnowledgeItem]) -> List[Dict]:
    """
    Small helper so callers (app.py) don't need to import `dataclasses`
    just to hand retrieval results to
    `src.genai_analyzer.build_incident_context()`.
    """
    return [
        {
            "knowledge_id": item.knowledge_id,
            "title": item.title,
            "topic": item.topic,
            "content": item.content,
            "source_type": item.source_type,
            "similarity_score": item.similarity_score,
        }
        for item in items
    ]
