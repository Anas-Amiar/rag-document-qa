"""
Behavioural tests for the RAG pipeline. The mock embedder + extractive generator
are fully deterministic, so exact statuses are asserted. The important property
under test is honesty: unanswerable questions must NOT come back as confident
answers.
"""

import pytest

from rag.retriever import build_store, VectorStore
from rag.chunker import split_into_chunks
from rag.pipeline import ask
from data.documents import KNOWLEDGE_BASE


@pytest.fixture(scope="module")
def store():
    return build_store(KNOWLEDGE_BASE)


# --- retrieval / indexing ---------------------------------------------------

def test_store_indexes_chunks():
    s = build_store(KNOWLEDGE_BASE)
    assert len(s) > len(KNOWLEDGE_BASE)      # each doc splits into several chunks


def test_retrieval_ranks_relevant_chunk_first(store):
    results = store.retrieve("How much does the Growth plan cost?", top_k=3)
    assert results
    assert results[0].chunk.doc_id == "doc_001"   # the pricing doc


def test_chunks_have_overlap_metadata():
    doc = KNOWLEDGE_BASE[0]
    chunks = split_into_chunks(doc)
    assert chunks
    assert all(c.chunk_id.startswith(doc.doc_id) for c in chunks)


# --- the honesty properties -------------------------------------------------

def test_in_kb_question_is_answered_and_grounded(store):
    r = ask("How do I connect Jira to Acme?", store)
    assert r.status == "answered"
    assert r.citation_check.is_grounded is True
    assert r.cited_chunks                       # answer carries its sources


def test_pricing_question_is_answered(store):
    r = ask("How much does the Growth plan cost?", store)
    assert r.status == "answered"
    assert r.citation_check.is_grounded is True


def test_out_of_kb_topic_is_flagged_not_hallucinated(store):
    # GraphQL isn't in the KB — the grounding check must catch it.
    r = ask("Does Acme support GraphQL queries?", store)
    assert r.status == "low_confidence"
    assert r.citation_check.is_grounded is False


def test_off_topic_question_cannot_be_answered(store):
    r = ask("What is the capital of France?", store)
    assert r.status == "cannot_answer"
    assert r.cited_chunks == []
    assert r.citation_check.is_grounded is False


def test_answered_status_always_carries_citations(store):
    for q in ["How do I authenticate API requests?",
              "What is the support response time for Enterprise customers?"]:
        r = ask(q, store)
        if r.status == "answered":
            assert r.cited_chunks, f"answered question left uncited: {q}"
