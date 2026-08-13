"""
HTTP-layer tests: answers carry their grounding verdict and citations, and
questions outside the knowledge base are flagged rather than answered.
"""

from fastapi.testclient import TestClient

from rag.app import app

client = TestClient(app)


def test_in_kb_question_answered_with_citations():
    r = client.post("/v1/ask", json={"question": "How do I connect Jira to Acme?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["citation_check"]["is_grounded"] is True
    assert body["cited_chunks"]


def test_out_of_kb_topic_flagged():
    r = client.post("/v1/ask", json={"question": "Does Acme support GraphQL queries?"})
    body = r.json()
    assert body["status"] == "low_confidence"
    assert body["citation_check"]["is_grounded"] is False


def test_off_topic_cannot_answer():
    r = client.post("/v1/ask", json={"question": "What is the capital of France?"})
    body = r.json()
    assert body["status"] == "cannot_answer"
    assert body["cited_chunks"] == []


def test_documents_and_health():
    docs = client.get("/documents").json()
    assert docs["count"] == 6
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["chunks"] > 6
    assert client.get("/").status_code == 200
