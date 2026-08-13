"""
HTTP layer for the RAG pipeline — ask questions against the knowledge base and
get back an answer *plus* its grounding verdict and citations.

    uvicorn rag.app:app --reload

Runs on a mock embedder + extractive generator by default (no API keys), so it
is safe to deploy as a public demo. Swap embed() (embedder.py) and generate_answer()
(generator.py) for real embedding/LLM calls to make it live.

Endpoints:
    POST /v1/ask     question -> answer, status (answered / low_confidence /
                     cannot_answer), citations, and a two-axis grounding check
    GET  /documents  the knowledge-base documents backing the answers
    GET  /health     liveness + index size
"""

from pydantic import BaseModel
from fastapi import FastAPI

from rag.models import QaResult
from rag.pipeline import ask
from rag.retriever import build_store
from data.documents import KNOWLEDGE_BASE

app = FastAPI(
    title="RAG Document QA",
    version="1.0.0",
    description="Retrieval-augmented Q&A that verifies every answer is grounded in its "
                "cited sources, and routes ungrounded answers to review instead of "
                "returning them as confident.",
)

# Index the knowledge base once at startup.
store = build_store(KNOWLEDGE_BASE)


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def root() -> dict:
    return {
        "service": "rag-document-qa",
        "documents_indexed": len(KNOWLEDGE_BASE),
        "docs": "/docs",
        "try": "POST /v1/ask with {\"question\": \"How much does the Growth plan cost?\"}",
        "note": "Ask something outside the KB (e.g. 'Does Acme support GraphQL?') to see "
                "the grounding check flag it as low_confidence instead of hallucinating.",
    }


@app.post("/v1/ask", response_model=QaResult)
def ask_question(req: AskRequest) -> QaResult:
    return ask(req.question, store, top_k=req.top_k)


@app.get("/documents")
def documents() -> dict:
    return {
        "count": len(KNOWLEDGE_BASE),
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "source": d.source}
            for d in KNOWLEDGE_BASE
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(KNOWLEDGE_BASE), "chunks": len(store)}
