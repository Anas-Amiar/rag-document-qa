"""
End-to-end RAG pipeline.

ask(question) is the single public entry point:
  1. Retrieve the top-k most relevant chunks from the vector store.
  2. Generate an answer grounded in those chunks (mock or real LLM).
  3. Verify that the answer's claims are actually traceable to the cited chunks.
  4. Return a QaResult with status: answered / low_confidence / cannot_answer.
"""

from rag.models import QaResult, CitationCheck
from rag.retriever import VectorStore
from rag.generator import generate_answer, RETRIEVAL_CONFIDENCE_THRESHOLD
from rag.verifier import verify_grounding

# Below this retrieval score, admit we can't answer rather than guess.
LOW_CONFIDENCE_THRESHOLD = 0.08


def ask(question: str, store: VectorStore, top_k: int = 3, use_mock: bool = True) -> QaResult:
    retrieved = store.retrieve(question, top_k=top_k)

    # No useful chunks found at all.
    if not retrieved or retrieved[0].similarity_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return QaResult(
            question=question,
            answer="I could not find relevant information in the knowledge base to answer this question.",
            cited_chunks=[],
            citation_check=CitationCheck(is_grounded=False, grounding_score=0.0, ungrounded_terms=[]),
            status="cannot_answer",
        )

    answer = generate_answer(question, retrieved, use_mock=use_mock)
    check = verify_grounding(answer.text, retrieved, question=question)

    if answer.confidence < LOW_CONFIDENCE_THRESHOLD:
        status = "low_confidence"
    elif not check.is_grounded:
        status = "low_confidence"
    else:
        status = "answered"

    cited = [r for r in retrieved if r.chunk.chunk_id in answer.cited_chunk_ids]

    return QaResult(
        question=question,
        answer=answer.text,
        cited_chunks=cited,
        citation_check=check,
        status=status,
    )


if __name__ == "__main__":
    from data.documents import KNOWLEDGE_BASE
    from rag.retriever import build_store

    store = build_store(KNOWLEDGE_BASE)

    questions = [
        "How much does the Growth plan cost?",
        "How do I authenticate API requests?",
        "What is the webhook retry policy if my endpoint is down?",
        "How do I connect Jira to Acme?",
        "What is the support response time for Enterprise customers?",
        "Does Acme support GraphQL queries?",          # not in KB
        "What is the capital of France?",              # completely off-topic
    ]

    for q in questions:
        result = ask(q, store)
        tag = {"answered": "OK ", "low_confidence": "LOW", "cannot_answer": "N/A"}[result.status]
        print(f"[{tag}] {q}")
        print(f"       {result.answer[:120]}")
        print(f"       grounded={result.citation_check.is_grounded}  "
              f"score={result.citation_check.grounding_score}")
        print()
