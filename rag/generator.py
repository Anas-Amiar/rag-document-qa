"""
Mock answer generator.

In production this would pass the question + retrieved chunks to an LLM
(e.g. GPT-4o with a system prompt like "Answer using only the context below").
In mock mode we use extractive QA: find the sentence in the top retrieved
chunk that has the highest word-overlap with the question, then combine
it with the next sentence for a natural-sounding two-sentence answer.

This is honest: the answer is genuinely derived from the retrieved text,
so the citation verifier has real signal to work with.
"""

import re
from rag.models import RetrievedChunk, GeneratedAnswer
from rag.embedder import embed, cosine_similarity

# If the best chunk's similarity is below this, we admit we can't answer.
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.05


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _best_sentence_index(question: str, sentences: list[str]) -> int:
    q_vec = embed(question)
    scores = [cosine_similarity(q_vec, embed(s)) for s in sentences]
    return scores.index(max(scores))


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    use_mock: bool = True,
) -> GeneratedAnswer:
    if not retrieved or retrieved[0].similarity_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return GeneratedAnswer(
            text="I could not find relevant information in the knowledge base to answer this question.",
            cited_chunk_ids=[],
            confidence=0.0,
        )

    if use_mock:
        # Use the top 2 chunks as context; extract the best matching sentence(s).
        top_chunks = retrieved[:2]
        context_text = " ".join(c.chunk.text for c in top_chunks)
        sentences = _sentences(context_text)

        if not sentences:
            answer_text = context_text[:300]
        else:
            best_idx = _best_sentence_index(question, sentences)
            # Return the best sentence + the next one (for context), capped at 2.
            selected = sentences[best_idx: best_idx + 2]
            answer_text = " ".join(selected)

        return GeneratedAnswer(
            text=answer_text,
            cited_chunk_ids=[c.chunk.chunk_id for c in top_chunks],
            confidence=round(retrieved[0].similarity_score, 4),
        )
    else:
        # Real mode: call an LLM with the retrieved context.
        # Stub — wire in openai.chat.completions.create() here.
        raise NotImplementedError("Real LLM mode not yet wired up.")


if __name__ == "__main__":
    from data.documents import KNOWLEDGE_BASE
    from rag.retriever import build_store

    store = build_store(KNOWLEDGE_BASE)

    questions = [
        "How much does the Growth plan cost?",
        "How do I revoke a compromised API key?",
        "What is the webhook retry policy?",
        "Does Acme support GraphQL?",  # not in KB — should get low confidence
    ]
    for q in questions:
        results = store.retrieve(q, top_k=3)
        answer = generate_answer(q, results)
        print(f"Q: {q}")
        print(f"A: {answer.text}")
        print(f"   confidence={answer.confidence}  cited={answer.cited_chunk_ids}")
        print()
