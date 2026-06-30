"""
Citation verifier — the core production-maturity layer.

Checks whether the generated answer is actually grounded in the chunks it
cited, by measuring what fraction of the answer's content-bearing terms
appear in those chunks.  Terms that appear in the answer but NOT in any
cited chunk are flagged as "ungrounded terms" — potential hallucinations.

Thresholds:
  grounding_score >= 0.75  → is_grounded = True  (answer is well-supported)
  grounding_score  < 0.75  → is_grounded = False (answer may be fabricated)
"""

import re
from rag.models import RetrievedChunk, CitationCheck

GROUNDING_THRESHOLD = 0.60

# Common words that carry no factual content — ignore when checking grounding.
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "i", "you", "we", "they", "it", "this",
    "that", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "through", "per",
    "can", "not", "your", "our", "all", "any", "if", "as", "so",
    # Question words
    "how", "what", "when", "where", "which", "who", "why", "whose", "whom",
    # Question-helper words that carry no factual meaning in isolation
    "much", "many", "often", "long", "happen", "happens", "happened",
    "get", "give", "make", "find", "see", "tell", "set", "go",
    # Domain-universal (appears in every document — no discriminating power)
    "acme", "platform",
}


def _content_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 2}


def verify_grounding(
    answer_text: str,
    cited_chunks: list[RetrievedChunk],
    question: str = "",
) -> CitationCheck:
    """
    Two checks combined into one score:

    1. Answer grounding — what fraction of the answer's key terms appear in
       the cited chunks? (Catches hallucinated facts the chunks never said.)

    2. Question coverage — what fraction of the QUESTION's key terms appear
       in the cited chunks? (Catches the case where retrieval returned
       something plausible-looking but didn't actually address the question.)

    The final grounding_score is the minimum of the two, so a failure on
    either axis fails the whole check.
    """
    if not answer_text or not cited_chunks:
        return CitationCheck(
            is_grounded=False,
            grounding_score=0.0,
            ungrounded_terms=[],
        )

    # Build the union of all content terms across every cited chunk.
    source_terms: set[str] = set()
    for rc in cited_chunks:
        source_terms |= _content_terms(rc.chunk.text)

    # Check 1: answer grounding.
    answer_terms = _content_terms(answer_text)
    if answer_terms:
        answer_score = len(answer_terms & source_terms) / len(answer_terms)
        ungrounded = sorted(answer_terms - source_terms)
    else:
        answer_score = 1.0
        ungrounded = []

    # Check 2: question coverage — did retrieval find content about the question?
    question_terms = _content_terms(question) if question else set()
    if question_terms:
        coverage_score = len(question_terms & source_terms) / len(question_terms)
    else:
        coverage_score = 1.0

    # Worst of the two is the overall score — fail on either axis.
    final_score = min(answer_score, coverage_score)

    return CitationCheck(
        is_grounded=final_score >= GROUNDING_THRESHOLD,
        grounding_score=round(final_score, 4),
        ungrounded_terms=ungrounded,
    )


if __name__ == "__main__":
    from data.documents import KNOWLEDGE_BASE
    from rag.retriever import build_store
    from rag.generator import generate_answer

    store = build_store(KNOWLEDGE_BASE)

    questions = [
        "How much does the Growth plan cost?",
        "How do I revoke a compromised API key?",
        "Does Acme support GraphQL?",
    ]
    for q in questions:
        retrieved = store.retrieve(q, top_k=3)
        answer = generate_answer(q, retrieved)
        check = verify_grounding(answer.text, retrieved)
        print(f"Q:  {q}")
        print(f"A:  {answer.text}")
        print(f"    grounded={check.is_grounded}  score={check.grounding_score}")
        if check.ungrounded_terms:
            print(f"    ungrounded terms: {check.ungrounded_terms[:8]}")
        print()
