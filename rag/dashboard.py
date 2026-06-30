"""
Batch dashboard — runs a set of questions through the full pipeline and
reports the numbers a team would actually care about:
  - answer rate (how many questions the system answered confidently)
  - grounding rate (of answered questions, how many were well-cited)
  - which failure type dominates (retrieval gap vs. grounding failure)
  - avg retrieval score (proxy for how well the KB covers the question set)
"""

from rag.models import QaResult
from rag.retriever import VectorStore, build_store
from rag.pipeline import ask
from rag.feedback import log_failure, failure_analytics


DEMO_QUESTIONS = [
    # Answerable from KB
    "How much does the Growth plan cost?",
    "How do I generate an API key?",
    "What is the rate limit for Starter plan API keys?",
    "How do I revoke a compromised API key?",
    "What is the webhook retry policy?",
    "How do I connect Slack to Acme?",
    "What is the support SLA for Enterprise customers?",
    "What happens to my data when I close my account?",
    "Can I download an export of my data?",
    "Does Acme offer a free trial?",
    # NOT in KB — should return cannot_answer
    "Does Acme support GraphQL?",
    "What programming language is Acme's backend built in?",
    "What is the capital of France?",
    "Does Acme have a mobile app?",
]


def run_batch(questions: list[str], store: VectorStore) -> dict:
    results: list[QaResult] = []
    for q in questions:
        result = ask(q, store)
        log_failure(result)
        results.append(result)

    total = len(results)
    answered = sum(1 for r in results if r.status == "answered")
    low_conf = sum(1 for r in results if r.status == "low_confidence")
    no_answer = sum(1 for r in results if r.status == "cannot_answer")

    grounding_scores = [r.citation_check.grounding_score for r in results if r.status == "answered"]
    avg_grounding = round(sum(grounding_scores) / len(grounding_scores), 4) if grounding_scores else 0.0

    return {
        "total_questions": total,
        "answered": answered,
        "low_confidence": low_conf,
        "cannot_answer": no_answer,
        "answer_rate_pct": round((answered / total) * 100, 1),
        "avg_grounding_score_on_answered": avg_grounding,
        "failure_analytics": failure_analytics(),
        "results": results,
    }


if __name__ == "__main__":
    import os
    # Clear previous run's eval dataset so the demo is clean.
    if os.path.exists("reports/eval_dataset.json"):
        os.remove("reports/eval_dataset.json")

    from data.documents import KNOWLEDGE_BASE
    store = build_store(KNOWLEDGE_BASE)

    stats = run_batch(DEMO_QUESTIONS, store)

    print("=== RAG Document QA — batch run ===\n")
    for r in stats["results"]:
        tag = {"answered": "OK ", "low_confidence": "LOW", "cannot_answer": "N/A"}[r.status]
        gs = f"  (grounding={r.citation_check.grounding_score:.2f})" if r.status == "answered" else ""
        print(f"[{tag}] {r.question[:65]:<65}{gs}")

    print("\n=== Dashboard ===")
    print(f"Total questions:        {stats['total_questions']}")
    print(f"Answered:               {stats['answered']}  ({stats['answer_rate_pct']}%)")
    print(f"Low confidence:         {stats['low_confidence']}")
    print(f"Cannot answer:          {stats['cannot_answer']}")
    print(f"Avg grounding score:    {stats['avg_grounding_score_on_answered']}")
    fa = stats["failure_analytics"]
    print(f"\nFailure breakdown:      {fa.get('by_failure_type', {})}")
    print(f"Most common failure:    {fa.get('most_common_failure', 'none')}")
