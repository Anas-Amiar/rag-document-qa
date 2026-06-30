"""
Feedback loop — turns ungrounded or unanswerable queries into permanent eval cases.

Every time the pipeline returns low_confidence or cannot_answer, that question
(plus the answer the system gave and why it failed) is appended to an eval
dataset.  Over time this dataset reveals:
  - Which question topics the knowledge base doesn't cover (retrieval gaps)
  - Which questions the KB covers but the generator gets wrong (grounding failures)

That distinction tells you whether to fix the knowledge base or the generator.
"""

import json
import os
from datetime import datetime

from rag.models import QaResult

EVAL_DATASET_PATH = "reports/eval_dataset.json"


def _load() -> list[dict]:
    if not os.path.exists(EVAL_DATASET_PATH):
        return []
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


def _save(data: list[dict]) -> None:
    os.makedirs("reports", exist_ok=True)
    with open(EVAL_DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)


def log_failure(result: QaResult) -> None:
    """Append a failed QaResult to the eval dataset."""
    if result.status == "answered":
        return  # only log failures

    data = _load()
    failure_type = (
        "retrieval_gap" if result.status == "cannot_answer" else "grounding_failure"
    )
    data.append({
        "question": result.question,
        "status": result.status,
        "failure_type": failure_type,
        "system_answer": result.answer,
        "grounding_score": result.citation_check.grounding_score,
        "ungrounded_terms": result.citation_check.ungrounded_terms[:10],
        "cited_chunks": [rc.chunk.chunk_id for rc in result.cited_chunks],
        "logged_at": datetime.utcnow().isoformat(),
    })
    _save(data)


def failure_analytics() -> dict:
    """Summarise the eval dataset — which failure type dominates?"""
    data = _load()
    if not data:
        return {"total_failures": 0}

    by_type: dict[str, int] = {}
    for entry in data:
        t = entry.get("failure_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    avg_grounding = (
        sum(e.get("grounding_score", 0) for e in data) / len(data)
    )

    return {
        "total_failures": len(data),
        "by_failure_type": by_type,
        "avg_grounding_score_on_failures": round(avg_grounding, 4),
        "most_common_failure": max(by_type, key=by_type.__getitem__) if by_type else None,
    }


if __name__ == "__main__":
    from data.documents import KNOWLEDGE_BASE
    from rag.retriever import build_store
    from rag.pipeline import ask

    store = build_store(KNOWLEDGE_BASE)

    questions = [
        "How much does the Growth plan cost?",
        "What is the webhook retry schedule?",
        "Does Acme support GraphQL?",
        "What is the capital of France?",
        "Can I export my data?",
        "What programming language is Acme built in?",  # not in KB
    ]

    print("=== Running pipeline + logging failures ===\n")
    for q in questions:
        result = ask(q, store)
        log_failure(result)
        tag = {"answered": "OK ", "low_confidence": "LOW", "cannot_answer": "N/A"}[result.status]
        print(f"[{tag}] {q}")

    print("\n=== Failure Analytics ===")
    analytics = failure_analytics()
    for k, v in analytics.items():
        print(f"  {k}: {v}")
