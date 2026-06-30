"""
Mock embedder using bag-of-words term frequency vectors.

In production this would call an embedding API (OpenAI, Cohere, etc.).
The bag-of-words approach is a legitimate baseline for keyword-heavy retrieval
and lets the entire pipeline run with zero API keys and zero cost.
"""

import math
import re


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "i", "you", "we", "they", "it", "this",
    "that", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "through", "per",
    "can", "not", "your", "our", "all", "any", "if", "as", "so", "how",
    "what", "when", "where", "which", "who", "my", "me", "its", "also",
    # domain-universal: appears in every doc, carries no discriminating signal
    "acme", "platform", "use",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def embed(text: str) -> dict[str, float]:
    """Return a TF vector: term → normalized term frequency."""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts: dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: count / total for t, count in counts.items()}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
