from rag.models import Chunk, RetrievedChunk
from rag.embedder import embed, cosine_similarity


class VectorStore:
    """
    In-memory vector store.  Index chunks once at startup; retrieve top-k
    at query time.  In production this would be a real vector DB
    (Pinecone, Chroma, pgvector, etc.).
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[dict[str, float]] = []

    def add(self, chunk: Chunk) -> None:
        self._chunks.append(chunk)
        self._vectors.append(embed(chunk.text))

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        q_vec = embed(query)
        scored = [
            (cosine_similarity(q_vec, v), chunk)
            for v, chunk in zip(self._vectors, self._chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, similarity_score=round(score, 4))
            for score, chunk in scored[:top_k]
            if score > 0
        ]

    def __len__(self) -> int:
        return len(self._chunks)


def build_store(documents) -> VectorStore:
    from rag.chunker import split_into_chunks

    store = VectorStore()
    for doc in documents:
        for chunk in split_into_chunks(doc):
            store.add(chunk)
    return store


if __name__ == "__main__":
    from data.documents import KNOWLEDGE_BASE

    store = build_store(KNOWLEDGE_BASE)
    print(f"Indexed {len(store)} chunks\n")

    test_queries = [
        "How much does the Growth plan cost?",
        "How do I authenticate API requests?",
        "What happens to my data when I cancel?",
    ]
    for q in test_queries:
        results = store.retrieve(q, top_k=2)
        print(f"Q: {q}")
        for r in results:
            print(f"  [{r.similarity_score:.4f}] {r.chunk.chunk_id}: {r.chunk.text[:80]}...")
        print()
