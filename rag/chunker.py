from rag.models import Document, Chunk


def split_into_chunks(doc: Document, chunk_size: int = 200, overlap: int = 50) -> list[Chunk]:
    """
    Split a document into overlapping fixed-size character chunks.
    Overlap lets a sentence that straddles two chunk boundaries still be
    fully represented in at least one of them.
    """
    text = doc.text
    chunks: list[Chunk] = []
    start = 0
    n = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Extend to the next sentence boundary (period + space) if possible,
        # so chunks don't cut mid-sentence.
        if end < len(text):
            boundary = text.rfind(". ", start, end + 60)
            if boundary != -1 and boundary > start:
                end = boundary + 1  # include the period

        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::chunk_{n}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            text=text[start:end].strip(),
            start_char=start,
            end_char=end,
        ))
        n += 1
        start = max(start + 1, end - overlap)  # step forward but keep overlap

    return chunks


if __name__ == "__main__":
    from data.documents import KNOWLEDGE_BASE

    for doc in KNOWLEDGE_BASE:
        chunks = split_into_chunks(doc)
        print(f"{doc.title}: {len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c.chunk_id}] ({len(c.text)} chars): {c.text[:80]}...")
        print()
