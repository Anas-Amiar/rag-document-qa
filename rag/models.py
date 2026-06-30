from pydantic import BaseModel
from typing import Optional


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    source: str  # e.g. "pricing_page", "api_docs", "support_article"


class Chunk(BaseModel):
    chunk_id: str        # "{doc_id}::chunk_{n}"
    doc_id: str
    doc_title: str
    text: str
    start_char: int
    end_char: int


class RetrievedChunk(BaseModel):
    chunk: Chunk
    similarity_score: float   # 0.0 – 1.0


class GeneratedAnswer(BaseModel):
    text: str
    cited_chunk_ids: list[str]
    confidence: float          # derived from top retrieval similarity score


class CitationCheck(BaseModel):
    is_grounded: bool
    grounding_score: float     # fraction of answer key terms found in cited chunks
    ungrounded_terms: list[str]


class QaResult(BaseModel):
    question: str
    answer: str
    cited_chunks: list[RetrievedChunk]
    citation_check: CitationCheck
    status: str   # "answered", "low_confidence", "cannot_answer"
