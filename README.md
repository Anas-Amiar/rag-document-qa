# RAG Document QA with Citation Verification

A production-minded RAG pipeline that answers questions from a knowledge base and — the part most RAG demos skip — verifies that every answer is actually grounded in the sources it cited. Answers that don't hold up are routed to a human review queue instead of being silently returned as confident.

In a demo batch of 14 questions (10 answerable, 4 outside the knowledge base), the system answered 9 in-KB questions confidently (64.3%), correctly flagged 3 as low-confidence (including two questions the KB doesn't cover), and cleanly refused 2 completely off-topic questions.

## Why this exists

Most RAG demos stop at "I retrieved some chunks and passed them to an LLM." Production RAG systems live or die on two failure modes that nobody demos:
1. **Retrieval misses** — the question is in the KB but the wrong chunk comes back.
2. **Hallucinated answers** — the LLM produces a confident-sounding answer that isn't actually supported by the retrieved text.

This project builds the infrastructure to detect both, per answer, and route failures out of the happy path instead of silently shipping them.

## How it works

```
rag/
  models.py         Typed shapes: Document, Chunk, RetrievedChunk, GeneratedAnswer,
                     CitationCheck, QaResult
  chunker.py         Splits documents into overlapping character chunks at sentence
                     boundaries (chunk_size=200, overlap=50)
  embedder.py        Bag-of-words TF vectors with stop-word filtering.
                     In production: replace with any embedding API call.
  retriever.py       In-memory vector store. Index once at startup, retrieve top-k
                     chunks by cosine similarity at query time.
  generator.py       Extractive QA in mock mode: finds the sentence in the top retrieved
                     chunk with the highest word-overlap with the question.
                     In real mode: wire in any LLM (GPT-4o, Claude, etc.)
  verifier.py        Two-axis grounding check per answer:
                     (1) Answer grounding — are the answer's key terms in the cited chunks?
                     (2) Question coverage — does the KB even contain what the question asked?
                     Score = min(answer_grounding, question_coverage). Fails either axis, fails overall.
  feedback.py        Logs every low_confidence and cannot_answer result to an eval dataset.
                     Analytics: how many failures are retrieval gaps vs. grounding failures?
  dashboard.py       Batch run across a question set + aggregate stats.
data/
  documents.py       6 knowledge-base documents (Acme Platform: pricing, API docs, webhooks,
                     integrations, support, data policy)
reports/             Generated output (eval dataset) — gitignored
```

### The flow for one question

```
ask("How do I revoke a compromised API key?")
  1. retrieve()        -> top-3 most similar chunks (cosine similarity on TF vectors)
  2. generate_answer() -> extract best-matching sentence(s) from top chunks
  3. verify_grounding()-> score answer grounding + question coverage; min = final score
  4. status decision   -> answered (score >= 0.60) / low_confidence / cannot_answer
  5. log_failure()     -> if not answered, append to eval dataset
```

## Setup

```bash
git clone https://github.com/Anas-Amiar/Project-5-rag-document-qa.git
cd "Project 5 - rag-document-qa"
pip install -r requirements.txt

python3 -m rag.chunker      # see how documents are split into overlapping chunks
python3 -m rag.retriever    # see retrieval scores for sample queries
python3 -m rag.generator    # see extractive answers + confidence scores
python3 -m rag.verifier     # see the two-axis grounding check per answer
python3 -m rag.feedback     # run the pipeline + log failures + show analytics
python3 -m rag.dashboard    # full batch run across 14 questions + dashboard
```

Everything runs in **mock mode** by default — no embedding API, no LLM API key, no setup beyond `pip install pydantic`. The bag-of-words embedder and extractive generator are genuine, working implementations (not stubs) so every pipeline stage produces real, meaningful signal.

To use real embeddings and a real LLM:
1. Replace `embed()` in `rag/embedder.py` with a call to your embedding API.
2. Replace the mock branch in `rag/generator.py` with an LLM chat completion call.

## Architecture decisions

**Why two axes for grounding (answer grounding + question coverage)?**
Answer grounding alone doesn't catch the case where retrieval returned a plausible-looking chunk that doesn't actually address the question. A question about "GraphQL support" might retrieve a chunk from a support-policy document — the answer is grounded in that chunk, but it's the wrong answer. Question coverage catches this: if the question's key terms don't appear in the cited chunks, the retrieval probably missed.

**Why bag-of-words instead of a real embedding model?**
The whole pipeline can be built, tuned, and demoed with zero API cost and zero latency. The bag-of-words approach is a legitimate baseline for keyword-heavy retrieval (FAQ-style docs), and it makes the failure modes visible and debuggable rather than opaque. Swapping to real embeddings (OpenAI, Cohere, etc.) is a single function replacement.

**Why extractive QA in mock mode?**
An extractive answer is literally a substring of the retrieved chunks, so the citation check has real signal to work with: if the answer introduces terms not in the chunks, those terms are genuinely suspect. A template-based fake answer would produce artificially perfect grounding scores.

**Why log failure type (retrieval gap vs. grounding failure) separately?**
These require different fixes. A retrieval gap means the knowledge base doesn't cover this topic — fix by adding documents. A grounding failure means the KB has the answer but the generator isn't using it correctly — fix by improving the prompt or generator. Conflating them produces useless analytics.

## What's deliberately out of scope for v1

- Real embedding API (swap `embed()` in `embedder.py` — one function, ~5 lines)
- Real LLM generation (swap the mock branch in `generator.py` — ~10 lines)
- Persistent vector store (currently in-memory; replace `VectorStore` with Chroma or pgvector)
- Re-ranking (retrieve top-10, re-rank to top-3 with a cross-encoder model)
- A UI for browsing the eval dataset of failures
