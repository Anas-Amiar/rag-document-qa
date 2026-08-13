# RAG Document QA with Citation Verification

[![CI](https://github.com/Anas-Amiar/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Anas-Amiar/rag-document-qa/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **▶ Live demo:** [https://rag-document-qa-2527.onrender.com/docs](https://rag-document-qa-2527.onrender.com/docs) — a running instance on Render's free tier. The first request after a while takes ~50s to wake the service, then it's fast.

A production-minded RAG pipeline that answers questions from a knowledge base and — the part most RAG demos skip — verifies that every answer is actually grounded in the sources it cited. Answers that don't hold up are routed to a human review queue instead of being silently returned as confident.

Ships as a real HTTP service (FastAPI) **and** a pure, unit-tested core. Runs on a mock
embedder + extractive generator by default, so it needs **no API keys** — clone it and the
service (or the one-click deploy) is live immediately.

In a **mock-mode** demo batch of 14 questions (10 answerable, 4 outside the knowledge base), the system answered 9 in-KB questions confidently (64.3%), correctly flagged 3 as low-confidence (including two questions the KB doesn't cover), and cleanly refused 2 completely off-topic questions. Reproduce it with `python3 -m rag.dashboard` — no keys, no network.

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

## Quickstart

```bash
git clone https://github.com/Anas-Amiar/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt

python3 -m rag.chunker      # see how documents are split into overlapping chunks
python3 -m rag.retriever    # see retrieval scores for sample queries
python3 -m rag.generator    # see extractive answers + confidence scores
python3 -m rag.verifier     # see the two-axis grounding check per answer
python3 -m rag.feedback     # run the pipeline + log failures + show analytics
python3 -m rag.dashboard    # full batch run across 14 questions + dashboard
```

Everything runs in **mock mode** by default — no embedding API, no LLM API key, no setup beyond `pip install pydantic`. The bag-of-words embedder and extractive generator are genuine, working implementations (not stubs) so every pipeline stage produces real, meaningful signal.

## Run the API

```bash
uvicorn rag.app:app --reload     # http://localhost:8000  (interactive docs at /docs)
```

```bash
# an in-KB question -> answered, grounded, with the source chunk cited
curl -s -X POST localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question":"How much does the Growth plan cost?"}'

# a topic the KB doesn't cover -> low_confidence, is_grounded=false (not a hallucination)
curl -s -X POST localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question":"Does Acme support GraphQL queries?"}'
```

Every response carries its `status` (`answered` / `low_confidence` / `cannot_answer`), the
cited chunks, and a two-axis `citation_check`. `GET /documents` lists the knowledge base.

## Deploy your own

Runs on the mock embedder + generator with no secrets, so a public demo is one click:

- **Render** — New → Blueprint → point at this repo (`render.yaml` included, free tier).
- **Docker** — `docker build -t rag-document-qa . && docker run -p 8000:8000 rag-document-qa`

To make it production-grade, swap `embed()` for an embedding API and `generate_answer()`
for an LLM call — the chunker, retriever, verifier, and feedback loop are unchanged.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q        # 12 tests, deterministic, no network
```

The tests lock in the honesty properties: in-KB questions come back answered *and* grounded
with citations, out-of-KB topics are flagged `low_confidence`, and off-topic questions
return `cannot_answer` with zero citations. CI runs the suite on Python 3.10–3.12.

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
