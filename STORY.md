# RAG Document QA with Citation Verification — the pitch

*A 2-minute walkthrough for presenting this project in an interview.*

## The 30-second version

"Most RAG demos stop at 'I retrieved some chunks and passed them to an LLM.' But that
leaves two production failures completely unguarded: retrieval can bring back the wrong
chunks, and the LLM can hallucinate claims that aren't in those chunks at all. I built a
pipeline that runs a two-axis grounding check on every answer — did the retrieval actually
find content about what was asked, and is the answer actually supported by what was
retrieved? Anything that fails either check is flagged as low-confidence and logged as a
failure case, so you always know whether a gap is a missing document or a generator problem."

## The problem, in plain terms

Imagine a customer-support bot powered by RAG. Someone asks "Does this product support
GraphQL?" The retrieval system finds a chunk about your Support SLA policy, because both
contain the word "support." The LLM reads that chunk and generates a confident-sounding
non-answer. The user gets a response that mentions response times and account managers
instead of the actual answer. Nobody flagged it. Nobody logged it.

This happens constantly in production RAG. The fix isn't "use a better retrieval system" —
even good retrieval misses sometimes. The fix is building a layer that checks whether the
answer actually addresses the question before shipping it.

## The idea

Two verification axes, applied per answer:

1. **Answer grounding** — are the key factual terms in the answer actually present in the
   chunks that were cited? If the answer introduces something the chunks never said, flag it.

2. **Question coverage** — do the cited chunks actually contain what the question asked
   about? If the question asks about "GraphQL" and no cited chunk mentions "GraphQL", the
   retrieval missed — even if the answer looks plausible.

Score = the minimum of the two. Fail either axis, fail the check. That score decides
whether the answer is returned confidently, flagged as low-confidence, or refused entirely.

## How I built it (in order, and why that order)

1. **The knowledge base + chunker** (`data/documents.py`, `rag/chunker.py`) — 6 documents
   on a fictional SaaS product (pricing, API docs, webhooks, integrations, support, data
   policy). Chunked with sentence-boundary awareness and overlap so a sentence that
   straddles two chunks is fully represented in at least one. The KB had to be varied enough
   to produce real retrieval ambiguity (the "support" confusion case is organic, not staged).

2. **The embedder + retriever** (`rag/embedder.py`, `rag/retriever.py`) — bag-of-words TF
   vectors with stop-word filtering, cosine similarity, in-memory vector store. Honest
   about what it is: a keyword-overlap baseline, not a semantic embedding model. Built first
   so every subsequent layer has a real retrieval signal to work with.

3. **The generator** (`rag/generator.py`) — extractive QA: finds the sentence in the top
   retrieved chunk with the highest overlap with the question. Deliberately extractive in
   mock mode so the grounding check has real signal: the answer is a genuine substring of
   the chunks, not a template that would produce artificially perfect grounding scores.

4. **The citation verifier** (`rag/verifier.py`) — the core of the project. Two-axis check
   (answer grounding + question coverage), both computed from the same chunk content.
   This had to come after the generator, because it needs an actual answer and actual cited
   chunks to check against.

5. **The pipeline glue** (`rag/pipeline.py`) — `ask(question)` chains everything together
   and makes the status decision: answered / low_confidence / cannot_answer. One entry
   point, one typed return value.

6. **The feedback loop + dashboard** (`rag/feedback.py`, `rag/dashboard.py`) — logs every
   failure with its failure type (retrieval gap vs. grounding failure), so you can tell
   whether you need more documents or a better generator.

## The result

Running 14 questions through the pipeline (10 answerable from the KB, 4 outside it):

| Status | Count | What it means |
|---|---|---|
| answered | 9 (64%) | In-KB questions confidently answered and grounded |
| low_confidence | 3 | GraphQL question, mobile app question, API rate-limit edge case |
| cannot_answer | 2 | "Capital of France" and "What language is Acme built in?" — clean refusal |
| Failure breakdown | 3 grounding failures, 2 retrieval gaps | Tells you WHERE to improve |

The system cleanly refused both completely off-topic questions and correctly flagged the
GraphQL question as low-confidence — "support" retrieved the wrong chunk from the Support
SLA doc, question coverage caught it.

## What I'd highlight if asked "what was the hardest design decision?"

Deciding on the two-axis grounding check instead of just one score. The natural first
instinct is to check whether the answer's terms appear in the cited chunks — but that
misses the case where retrieval succeeds (it found a real chunk, properly cited) but
returned the wrong thing. Answer grounding alone is blind to retrieval errors. Adding
question coverage as a second axis meant a retrieval that brings back irrelevant content
gets caught even when the answer "looks" grounded, because the question's key terms aren't
in the retrieved text at all.

That design decision is also what makes the failure analytics meaningful: "retrieval gap"
and "grounding failure" point at different parts of the system to fix.

## What I'd build next

- Swap `embed()` for a real embedding API (OpenAI, Cohere) for semantic rather than
  keyword retrieval — the "support" false-positive would disappear with real embeddings
- Wire in a real LLM for generation instead of extractive QA
- Re-ranking: retrieve top-10, re-rank to top-3 with a cross-encoder, then generate
- A persistent vector store (Chroma, pgvector) so the index survives restarts
- A UI for browsing the failure eval dataset and marking whether it was a retrieval or
  generation problem

## Companion projects

This project is the "can the system answer reliably?" layer — which pairs with the
**Failure Forensics Tool** (Project 3), which asks "when a pipeline step fails, which one
was it?" Both are about building AI systems that can inspect and explain their own failures
instead of silently returning bad output.

Together with the **Model Regression Detection System** (Project 1), the **LLM Cost
Autopilot** (Project 2), and the **OCR Document Intelligence Pipeline** (Project 4),
these five projects cover correctness, cost, observability, uncertainty-aware automation,
and retrieval reliability — the full practical stack of concerns for shipping AI in production.
