# DocMind - capstone roadmap

One project that grows with every track. By the end it's a production-style
**document-intelligence platform**: ingestion → retrieval → grounded answers → agents → API,
with evals, observability and CI. That's the project you talk about in interviews for
40 minutes without running out of depth.

**The rule that makes it extensible:** everything talks through the Protocols in
`src/docmind/interfaces.py` (`Chunker`, `Embedder`, `VectorStore`, `LLM`, `Retriever`).
New phases **add implementations** and **add packages** (`llm/`, `rag/`, `eval/`, `agents/`,
`api/`); they never rewrite earlier code. Old milestone tests keep passing, which is your
regression suite.

```
              ┌──────────── Phase 4: agents/ (LangGraph, MCP server) ────────────┐
              │                                                                   │
 files ─► ingest/ ─► Chunker ─► Embedder ─► VectorStore ◄── Retriever ◄── rag/ ◄── LLM (Phase 2)
 (P0)     loaders     (P0)       (P0/P1)      (P0/P3)         (P3)        (P3)
                                                   ▲                         │
                                     eval/ (P3) ───┴─── golden set, metrics ─┘
                       api/ (P5): FastAPI + SSE + auth + tracing + caching + Docker + CI
```

## How to work a milestone

1. Read the docstring at the top of the module (it's the spec).
2. Implement it in `capstone/src/docmind/...`.
3. `study test capstone` (or `pytest capstone/tests/test_m02_loaders.py`). Results feed your
   mastery stats automatically, because the tests are tagged with curriculum topics.
4. Stuck for 30+ minutes? Compare with `solutions/capstone/src/docmind/...`, then **close it and
   re-write from memory the next day**.
5. Tick the box below, commit, and write 2-3 lines in `journal/decisions.md`: what you chose,
   what you rejected, and why. (That file becomes your interview stories.)

---

## Phase 0: Python foundations (current track: `py`) ✅ tests provided

| ✓ | Milestone | Module to implement | Concepts |
|---|-----------|---------------------|----------|
| [ ] | **M0.1** config & logging | `config.py` | pydantic-settings, SecretStr, validators, lru_cache, logging handlers, JSON logs |
| [ ] | **M0.2** loaders | `ingest/loaders.py` | pathlib, encodings/BOM, lazy generators, eager validation, JSONL, resilient error handling |
| [ ] | **M0.3** chunkers | `ingest/chunking.py` | generators, offsets & metadata, recursive splitting, Protocols |
| [ ] | **M0.4a** hashing embedder | `embeddings/hashing.py` | NumPy vectorisation, normalisation, deterministic hashing |
| [ ] | **M0.4b** HTTP embedder | `embeddings/http.py` | httpx AsyncClient, Semaphore, gather ordering, retries + jitter + Retry-After, pydantic validation |
| [ ] | **M0.5** vector store | `store/memory.py` | cosine via matmul, argpartition top-k, boolean masks, dunders, persistence |
| [ ] | **M0.6** pipeline | `pipeline.py` | streaming batch pipeline, async, dependency inversion |
| [ ] | **M0.7** CLI | `cli.py` | argparse, console scripts, asyncio.run, exit codes |

Stretch goals (no tests given, so write your own; that's the `py.testing` practice):
- [ ] `docmind stats` command: docs, chunks, avg chunk length, per-source counts (pandas)
- [ ] Profile ingesting 10k docs with cProfile, optimise the hottest function, record before/after
- [ ] Run `mypy --strict src/docmind` clean
- [ ] Ingest with a thread pool for file reads vs. sequential: measure and explain

## Phase 1: ML / DL (`ml`, `nlp` tracks)
- [ ] `ml/classifier.py`: TF-IDF + logistic regression doc classifier (scikit-learn); metrics report
- [ ] `nlp/bm25.py`: BM25 index from scratch (becomes the sparse half of hybrid search)
- [ ] `embeddings/local.py`: sentence-transformers embedder behind the same `Embedder` Protocol
- [ ] Notebook: compare hashing vs. TF-IDF vs. neural embeddings on a small labelled set (recall@5)
- Tag tests with `ml.*` / `nlp.*` topics (`study start ml.classical` first).

## Phase 2: LLMs (`llm` track)
- [ ] `llm/base.py` + `llm/anthropic.py` / `llm/openai.py` / `llm/ollama.py`: `LLM` Protocol impls
- [ ] Streaming (`stream()` async generator), timeouts, retries, token + cost accounting per call
- [ ] Structured output: answer → pydantic model with a validation-error repair loop
- [ ] Prompt templates as versioned files; a fake LLM for tests
- [ ] Wire `HttpEmbedder` into the CLI via settings (`DOCMIND_EMBEDDER=http`)

## Phase 3: RAG + vector DB + evaluation (`rag` track)
- [ ] `rag/answer.py`: retrieve → assemble context with citations → generate → `Answer(text, citations)`
- [ ] `store/qdrant.py` or `store/pgvector.py`: same `VectorStore` Protocol, so the old tests run against it too
- [ ] `rag/hybrid.py`: BM25 + dense with reciprocal-rank fusion; `rag/rerank.py`: cross-encoder
- [ ] `eval/`: golden Q/A set, recall@k / MRR, faithfulness + answer-relevance with LLM-as-judge
- [ ] `docmind eval`: prints a metrics table; CI fails if recall@5 drops more than X%
- [ ] Experiment log: chunk size × overlap × retriever → metrics (pandas), with a decision recorded

## Phase 4: agents, LangGraph, MCP (`agents` track)
- [ ] `agents/tools.py`: search / fetch-document / calculator tools with pydantic schemas
- [ ] `agents/research.py`: LangGraph graph (plan → retrieve → answer → self-check → retry) with checkpointing
- [ ] Human-in-the-loop interrupt before an expensive action
- [ ] `mcp_server.py`: expose `search` and `ingest` as MCP tools; use it from an MCP client
- [ ] Agent eval: task success rate, steps per task, cost per task

## Phase 5: production (`prod` track)
- [ ] `api/`: FastAPI `/ingest`, `/search`, `/ask` (SSE streaming), auth, per-tenant metadata filters
- [ ] Background ingestion jobs (queue), idempotency keys
- [ ] Caching (exact + semantic), rate limiting, model fallback routing
- [ ] Tracing (OpenTelemetry / Langfuse): latency, tokens, cost per request
- [ ] Dockerfile + docker-compose (api + vector DB), GitHub Actions: lint → mypy → tests → evals
- [ ] Load test (locust/k6) with a p95 latency target; write up the bottleneck you found

## Phase 6: interview packaging (`interview` track)
- [ ] README with an architecture diagram and *measured* numbers (recall, latency, cost)
- [ ] 5 ADRs (architecture decision records) from `journal/decisions.md`
- [ ] 6 STAR stories: a bug you debugged, a trade-off, a performance win, an eval-driven change,
      a failure / incident, a design you'd change
- [ ] Be able to whiteboard "design DocMind for 10M docs, 100 QPS, multi-tenant" in 45 minutes
