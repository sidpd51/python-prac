"""
Milestone M0.6 - streaming ingestion pipeline + search
topics: py.generators.pipelines, py.genai.pipelines, py.genai.batching, py.async.asyncio_run,
        py.internals.profiling, py.type_hints.protocol

Build
-----
1. async ingest(docs, chunker, embedder, store, *, batch_size=64, replace=True) -> IngestReport
     * docs: any Iterable[Document] (e.g. load_documents(...)) - consume LAZILY: the pipeline
       must hold at most ~batch_size chunks in memory at a time (no list(all_chunks))
     * for each document: if replace, store.delete(doc.id) first (re-ingest = replace).
       Documents producing zero chunks count as `skipped`.
     * chunks are embedded in batches of batch_size (a batch may span documents) and
       upserted after each batch.
     * report documents, chunks, skipped, seconds (time.perf_counter)
     * depends ONLY on the Protocols in interfaces.py - works with any chunker/embedder/store

2. async search(query, embedder, store, k=5, where=None) -> list[ScoredChunk]
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from .interfaces import Chunker, Embedder, VectorStore
from .models import Document, IngestReport, ScoredChunk

log = logging.getLogger("docmind.pipeline")


async def ingest(
    docs: Iterable[Document],
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    *,
    batch_size: int = 64,
    replace: bool = True,
) -> IngestReport:
    raise NotImplementedError


async def search(
    query: str, embedder: Embedder, store: VectorStore, k: int = 5, where: Mapping[str, Any] | None = None
) -> list[ScoredChunk]:
    raise NotImplementedError
