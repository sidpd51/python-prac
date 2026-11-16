"""Extension points. (GIVEN - read this first; it's the architecture.)

Every phase of the roadmap plugs in by implementing one of these Protocols, so the rest of
the system never changes:

    Phase 0  CharChunker, RecursiveChunker, HashingEmbedder, HttpEmbedder, InMemoryVectorStore
    Phase 1  (ml)      a learned embedder / doc classifier
    Phase 2  (llm)     LLM implementations: Anthropic / OpenAI / local (Ollama)
    Phase 3  (rag)     Qdrant / pgvector VectorStore, HybridRetriever, Reranker, Evaluator
    Phase 4  (agents)  Tools wrapping Retriever; LangGraph agent; MCP server
    Phase 5  (prod)    FastAPI app that depends only on these Protocols (easy to test/mock)

Structural typing (Protocol) means implementations don't inherit from anything - they just
need the right methods. That's also what makes fakes trivial in tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .models import Chunk, Document, ScoredChunk


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> Iterator[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim), rows L2-normalised."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(
        self, vector: np.ndarray, k: int = 5, where: Mapping[str, Any] | None = None
    ) -> list[ScoredChunk]: ...

    def delete(self, doc_id: str) -> int: ...

    def __len__(self) -> int: ...


# ---- future phases (defined now so the shape of the system is visible) -------------------


@runtime_checkable
class LLM(Protocol):  # Phase 2
    async def generate(self, messages: Sequence[Mapping[str, str]], **params: Any) -> str: ...

    def stream(self, messages: Sequence[Mapping[str, str]], **params: Any) -> AsyncIterator[str]: ...


@runtime_checkable
class Retriever(Protocol):  # Phase 3
    async def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]: ...
