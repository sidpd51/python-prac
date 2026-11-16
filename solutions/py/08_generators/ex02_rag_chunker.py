"""
Exercise 08.2 - RAG chunker with metadata                       level: ★★☆
topics: generators.rag_chunking, generators.chunk_metadata, generators.yield_from,
        generators.document_processing

Problem
-------
1. chunk_text(text, size, overlap) -> iterator of Chunk
     * character-window chunking: each chunk is text[start:start+size]
     * consecutive chunks overlap by `overlap` characters (step = size - overlap)
     * the last chunk may be shorter; never yield an empty chunk; stop once a chunk
       reaches the end of the text (no tiny trailing chunk fully inside the previous one)
     * Chunk.index is 0,1,2..., start/end are character offsets into `text`
     * VALIDATE EAGERLY: chunk_text(..., size=0) or overlap >= size or overlap < 0
       must raise ValueError *at call time*, not on the first next().
       (A plain generator function can't do that - think about why, and how to fix it.)

2. chunk_documents(docs, size, overlap) -> iterator of Chunk
     * docs is an iterable of dicts: {"id": str, "text": str, **any_other_metadata}
     * every chunk carries metadata = {"doc_id": id, **other_metadata}  (not "text")
     * index restarts at 0 for each document
     * use `yield from` to delegate to chunk_text

Interviewers look for
---------------------
* Generator bodies don't run until first next() -> validation inside them is deferred.
  The fix: a regular function that validates, then returns an inner generator.
* Preserving source metadata (doc id, page, url) through chunking is what makes
  citations and metadata filtering in a vector DB possible.
* Metadata dicts must not be shared/mutated across chunks accidentally.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str, size: int, overlap: int = 0, metadata: dict[str, Any] | None = None
) -> Iterator[Chunk]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("need 0 <= overlap < size")
    return _chunks(text, size, overlap, metadata or {})


def _chunks(text: str, size: int, overlap: int, metadata: dict[str, Any]) -> Iterator[Chunk]:
    step = size - overlap
    index = 0
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        yield Chunk(text[start:end], index, start, end, dict(metadata))
        index += 1
        if end == len(text):
            return


def chunk_documents(docs: Iterable[dict[str, Any]], size: int, overlap: int = 0) -> Iterator[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    return _chunk_docs(docs, size, overlap)


def _chunk_docs(docs: Iterable[dict[str, Any]], size: int, overlap: int) -> Iterator[Chunk]:
    for doc in docs:
        meta = {k: v for k, v in doc.items() if k not in ("id", "text")}
        yield from chunk_text(doc["text"], size, overlap, {"doc_id": doc["id"], **meta})
