"""
Milestone M0.3 - chunkers (both satisfy interfaces.Chunker)
topics: py.genai.chunking, py.generators.rag_chunking, py.generators.chunk_metadata, py.oop.composition

Build
-----
1. CharChunker(size, overlap=0)
     * __init__ validates: size > 0 and 0 <= overlap < size, else ValueError
     * chunk(doc) lazily yields Chunk objects: windows of `size` chars, step size-overlap,
       stop once a window reaches the end of the text; empty text -> no chunks
     * Chunk.id = f"{doc.id}#{index}", doc_id, index, start/end offsets, metadata = a COPY of
       doc.metadata

2. RecursiveChunker(size, separators=("\\n\\n", "\\n", ". ", " "))
     Structure-aware: never cut mid-paragraph/sentence/word unless unavoidable.
     * Split the text on the first separator that occurs in it (keep the separator attached
       to the END of the preceding piece so offsets still add up); any piece still longer than
       `size` is split recursively with the remaining separators; with no separators left,
       hard-split every `size` chars.
     * Then greedily MERGE consecutive pieces while the merged span is <= size.
     * Yield Chunks like CharChunker (text = doc.text[start:end], so offsets are exact);
       skip whitespace-only chunks; index counts yielded chunks only.
     * Invariant: every chunk is <= size chars, and joining all chunks (incl. the skipped
       whitespace) reproduces the document exactly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..models import Chunk, Document


class CharChunker:
    def __init__(self, size: int, overlap: int = 0) -> None:
        raise NotImplementedError

    def chunk(self, doc: Document) -> Iterator[Chunk]:
        raise NotImplementedError


class RecursiveChunker:
    def __init__(self, size: int, separators: Sequence[str] = ("\n\n", "\n", ". ", " ")) -> None:
        raise NotImplementedError

    def chunk(self, doc: Document) -> Iterator[Chunk]:
        raise NotImplementedError
