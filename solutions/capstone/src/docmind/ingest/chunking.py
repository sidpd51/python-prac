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
        if size <= 0:
            raise ValueError("size must be > 0")
        if not 0 <= overlap < size:
            raise ValueError("need 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> Iterator[Chunk]:
        text, step = doc.text, self.size - self.overlap
        for index, start in enumerate(range(0, len(text), step)):
            end = min(start + self.size, len(text))
            yield Chunk(
                id=f"{doc.id}#{index}",
                doc_id=doc.id,
                index=index,
                text=text[start:end],
                start=start,
                end=end,
                metadata=dict(doc.metadata),
            )
            if end == len(text):
                return


class RecursiveChunker:
    def __init__(self, size: int, separators: Sequence[str] = ("\n\n", "\n", ". ", " ")) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        self.size = size
        self.separators = tuple(separators)

    def _spans(self, text: str, offset: int, seps: tuple[str, ...]) -> Iterator[tuple[int, int]]:
        if len(text) <= self.size:
            if text:
                yield offset, offset + len(text)
            return
        for i, sep in enumerate(seps):
            if sep in text:
                pos = 0
                while pos < len(text):
                    cut = text.find(sep, pos)
                    nxt = len(text) if cut == -1 else cut + len(sep)
                    yield from self._spans(text[pos:nxt], offset + pos, seps[i + 1 :])
                    pos = nxt
                return
        for s in range(0, len(text), self.size):
            yield offset + s, offset + min(s + self.size, len(text))

    def chunk(self, doc: Document) -> Iterator[Chunk]:
        index = 0
        cur: tuple[int, int] | None = None

        def emit(span: tuple[int, int]) -> Chunk | None:
            nonlocal index
            s, e = span
            if not doc.text[s:e].strip():
                return None
            c = Chunk(
                id=f"{doc.id}#{index}",
                doc_id=doc.id,
                index=index,
                text=doc.text[s:e],
                start=s,
                end=e,
                metadata=dict(doc.metadata),
            )
            index += 1
            return c

        for s, e in self._spans(doc.text, 0, self.separators):
            if cur is None:
                cur = (s, e)
            elif e - cur[0] <= self.size:
                cur = (cur[0], e)
            else:
                if (c := emit(cur)) is not None:
                    yield c
                cur = (s, e)
        if cur is not None and (c := emit(cur)) is not None:
            yield c
