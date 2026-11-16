"""
Exercise 23.3 - Recursive chunker with token budget, overlap + a lazy ingest pipeline   level: ★★★
topics: genai.chunking, genai.document_processing, genai.pipelines, genai.logging_tracking

Problem
-------
Fixed-size character windows (ex 08.2) cut sentences in half. A *recursive* splitter
prefers natural boundaries: paragraphs, then lines, then sentences, then words.
In this exercise a token is simply a whitespace-separated word: count_tokens is given.

1. split_recursive(text, max_tokens, separators=("\\n\\n", "\\n", ". ", " ")) -> list[str]
   * if count_tokens(text) <= max_tokens: return [text.strip()] (or [] if that is empty)
   * otherwise take the FIRST separator that occurs in text and split on it, keeping the
     separator attached to the end of the piece before it (so "".join(pieces) == text)
   * greedily merge consecutive pieces (plain concatenation) while the merged text stays
     within max_tokens; when the next piece doesn't fit, emit the current one and start over
   * a single piece that is itself over budget is split recursively with the REMAINING
     separators (those after the one just used)
   * no separator left (or none occurs) -> return the text as one piece, even if too big
   * every emitted piece is stripped; empty pieces are dropped

2. chunk_document(doc, *, max_tokens, overlap=0) -> list[Chunk]
   * doc = {"id": str, "text": str, **metadata}
   * VALIDATE: max_tokens >= 1 and 0 <= overlap < max_tokens, else ValueError
   * pieces = split_recursive(text, max_tokens - overlap)
   * chunk i > 0 starts with the last `overlap` words of piece i-1, then a space, then
     piece i (so no chunk exceeds max_tokens). Chunk 0 has no prefix.
   * Chunk(id=f"{doc_id}:{i}", doc_id, index=i, text, n_tokens=count_tokens(text),
     metadata=copy of the doc's other keys - not "id"/"text", and a NEW dict per chunk)

3. ingest(lines, *, max_tokens, overlap=0, normalize=str.strip) -> Iterator[Chunk]
   A LAZY pipeline over JSONL lines (think: a 20 GB file object):
   * validate max_tokens/overlap EAGERLY (at call time, not at first next())
   * skip blank lines; a line that isn't valid JSON, or isn't an object with str "id" and
     str "text", is skipped with logger "genai.ingest" .warning("skipping line %d: %s",
     line_number, reason)   (line numbers start at 1)
   * text = normalize(doc["text"]); skip docs whose normalised text is empty
   * skip exact duplicates of an earlier doc's normalised text - remember sha256 digests,
     not the texts (memory stays small)
   * yield the chunks of each remaining document (normalised text, same metadata)
   * never read further ahead in `lines` than needed for the next chunk

Interviewers look for
---------------------
* Chunk size is a retrieval-quality knob: too small loses context, too big dilutes the
  embedding and wastes prompt tokens. Overlap protects facts that straddle a boundary.
* Metadata (doc id, source, page) on every chunk = citations + filtered search later.
* `words[-0:]` is the WHOLE list - the classic overlap=0 bug.
* Generator pipelines: O(chunk) memory, natural back-pressure, but errors surface late -
  hence eager validation and per-line error handling (log + skip, don't crash a 6h job).
"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("genai.ingest")

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")

__stub_keep__ = ["count_tokens"]


def count_tokens(text: str) -> int:
    """Given: a token is a whitespace-separated word."""
    return len(text.split())


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    index: int
    text: str
    n_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


def split_recursive(text: str, max_tokens: int, separators: Sequence[str] = DEFAULT_SEPARATORS) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text.strip()] if text.strip() else []
    for pos, sep in enumerate(separators):
        if sep in text:
            rest = separators[pos + 1 :]
            break
    else:
        return [text.strip()]

    raw = text.split(sep)
    pieces = [p + sep for p in raw[:-1]] + [raw[-1]]
    out: list[str] = []
    current = ""
    for piece in pieces:
        if count_tokens(current + piece) <= max_tokens:
            current += piece
            continue
        if current.strip():
            out.append(current.strip())
        if count_tokens(piece) > max_tokens:
            out.extend(split_recursive(piece, max_tokens, rest))
            current = ""
        else:
            current = piece
    if current.strip():
        out.append(current.strip())
    return out


def _validate(max_tokens: int, overlap: int) -> None:
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not 0 <= overlap < max_tokens:
        raise ValueError("need 0 <= overlap < max_tokens")


def chunk_document(doc: dict[str, Any], *, max_tokens: int, overlap: int = 0) -> list[Chunk]:
    _validate(max_tokens, overlap)
    doc_id = doc["id"]
    meta = {k: v for k, v in doc.items() if k not in ("id", "text")}
    pieces = split_recursive(doc["text"], max_tokens - overlap)
    chunks = []
    for i, piece in enumerate(pieces):
        text = piece
        if i > 0 and overlap:  # guard: words[-0:] would be ALL words
            prefix = " ".join(pieces[i - 1].split()[-overlap:])
            text = f"{prefix} {piece}"
        chunks.append(Chunk(f"{doc_id}:{i}", doc_id, i, text, count_tokens(text), dict(meta)))
    return chunks


def ingest(
    lines: Iterable[str],
    *,
    max_tokens: int,
    overlap: int = 0,
    normalize: Callable[[str], str] = str.strip,
) -> Iterator[Chunk]:
    _validate(max_tokens, overlap)
    return _ingest(lines, max_tokens, overlap, normalize)


def _parse(line: str) -> tuple[dict[str, Any] | None, str]:
    try:
        doc = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc.msg})"
    if not isinstance(doc, dict):
        return None, "not a JSON object"
    if not isinstance(doc.get("id"), str) or not isinstance(doc.get("text"), str):
        return None, "missing str 'id' or 'text'"
    return doc, ""


def _ingest(
    lines: Iterable[str], max_tokens: int, overlap: int, normalize: Callable[[str], str]
) -> Iterator[Chunk]:
    seen: set[bytes] = set()
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        doc, reason = _parse(line)
        if doc is None:
            log.warning("skipping line %d: %s", lineno, reason)
            continue
        text = normalize(doc["text"])
        if not text:
            continue
        digest = hashlib.sha256(text.encode()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        yield from chunk_document({**doc, "text": text}, max_tokens=max_tokens, overlap=overlap)
