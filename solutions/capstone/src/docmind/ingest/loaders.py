"""
Milestone M0.2 - document loaders
topics: py.files.pathlib, py.files.encoding, py.files.large_files, py.generators.document_processing,
        py.exceptions.logging, py.files.json

Build
-----
1. iter_files(root, suffixes=SUPPORTED, ignore_dirs=IGNORE_DIRS) -> Iterator[Path]
     * recursive, deterministic (sorted) order, only files whose suffix (lower-cased) is in
       `suffixes`, skipping any path that has a directory named in `ignore_dirs`
     * root that doesn't exist -> FileNotFoundError raised at CALL time (eager validation)

2. read_text(path) -> str
     * decode UTF-8; strip a UTF-8 BOM if present; if the bytes are not valid UTF-8 fall back
       to latin-1 and log a WARNING on the "docmind.ingest" logger mentioning the file name

3. load_documents(root) -> Iterator[Document]   (lazy - one file in memory at a time)
     * .txt / .md: one Document per file; id = path relative to root, POSIX style ("a/b.md");
       metadata = {"source": <same relative path>, "suffix": ".md", "sha256": <hex of the bytes>}
     * .jsonl: one Document per non-blank line; each line is an object with "id" and "text";
       all other keys go into metadata, plus "source". id = f"{relative path}:{line id}".
       Invalid lines (bad JSON / missing keys) are logged as WARNING and skipped - one bad line
       must not kill a multi-hour ingestion job.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..models import Document

log = logging.getLogger("docmind.ingest")

SUPPORTED = frozenset({".txt", ".md", ".jsonl"})
IGNORE_DIRS = frozenset({".git", "__pycache__", ".venv", "node_modules"})


def iter_files(
    root: Path | str, suffixes: Iterable[str] = SUPPORTED, ignore_dirs: Iterable[str] = IGNORE_DIRS
) -> Iterator[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    return _iter_files(root, frozenset(s.lower() for s in suffixes), frozenset(ignore_dirs))


def _iter_files(root: Path, suffixes: frozenset[str], ignore: frozenset[str]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in ignore for part in rel_parts[:-1]) or path.name in ignore:
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def _decode(raw: bytes, name: str) -> str:
    try:
        return raw.decode("utf-8-sig")  # utf-8-sig == utf-8 that also strips a BOM
    except UnicodeDecodeError:
        log.warning("%s is not valid UTF-8; falling back to latin-1", name)
        return raw.decode("latin-1")


def read_text(path: Path | str) -> str:
    path = Path(path)
    return _decode(path.read_bytes(), path.name)


def load_documents(root: Path | str) -> Iterator[Document]:
    root = Path(root)
    files = iter_files(root)  # validates eagerly
    return _load(root, files)


def _load(root: Path, files: Iterator[Path]) -> Iterator[Document]:
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".jsonl":
            yield from _load_jsonl(path, rel)
        else:
            raw = path.read_bytes()
            yield Document(
                id=rel,
                text=_decode(raw, path.name),
                metadata={
                    "source": rel,
                    "suffix": path.suffix.lower(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
            )


def _load_jsonl(path: Path, rel: str) -> Iterator[Document]:
    with path.open(encoding="utf-8") as f:  # file objects iterate lazily, line by line
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                doc_id, text = str(obj.pop("id")), obj.pop("text")
                if not isinstance(text, str):
                    raise TypeError("text must be a string")
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                log.warning("%s:%d skipped invalid line (%s)", rel, lineno, e)
                continue
            yield Document(id=f"{rel}:{doc_id}", text=text, metadata={**obj, "source": rel})
