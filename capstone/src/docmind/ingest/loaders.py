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
    raise NotImplementedError


def read_text(path: Path | str) -> str:
    raise NotImplementedError


def load_documents(root: Path | str) -> Iterator[Document]:
    raise NotImplementedError
