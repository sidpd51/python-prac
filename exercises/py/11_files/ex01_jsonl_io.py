"""
Exercise 11.1 - JSONL datasets: write, append, stream            level: ★☆☆
topics: files.json, files.read_write, files.modes, files.with, files.open,
        files.open_close, files.encoding

Problem
-------
JSONL (one JSON object per line) is the format of fine-tuning sets, eval results and
batch-API requests. Implement:

1. write_jsonl(path, records: Iterable[Mapping[str, Any]]) -> int
     * (over)writes the file; one compact JSON object per line, each ending in "\\n"
     * UTF-8, non-ASCII kept as-is (ensure_ascii=False: "café", not "caf\\u00e9")
     * returns the number of records written

2. append_jsonl(path, record) -> None
     * adds one line at the end; creates the file if it doesn't exist

3. iter_jsonl(path, *, skip_invalid=False, errors=None) -> Iterator[dict[str, Any]]
     * LAZY: yields one dict per line, never reads the whole file into memory
     * blank / whitespace-only lines are ignored
     * a line that isn't valid JSON, or is valid JSON but not an object (e.g. [1, 2]):
         - skip_invalid=False -> ValueError(f"{path}:{lineno}: ...") chained from the
           original error where there is one (lineno is 1-based)
         - skip_invalid=True  -> skipped; if `errors` (a list) is given, append
           (lineno, stripped_line) to it
     * a missing file raises FileNotFoundError when iter_jsonl() is CALLED, not on the
       first next()
     * the file must be closed once the iterator is exhausted (or closed early)

Interviewers look for
---------------------
* Always pass encoding= explicitly - the default is platform/locale dependent.
* `with` guarantees close() even on exceptions; why leaked handles matter at scale
  ("Too many open files").
* Mode semantics: "w" truncates, "a" appends (and creates), "x" fails if it exists.
* Iterating a file object is line-by-line and lazy; f.read()/readlines() is not.
"""

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    raise NotImplementedError


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    raise NotImplementedError


def iter_jsonl(
    path: str | Path, *, skip_invalid: bool = False, errors: list[tuple[int, str]] | None = None
) -> Iterator[dict[str, Any]]:
    raise NotImplementedError
