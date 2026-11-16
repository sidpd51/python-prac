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

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(
    path: str | Path,
    *,
    skip_invalid: bool = False,
    errors: list[tuple[int, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    f = open(path, encoding="utf-8")  # noqa: SIM115 - opened eagerly, closed by _lines
    return _records(f, str(path), skip_invalid, errors)


def _records(
    f: IO[str], name: str, skip_invalid: bool, errors: list[tuple[int, str]] | None
) -> Iterator[dict[str, Any]]:
    with f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if not skip_invalid:
                    raise ValueError(f"{name}:{lineno}: invalid JSON ({e.msg})") from e
                obj = None
            if not isinstance(obj, dict):
                if not skip_invalid:
                    raise ValueError(f"{name}:{lineno}: expected a JSON object")
                if errors is not None:
                    errors.append((lineno, line))
                continue
            yield obj
