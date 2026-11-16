"""
Exercise 18.3 - Measure it: tracemalloc, streaming vs loading         level: ★★☆
topics: memory.profiling, memory.generator_memory, memory.efficient_processing, memory.allocation

Problem
-------
"Use a generator, it's more memory-efficient" - prove it with numbers.

1. measure_peak(fn, *args, **kwargs) -> tuple[result, peak_bytes]
   Run fn(*args, **kwargs) under tracemalloc and return its result together with the
   PEAK traced memory during the call, measured relative to the traced memory at the
   start of the call (so it's ~0 for a function that allocates nothing).
     * if tracemalloc is NOT already tracing: start it, and always stop it afterwards
       (even if fn raises - the exception must propagate)
     * if it IS already tracing (someone else's profiling session): don't stop it -
       use tracemalloc.reset_peak() and leave tracing on

2. iter_docs(path) -> iterator of dicts, one per line of a JSONL file
   (each line is {"id": ..., "text": ...}); blank lines are skipped. Lazy: hold at
   most one line in memory, and keep the file open only while iterating.

3. corpus_stats(docs) -> {"docs": int, "tokens": int, "longest": int}
   tokens = whitespace-separated words in doc["text"] summed over all docs,
   longest = the max per-doc token count (0 for no docs). Consume `docs` in ONE pass
   without materialising it (it may be a generator over a 50 GB file).
   `corpus_stats_eager(path)` (given) is the "readlines + list" version for comparison.

4. biggest_allocation_site(fn) -> (filename, lineno, size_bytes)
   Take a tracemalloc snapshot, call fn() and KEEP its result alive, take a second
   snapshot, compare them by "lineno" and return the source line whose allocations grew
   the most. Same start/stop rule as measure_peak.

Interviewers look for
---------------------
* tracemalloc traces Python-level allocations (pymalloc + PyMem): it shows WHERE memory
  was allocated (file:line), current vs peak. RSS-based tools (psutil, memray for native
  allocations) answer different questions.
* Peak memory is what gets your container OOM-killed, not the final number.
* A generator pipeline makes peak memory O(one record) instead of O(file).
"""

import json
import tracemalloc
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

__stub_keep__ = ["corpus_stats_eager"]


def corpus_stats_eager(path: Path) -> dict[str, int]:
    """Reference "load everything" version - correct but O(file) memory."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    docs = [json.loads(line) for line in lines if line.strip()]
    counts = [len(d["text"].split()) for d in docs]
    return {"docs": len(docs), "tokens": sum(counts), "longest": max(counts, default=0)}


def measure_peak(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, int]:
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline, _ = tracemalloc.get_traced_memory()
        result = fn(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        return result, max(0, peak - baseline)
    finally:
        if not was_tracing:
            tracemalloc.stop()


def iter_docs(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:  # file objects are lazy line iterators
            if line.strip():
                yield json.loads(line)


def corpus_stats(docs: Iterable[dict[str, Any]]) -> dict[str, int]:
    n = tokens = longest = 0
    for doc in docs:
        count = len(doc["text"].split())
        n += 1
        tokens += count
        longest = max(longest, count)
    return {"docs": n, "tokens": tokens, "longest": longest}


def biggest_allocation_site(fn: Callable[[], Any]) -> tuple[str, int, int]:
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        before = tracemalloc.take_snapshot()
        result = fn()  # noqa: F841 - must stay alive until the second snapshot
        after = tracemalloc.take_snapshot()
    finally:
        if not was_tracing:
            tracemalloc.stop()
    ignore = [tracemalloc.Filter(False, tracemalloc.__file__)]
    stats = after.filter_traces(ignore).compare_to(before.filter_traces(ignore), "lineno")
    top = max(stats, key=lambda s: s.size_diff)
    frame = top.traceback[0]
    return frame.filename, frame.lineno, top.size_diff
