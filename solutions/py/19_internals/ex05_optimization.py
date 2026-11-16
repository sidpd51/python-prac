"""
Exercise 19.5 - Make it O(n): rewriting slow eval-pipeline code        level: ★★★
topics: internals.optimization, internals.cpython

Problem
-------
Three hot spots from a retrieval-eval script. Keep the exact behaviour, fix the complexity.
(Tests count how often your code compares or reads objects, so an O(n^2) version fails
even if it is "fast enough" on small inputs.)

1. dedupe_preserve_order(items, key=None) -> list
   Original:
       out = []
       for x in items:
           if x not in out:        # list membership: O(len(out)) each time
               out.append(x)
   Keep the first occurrence of each item (or of each key(item) when key is given),
   preserving order, in O(n) using hashing. Items (or keys) are hashable.

2. join_results(queries, docs) -> list[tuple[str, str]]
   Original (O(len(queries) * len(docs)) - 10k x 100k = a billion comparisons):
       out = []
       for q in queries:
           for d in docs:
               if d["id"] == q["doc_id"]:
                   out.append((q["qid"], d["title"]))
       return out
   Same output, same order (queries order, then docs order for duplicate doc ids;
   queries with no matching doc contribute nothing) in O(len(queries) + len(docs)):
   build an index once. Read each doc's fields only a constant number of times.

3. top_k_frequent(tokens, k) -> list[tuple[str, int]]
   The k most common tokens with counts, most common first; ties broken by first
   appearance in `tokens`. k <= 0 -> []. Don't sort the full vocabulary by hand in a
   Python loop - collections.Counter (and heapq.nlargest inside most_common) exist.

Interviewers look for
---------------------
* Algorithmic wins (right data structure: set/dict O(1) average lookup) beat every
  micro-optimisation. Profile first, then fix the complexity.
* Then, in a measured hot loop: bind attribute lookups to locals (`append = out.append`),
  prefer built-ins / comprehensions (C loops), `"".join(parts)` instead of `+=` in a loop,
  avoid repeated work (cache with functools.lru_cache / cache), vectorise with NumPy.
* Always preserve behaviour - ordering and duplicate semantics are where rewrites break.
"""

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


def dedupe_preserve_order(
    items: Iterable[T], key: Callable[[T], Hashable] | None = None
) -> list[T]:
    seen: set[Hashable] = set()
    out: list[T] = []
    add, append = seen.add, out.append  # local binding in the hot loop
    for item in items:
        k = item if key is None else key(item)
        if k not in seen:
            add(k)
            append(item)
    return out


def join_results(
    queries: Iterable[Mapping[str, Any]], docs: Iterable[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    titles_by_id: dict[Any, list[str]] = {}
    for d in docs:
        titles_by_id.setdefault(d["id"], []).append(d["title"])
    return [
        (q["qid"], title)
        for q in queries
        for title in titles_by_id.get(q["doc_id"], ())
    ]


def top_k_frequent(tokens: Iterable[str], k: int) -> list[tuple[str, int]]:
    if k <= 0:
        return []
    # Counter preserves first-insertion order and most_common() is stable for ties
    return Counter(tokens).most_common(k)
