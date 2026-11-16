"""
Exercise 07.4 - Fix the iterator-exhaustion bugs                  level: ★★☆
topics: iterators.exhaustion, iterators.one_pass, iterators.next, iterators.stopiteration

Problem
-------
All three helpers work in unit tests that pass lists, and break in production where
the data arrives as generators / DB cursors / map objects. Write fixed versions.

1. summarize_scores(scores) -> {"count": int, "mean": float, "max": float}
   Buggy:
       def summarize_scores(scores):
           return {"count": len(list(scores)), "mean": sum(scores) / len(list(scores)),
                   "max": max(scores)}
   With a generator, the first pass consumes everything; later passes see nothing.
   Fixed: ONE pass, O(1) memory (don't materialize the input), any iterable.
   Empty input -> ValueError.

2. filter_new(candidate_ids, seen_ids) -> list
   Buggy:
       def filter_new(candidate_ids, seen_ids):
           return [c for c in candidate_ids if c not in seen_ids]
   When `seen_ids` is an iterator, each `in` test consumes it up to the match, so later
   checks miss ids. Fixed: candidates in original order (duplicates kept) that are not in
   seen_ids; consume `seen_ids` exactly once; O(1) membership checks.

3. peek(iterable, default=None) -> tuple[first, iterator]
   Buggy:
       hits = map(rerank, retriever(query))
       if not hits:              # a map object is ALWAYS truthy
           return FALLBACK
   Fixed helper: return (first_item, it) where `it` yields ALL items INCLUDING the first
   (nothing lost). Empty input -> (default, an empty iterator). Must stay lazy: pull exactly
   one item from the source, and work on infinite iterators. (itertools.chain helps.)

Interviewers look for
---------------------
* Iterators are one-pass: a second traversal silently yields nothing - no error.
* `x in iterator` consumes items; `bool(iterator)` is always True; len() doesn't exist.
* Fixes: single-pass algorithms, materialize ONCE on purpose (set/list), or peek + chain.
* itertools.tee exists but buffers - it's not free memory-wise.
"""

from collections.abc import Hashable, Iterable, Iterator
from typing import Any


def summarize_scores(scores: Iterable[float]) -> dict[str, float]:
    raise NotImplementedError


def filter_new(candidate_ids: Iterable[Hashable], seen_ids: Iterable[Hashable]) -> list:
    raise NotImplementedError


def peek(iterable: Iterable[Any], default: Any = None) -> tuple[Any, Iterator[Any]]:
    raise NotImplementedError
