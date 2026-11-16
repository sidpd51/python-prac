"""
Exercise 04.1 - Ranking retrieval results with lambda/map/filter    level: ★☆☆
topics: functional.lambda, functional.map, functional.filter, functional.immutability

Problem
-------
A retriever returns results like {"doc_id": "d1", "score": 0.82, "source": "wiki"}.

1. sort_results(results) -> list[dict]
     * a NEW list sorted by score descending, ties broken by doc_id ascending
     * use sorted() with a lambda key (one sort, tuple key)

2. top_ids(results, min_score) -> list[str]
     * doc_ids of results with score >= min_score, in ranked order (as sort_results)
     * build it with filter() and map()

3. score_at_least(threshold) -> Callable[[dict], bool]
     * returns a predicate (a lambda) usable with filter(): r["score"] >= threshold

4. apply_boosts(results, boosts) -> list[dict]
     * boosts maps source -> multiplier (missing source -> 1.0); return NEW dicts with the
       boosted score (all other keys copied); the input dicts must not be modified
     * use map()

Interviewers look for
---------------------
* `key=lambda r: (-r["score"], r["doc_id"])` - a tuple key sorts by several fields in one
  pass; negating a number reverses just that field.
* Lambdas are single-expression anonymous functions - fine as short keys/predicates,
  but name anything non-trivial with def (better tracebacks, testable).
* map/filter return lazy ITERATORS in Python 3 - wrap in list() when you need a list.
* `{**r, "score": ...}` builds a new dict instead of mutating shared data.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

Result = Mapping[str, Any]


def sort_results(results: Iterable[Result]) -> list[Result]:
    raise NotImplementedError


def top_ids(results: Iterable[Result], min_score: float) -> list[str]:
    raise NotImplementedError


def score_at_least(threshold: float) -> Callable[[Result], bool]:
    raise NotImplementedError


def apply_boosts(results: Iterable[Result], boosts: Mapping[str, float]) -> list[dict[str, Any]]:
    raise NotImplementedError
