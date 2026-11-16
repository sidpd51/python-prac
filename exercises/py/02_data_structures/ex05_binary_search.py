"""
Exercise 02.5 - Binary search variants                              level: ★★★
topics: data_structures.binary_search, data_structures.searching, data_structures.complexity,
        data_structures.linear_search

Problem
-------
Implement 1-3 WITHOUT the bisect module (that's what the interviewer will ask for).

1. lower_bound(xs, x) -> int
     * xs is sorted ascending; return the first index i with xs[i] >= x
       (== len(xs) if there is none). Same contract as bisect.bisect_left.

2. upper_bound(xs, x) -> int
     * first index i with xs[i] > x. Same contract as bisect.bisect_right.
     * upper_bound(xs, x) - lower_bound(xs, x) == number of occurrences of x

3. first_true(lo, hi, pred) -> int
     * pred is monotone on [lo, hi): False ... False True ... True
     * return the smallest i in [lo, hi) with pred(i) True, or hi if there is none
     * at most ceil(log2(hi - lo)) + 1 calls to pred; lo >= hi -> return hi without calling pred
     * e.g. "the smallest chunk size whose prompt overflows the context window" or
       "the first checkpoint where the eval regressed" (git bisect)

4. events_between(timestamps, start, end) -> list
     * timestamps sorted ascending (may contain duplicates); return those in the
       half-open interval [start, end) in O(log n + k). Here you MAY use bisect.
     * start >= end -> []

Interviewers look for
---------------------
* Half-open [lo, hi) invariants: loop while lo < hi, mid = (lo + hi) // 2, then either
  hi = mid or lo = mid + 1. Off-by-one and infinite loops come from mixing conventions.
* Binary search works on any MONOTONE predicate, not just arrays ("binary search the answer").
* O(log n) requires random access AND sorted data. Sorting first costs O(n log n) - for a
  single lookup a linear scan O(n) is cheaper; for many lookups sort once (or use a dict/set
  for exact-match lookups, O(1)).
* In Python, mid overflow isn't an issue (big ints) but it is in Java/C - worth mentioning.
"""

from collections.abc import Callable, Sequence
from typing import Any


def lower_bound(xs: Sequence[Any], x: Any) -> int:
    raise NotImplementedError


def upper_bound(xs: Sequence[Any], x: Any) -> int:
    raise NotImplementedError


def first_true(lo: int, hi: int, pred: Callable[[int], bool]) -> int:
    raise NotImplementedError


def events_between(timestamps: Sequence[float], start: float, end: float) -> list[float]:
    raise NotImplementedError
