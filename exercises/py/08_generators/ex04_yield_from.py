"""
Exercise 08.4 - yield from: flattening and sub-generator return values   level: ★★☆
topics: generators.yield_from, generators.chaining

Problem
-------
1. flatten(nested) -> lazily yields leaves of arbitrarily nested lists/tuples/sets/
   generators. Strings and bytes are LEAVES (don't iterate characters - why would
   naive code recurse forever on "a"?). Dicts are leaves too.
       list(flatten([1, [2, (3, [4])], "ab", iter([5])])) == [1, 2, 3, 4, "ab", 5]

2. Sub-generator return values:
   `counted(iterable)` is a generator that yields every item and then *returns* the
   number of items it yielded.
   `concat_with_counts(*iterables)` yields every item of every iterable (in order)
   using `yield from counted(...)`, and RETURNS a list of the per-iterable counts.
       g = concat_with_counts("ab", [1, 2, 3])
       list(g) -> ["a", "b", 1, 2, 3]; the generator's return value is [2, 3]
   (the return value travels in StopIteration.value)

Interviewers look for
---------------------
* `yield from` is not just a loop: it also forwards send()/throw() into the
  sub-generator and evaluates to the sub-generator's `return` value.
"""

from collections.abc import Generator, Iterable, Iterator
from typing import Any

LEAF_TYPES = (str, bytes, bytearray, dict)


def flatten(nested: Any) -> Iterator[Any]:
    raise NotImplementedError


def counted(iterable: Iterable[Any]) -> Generator[Any, None, int]:
    raise NotImplementedError


def concat_with_counts(*iterables: Iterable[Any]) -> Generator[Any, None, list[int]]:
    raise NotImplementedError
