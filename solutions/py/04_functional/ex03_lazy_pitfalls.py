'''
Exercise 04.3 - Lazy map/filter pitfalls                            level: ★★☆
topics: functional.lazy_evaluation, functional.iterator_consumption, functional.map,
        functional.filter, functional.side_effects

Problem
-------
Three helpers from an ingestion job are subtly broken because map/filter are LAZY,
ONE-SHOT iterators:

```python
def embed_valid(docs, embed):
    valid = filter(lambda d: d.strip(), docs)
    print(f"embedding {len(list(valid))} docs")
    return list(map(embed, valid))          # always [] - `valid` was already consumed

def warm_cache(keys, load):
    map(load, keys)                          # loads nothing at all

def min_max(scores):                         # scores is often a generator
    return min(scores), max(scores)          # ValueError: max() arg is an empty sequence
```

Write correct versions, plus one helper:

1. embed_valid(docs, embed) -> tuple[int, list]
     * (number of docs that are not blank, [embed(doc) for each such doc] in order)
     * docs may be a one-shot iterator; embed must be called EXACTLY once per valid doc

2. warm_cache(keys, load) -> int
     * call load(key) for every key, eagerly, in order; return how many were loaded

3. min_max(scores) -> tuple[float, float]
     * ONE pass over any iterable (generators included); empty -> ValueError

4. peek(iterable) -> tuple[first, iterator]
     * look at the first item of a stream without losing it: returns (first, it) where `it`
       yields ALL items including the first, lazily
     * pulls exactly ONE item from the source when called
     * empty iterable -> ValueError
     * use case: sniff the first JSONL record to detect the schema, then process everything

Interviewers look for
---------------------
* map/filter/zip/generator expressions compute nothing until iterated, and can be
  iterated only once; a second pass silently yields nothing - no error.
* Using map() for side effects is a smell: it's lazy (does nothing) and allocates results
  you don't want. Side effects belong in a plain for loop.
* len(list(it)) consumes `it`. Materialise once (list(...)) if you need both a count and the
  items, or count while you go.
* itertools.chain([first], it) "un-reads" an item; itertools.tee buffers for multiple
  consumers (memory!).
'''

import itertools
from collections.abc import Callable, Iterable, Iterator
from typing import Any


def embed_valid[V](docs: Iterable[str], embed: Callable[[str], V]) -> tuple[int, list[V]]:
    valid = [d for d in docs if d.strip()]  # materialise once
    return len(valid), [embed(d) for d in valid]


def warm_cache(keys: Iterable[Any], load: Callable[[Any], Any]) -> int:
    count = 0
    for key in keys:  # side effects -> a plain loop, not map()
        load(key)
        count += 1
    return count


def min_max(scores: Iterable[float]) -> tuple[float, float]:
    it = iter(scores)
    try:
        lo = hi = next(it)
    except StopIteration:
        raise ValueError("min_max() of an empty iterable") from None
    for s in it:
        if s < lo:
            lo = s
        elif s > hi:
            hi = s
    return lo, hi


def peek[T](iterable: Iterable[T]) -> tuple[T, Iterator[T]]:
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("peek() of an empty iterable") from None
    return first, itertools.chain([first], it)
