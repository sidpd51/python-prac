"""
Exercise 07.5 - Round-robin iterators: API-key rotation & result interleaving   level: ★★★
topics: iterators.custom, iterators.protocol, iterators.dunder_next, iterators.stopiteration, iterators.exhaustion

Problem
-------
Write two iterator CLASSES (implement the protocol by hand: no generator functions, no
itertools.cycle). Both: __iter__ returns self.

1. KeyRotator(keys)
   Spreads LLM requests over several API keys: an INFINITE iterator cycling through keys
   in order:  k1, k2, k3, k1, k2, ...
   * empty keys or duplicate keys -> ValueError
   * disable(key): take a key out of rotation (it got 401 / quota exhausted). Unknown or
     already-disabled key -> KeyError. The rotation continues with the key that would have
     come next among the remaining ones:
         k1, k2, k3 rotation; after next() returned k1:
           disable("k2") -> following next() returns k3
           disable("k1") -> following next() returns k2
   * `active` property -> tuple of the keys still in rotation, in rotation order
   * once every key is disabled, next() raises StopIteration (and keeps raising it)

2. RoundRobin(*iterables)
   Merges results from several retrievers (BM25, dense, web) fairly: one item from each
   source in turn, skipping sources as they run out:
       list(RoundRobin("ABC", "D", "EF")) == ["A", "D", "E", "B", "F", "C"]
   * lazy: never pull an item from a source before it's needed (sources may be infinite
     or expensive); call iter() on every source at construction.
   * after all sources are exhausted, keep raising StopIteration.

Interviewers look for
---------------------
* Infinite iterators are fine - the consumer decides when to stop (islice, zip, break).
* Mutating the collection behind an iterator while iterating is a classic bug - here it's
  designed in, so the index bookkeeping must be right.
* Once StopIteration is raised, a well-behaved iterator must not "come back to life".
* collections.deque rotation for round-robin scheduling.
"""

from collections.abc import Hashable, Iterable
from typing import Any


class KeyRotator:
    def __init__(self, keys: Iterable[Hashable]) -> None:
        raise NotImplementedError

    def __iter__(self) -> "KeyRotator":
        raise NotImplementedError

    def __next__(self) -> Hashable:
        raise NotImplementedError

    def disable(self, key: Hashable) -> None:
        raise NotImplementedError

    @property
    def active(self) -> tuple[Hashable, ...]:
        raise NotImplementedError


class RoundRobin:
    def __init__(self, *iterables: Iterable[Any]) -> None:
        raise NotImplementedError

    def __iter__(self) -> "RoundRobin":
        raise NotImplementedError

    def __next__(self) -> Any:
        raise NotImplementedError
