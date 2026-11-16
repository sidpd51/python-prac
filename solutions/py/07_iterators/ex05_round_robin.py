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

from collections import deque
from collections.abc import Hashable, Iterable, Iterator
from typing import Any


class KeyRotator:
    def __init__(self, keys: Iterable[Hashable]) -> None:
        self._keys = list(keys)
        if not self._keys:
            raise ValueError("need at least one key")
        if len(set(self._keys)) != len(self._keys):
            raise ValueError("duplicate keys")
        self._index = 0  # position of the key returned by the next next()

    def __iter__(self) -> "KeyRotator":
        return self

    def __next__(self) -> Hashable:
        if not self._keys:
            raise StopIteration
        key = self._keys[self._index]
        self._index = (self._index + 1) % len(self._keys)
        return key

    def disable(self, key: Hashable) -> None:
        try:
            pos = self._keys.index(key)
        except ValueError:
            raise KeyError(key) from None
        del self._keys[pos]
        if pos < self._index:
            self._index -= 1  # everything after `pos` shifted left by one
        self._index = self._index % len(self._keys) if self._keys else 0

    @property
    def active(self) -> tuple[Hashable, ...]:
        return tuple(self._keys)


class RoundRobin:
    def __init__(self, *iterables: Iterable[Any]) -> None:
        self._sources: deque[Iterator[Any]] = deque(iter(it) for it in iterables)

    def __iter__(self) -> "RoundRobin":
        return self

    def __next__(self) -> Any:
        while self._sources:
            source = self._sources.popleft()
            try:
                item = next(source)
            except StopIteration:
                continue  # drop the exhausted source
            self._sources.append(source)
            return item
        raise StopIteration
