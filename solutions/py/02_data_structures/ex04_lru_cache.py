"""
Exercise 02.4 - LRU cache for embeddings                            level: ★★☆
topics: data_structures.ordereddict, data_structures.dict_lookup, data_structures.complexity,
        data_structures.namedtuple

Problem
-------
Embedding calls cost money and latency; the same chunks/queries repeat. Cache them.

class LRUCache(capacity: int)
  * capacity <= 0 -> ValueError
  * get(key, default=None)
      - hit: return the value and mark the key MOST recently used; counts a hit
      - miss: return default; counts a miss
      - a stored value of None is a legitimate hit (don't confuse it with a miss)
  * put(key, value) -> tuple[key, value] | None
      - insert or update; the key becomes most recently used
      - if the cache is over capacity, evict the LEAST recently used entry and return it
        as (key, value); otherwise return None
  * `key in cache` -> bool  : does NOT change recency or stats
  * len(cache) -> int
  * keys() -> list          : keys from least to most recently used
  * stats() -> CacheStats   : CacheStats(hits, misses, size) (namedtuple given below)
  * every operation is O(1)

Interviewers look for
---------------------
* OrderedDict.move_to_end(key) and popitem(last=False) are both O(1) - it is a dict plus a
  doubly linked list. (A plain dict keeps insertion order but can't cheaply move a key to
  the end without delete + reinsert, and has no "pop oldest".)
* The from-scratch version (dict + doubly linked list) is a classic interview follow-up.
* Sentinel for "missing" so cached None values aren't treated as misses.
* functools.lru_cache exists - but only for hashable function args, per-process, no TTL,
  no visibility into individual entries.
"""

from collections import OrderedDict, namedtuple
from collections.abc import Hashable
from typing import Any

CacheStats = namedtuple("CacheStats", ["hits", "misses", "size"])

_MISSING = object()


class LRUCache:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        value = self._data.get(key, _MISSING)
        if value is _MISSING:
            self._misses += 1
            return default
        self._data.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: Hashable, value: Any) -> tuple[Hashable, Any] | None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            return self._data.popitem(last=False)
        return None

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[Hashable]:
        return list(self._data)

    def stats(self) -> CacheStats:
        return CacheStats(self._hits, self._misses, len(self._data))
