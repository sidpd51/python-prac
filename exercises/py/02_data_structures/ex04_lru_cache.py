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

from collections import namedtuple
from collections.abc import Hashable
from typing import Any

CacheStats = namedtuple("CacheStats", ["hits", "misses", "size"])


class LRUCache:
    def __init__(self, capacity: int) -> None:
        raise NotImplementedError

    def get(self, key: Hashable, default: Any = None) -> Any:
        raise NotImplementedError

    def put(self, key: Hashable, value: Any) -> tuple[Hashable, Any] | None:
        raise NotImplementedError

    def __contains__(self, key: object) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def keys(self) -> list[Hashable]:
        raise NotImplementedError

    def stats(self) -> CacheStats:
        raise NotImplementedError
