"""
Exercise 05.4 - memoize: an embedding cache held in a closure       level: ★★☆
topics: closures.closures, closures.free_variables, closures.nonlocal, decorators.wraps

Problem
-------
Embedding the same text twice costs money. Write `memoize(func)` - a decorator whose cache
lives in the decorator's closure (NOT a global, NOT an attribute holding the dict).

    @memoize
    def embed(text, model="small"): ...

* Cache key: positional args + keyword args, where `f(a, k=1, j=2)` and `f(a, j=2, k=1)`
  are the same key (sort the kwargs). Positional vs keyword spelling of the same argument
  may be different keys - that's fine.
* Unhashable arguments (e.g. a list) -> just call `func` (no caching) and don't count it as
  a hit or a miss.
* Exceptions raised by `func` are NOT cached (the next call tries again) but count as a miss.
* The wrapper exposes two functions as attributes, both closures over the same state:
      embed.cache_info()  -> (hits, misses)   a tuple of ints
      embed.cache_clear() -> None             empties the cache AND resets the counters
* Preserve __name__, __doc__ and __wrapped__ (functools.wraps).

Don't use functools.lru_cache / functools.cache (that's what you'd use in real life).

Interviewers look for
---------------------
* The decorator's local variables (cache dict, counters) become free variables of wrapper:
  one cache per decorated function, created once at decoration time.
* Mutating the dict needs no `nonlocal`; rebinding the hit/miss ints does.
* Why kwargs order must not matter, and why unhashable args can't be keys.
* Real-world: unbounded caches leak memory -> lru_cache(maxsize=...), TTLs, or a vector-store cache.
"""

from collections.abc import Callable
from typing import Any


def memoize(func: Callable[..., Any]) -> Callable[..., Any]:
    raise NotImplementedError
