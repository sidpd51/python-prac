"""
Exercise 18.5 - A weak-reference model cache                         level: ★★★
topics: memory.weakref, memory.refcounting, memory.gc

Problem
-------
An inference service loads models (tokenizers, rerankers, embedding models) on demand.
Loading is slow, so concurrent requests should share one instance - but a model nobody
uses any more should be freed, not pinned in RAM forever by the cache.

Build ModelRegistry(loader), where loader(name) -> Model (Model is given):

  * get(name) -> Model
      - returns the cached instance while ANY strong reference to it exists elsewhere
      - otherwise calls loader(name), caches it WEAKLY, and returns it
      - thread-safe: N threads calling get("x") at the same time trigger exactly ONE
        loader call and all receive the same object
  * loads -> int (property): how many times loader was called
  * `name in registry` -> True only while that model is alive; len(registry) -> number
    of live models
  * evicted -> list[str]: names of models in the order they were freed - recorded by a
    weakref.finalize callback registered when the model is loaded
  * close() -> detach all pending finalizers (no more `evicted` entries after close)
    and empty the cache. Idempotent.

Use weakref.WeakValueDictionary for the cache.

Classic bug to avoid:
    self._cache[name] = self._loader(name)
    return self._cache[name]          # KeyError! nothing else holds the model, so it
                                      # died the moment the assignment finished
Keep a strong local reference until you return.

Interviewers look for
---------------------
* weakref.ref / WeakValueDictionary / WeakKeyDictionary / WeakSet: refer to an object
  without keeping it alive; entries vanish when the referent is freed.
* weakref.finalize(obj, fn, *args) must not capture obj (a bound method of obj or obj
  in args would keep it alive forever). It's safer than __del__: runs at most once,
  can be detached, and runs at interpreter exit if still alive.
* Not every object is weak-referenceable (int, str, tuple, list; __slots__ classes need
  '__weakref__').
* When you DO want to keep models resident, a bounded LRU (strong refs) is the right
  tool - weak caching only dedups what's already in use.
"""

import threading
import weakref
from collections.abc import Callable

__stub_keep__ = ["Model"]


class Model:
    """Stand-in for a heavyweight loaded model."""

    def __init__(self, name: str, size_mb: int = 100) -> None:
        self.name = name
        self.size_mb = size_mb

    def __repr__(self) -> str:
        return f"Model({self.name!r})"


class ModelRegistry:
    def __init__(self, loader: Callable[[str], Model]) -> None:
        self._loader = loader
        self._cache: weakref.WeakValueDictionary[str, Model] = weakref.WeakValueDictionary()
        self._finalizers: list[weakref.finalize] = []
        self._lock = threading.Lock()
        self._loads = 0
        self.evicted: list[str] = []

    def get(self, name: str) -> Model:
        with self._lock:  # one lock: simple; per-name locks would let different models load in parallel
            model = self._cache.get(name)
            if model is None:
                model = self._loader(name)
                self._loads += 1
                self._cache[name] = model
                # callback args reference the list and the name - never the model itself
                self._finalizers = [f for f in self._finalizers if f.alive]
                self._finalizers.append(weakref.finalize(model, self.evicted.append, name))
            return model

    @property
    def loads(self) -> int:
        return self._loads

    def __contains__(self, name: object) -> bool:
        return name in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        with self._lock:
            for fin in self._finalizers:
                fin.detach()
            self._finalizers.clear()
            self._cache.clear()
