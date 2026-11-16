"""
Exercise 14.3 - Generic TTL cache, Result type, bounded TypeVar  level: ★★☆
topics: type_hints.generics, type_hints.typevar, type_hints.aliases,
        type_hints.protocol, type_hints.static_checking

Problem
-------
Uses Python 3.12 generic syntax (PEP 695): `class C[K, V]`, `def f[T](...)`,
`type Alias[T] = ...`. The Ok/Err/Result types and the HasScore protocol are given.

1. class TTLCache[K, V] - cache for embeddings / LLM responses
     TTLCache(ttl: float, *, max_size: int = 128, clock=time.monotonic)
     * ttl <= 0 or max_size < 1 -> ValueError
     * set(key, value): store with expiry clock() + ttl; re-setting a key refreshes it
     * get(key) -> V | None: None if missing or expired (expired entries are removed)
     * get_or_set(key, factory: Callable[[], V]) -> V: cached value, or call factory
       once, store and return its result
     * len(cache): number of NOT expired entries
     * when a set() would exceed max_size, drop expired entries first, then the
       oldest-inserted ones

2. try_call[T](fn: Callable[[], T]) -> Result[T]
     Ok(fn()) or Err(exception) for any Exception (BaseException propagates)

3. unwrap_or[T](result: Result[T], default: T) -> T

4. collect[T](results: Iterable[Result[T]]) -> Result[list[T]]
     Ok(list of values) if all are Ok, otherwise the FIRST Err (stop iterating there)

5. top_k[S: HasScore](items: Iterable[S], k: int) -> list[S]
     the k items with the highest .score, highest first, ties keep input order;
     k < 0 -> ValueError. The return type is list[S] - the caller's own type, not
     list[HasScore] - that's what the bound TypeVar buys you (mypy checks this).

Interviewers look for
---------------------
* TypeVar vs Union: `def first[T](xs: list[T]) -> T` links input and output types;
  `list[int | str] -> int | str` loses that.
* Bounded TypeVars (`S: HasScore`) vs constrained (`T: (str, bytes)`).
* Generic classes: the type parameters exist for the type checker; at runtime
  TTLCache[str, int] is the same class (type erasure), visible only via __type_params__ /
  __orig_class__ / get_args.
* Result/Either types as an alternative to exceptions for batch pipelines.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Ok[T]:
    value: T


@dataclass(frozen=True)
class Err:
    error: Exception


type Result[T] = Ok[T] | Err


class HasScore(Protocol):
    @property
    def score(self) -> float: ...


class TTLCache[K, V]:
    def __init__(
        self, ttl: float, *, max_size: int = 128, clock: Callable[[], float] = time.monotonic
    ) -> None:
        raise NotImplementedError

    def set(self, key: K, value: V) -> None:
        raise NotImplementedError

    def get(self, key: K) -> V | None:
        raise NotImplementedError

    def get_or_set(self, key: K, factory: Callable[[], V]) -> V:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def try_call[T](fn: Callable[[], T]) -> Result[T]:
    raise NotImplementedError


def unwrap_or[T](result: Result[T], default: T) -> T:
    raise NotImplementedError


def collect[T](results: Iterable[Result[T]]) -> Result[list[T]]:
    raise NotImplementedError


def top_k[S: HasScore](items: Iterable[S], k: int) -> list[S]:
    raise NotImplementedError
