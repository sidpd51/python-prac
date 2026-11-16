"""
Exercise 19.3 - Threads for I/O: fan-out, partial failures, hedging   level: ★★☆
topics: internals.threadpool, internals.futures, internals.threading, internals.cpu_vs_io

Problem
-------
Blocking I/O (HTTP calls with a sync SDK, DB queries, file reads) releases the GIL, so a
thread pool gives real concurrency for it. Implement with concurrent.futures:

1. fetch_all(items, fetch, max_workers=8) -> list
   Call fetch(item) for every item concurrently (at most max_workers at a time) and return
   the results IN INPUT ORDER. If any call raises, the exception propagates.
   (Executor.map is made for this.)

2. embed_settled(texts, embed, max_workers=8) -> tuple[list, dict[int, Exception]]
   Like fetch_all but failures don't abort the batch: return (results, errors) where
   results[i] is embed(texts[i]) or None if it failed, and errors maps index -> the
   exception raised. Use submit() + as_completed() (map can't do per-item errors).

3. first_successful(endpoints, call, max_workers=4) -> result
   "Hedged request": call(endpoint) for all endpoints concurrently, return the FIRST
   successful result as soon as it arrives - do not wait for the slower calls to finish
   (hint: `with ThreadPoolExecutor()` waits for everything on exit; you need
   shutdown(wait=False, cancel_futures=True)). If every call fails raise
   ExceptionGroup("all endpoints failed", [the exceptions]).

4. TokenMeter - thread-safe usage counter shared by worker threads:
     add(model, tokens) - accumulate; totals() -> dict copy {model: tokens}
   Many threads call add() concurrently; no update may be lost (use threading.Lock).

Interviewers look for
---------------------
* Threads help I/O-bound work (the GIL is released while blocked on sockets/files/sleep),
  not pure-Python CPU-bound work on the default (GIL) build.
* map() preserves order and raises the first error when you reach it; as_completed()
  yields in completion order - keep a future->index mapping.
* `x += 1` / read-modify-write on shared state is not atomic: protect with a Lock.
* Bounded concurrency = max_workers; for asyncio code the equivalent is a Semaphore.
"""

import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def fetch_all(items: Iterable[T], fetch: Callable[[T], R], max_workers: int = 8) -> list[R]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fetch, items))


def embed_settled(
    texts: Sequence[str], embed: Callable[[str], R], max_workers: int = 8
) -> tuple[list[R | None], dict[int, Exception]]:
    results: list[R | None] = [None] * len(texts)
    errors: dict[int, Exception] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(embed, text): i for i, text in enumerate(texts)}
        for fut in as_completed(futures):
            i = futures[fut]
            exc = fut.exception()
            if exc is None:
                results[i] = fut.result()
            elif isinstance(exc, Exception):
                errors[i] = exc
            else:  # KeyboardInterrupt/SystemExit-style: don't swallow
                raise exc
    return results, errors


def first_successful(
    endpoints: Sequence[T], call: Callable[[T], R], max_workers: int = 4
) -> R:
    if not endpoints:
        raise ValueError("need at least one endpoint")
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures: list[Future[R]] = [pool.submit(call, ep) for ep in endpoints]
        errors: list[Exception] = []
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc is None:
                return fut.result()
            if not isinstance(exc, Exception):
                raise exc
            errors.append(exc)
        raise ExceptionGroup("all endpoints failed", errors)
    finally:
        # don't block on the slow calls; drop the ones that haven't started
        pool.shutdown(wait=False, cancel_futures=True)


class TokenMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, int] = {}

    def add(self, model: str, tokens: int) -> None:
        with self._lock:
            self._totals[model] = self._totals.get(model, 0) + tokens

    def totals(self) -> dict[str, int]:
        with self._lock:
            return dict(self._totals)
