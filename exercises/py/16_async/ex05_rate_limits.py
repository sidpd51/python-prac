"""
Exercise 16.5 - Bounded concurrency and an async token bucket    level: ★★☆
topics: async.semaphore, async.lock, async.io_bound

Problem
-------
LLM providers limit you two ways: max concurrent requests, and requests per second.

1. `async def bounded_map(fn, items, *, limit) -> list` - apply the async `fn` to every
   item concurrently, but with at most `limit` calls in flight at any moment
   (`asyncio.Semaphore`). Results in input order. limit < 1 -> ValueError (raise it
   before starting any work). If one call raises, the exception propagates (gather
   semantics are fine).

2. `class TokenBucket(rate, capacity)` - an async rate limiter:
   * the bucket starts full with `capacity` tokens and refills continuously at `rate`
     tokens/second, never above capacity
   * `async acquire(tokens=1)` waits (asyncio.sleep, never time.sleep) until `tokens`
     are available, then takes them. tokens > capacity -> ValueError (it could never
     succeed); rate <= 0 or capacity < 1 -> ValueError in __init__
   * many coroutines call acquire() concurrently: no more tokens may be handed out than
     the bucket allows, and waiters are served in FIFO order - guard the refill/check/take
     sequence with an `asyncio.Lock` (hold it while waiting, so later callers queue up).
   * use the event loop clock: `asyncio.get_running_loop().time()`

Interviewers look for
---------------------
* Semaphore limits *concurrency* (in-flight), a token bucket limits *throughput* (rate);
  production clients usually need both.
* `async with sem:` releases even when the call raises or is cancelled.
* asyncio.Lock is not thread-safe and isn't reentrant; it's needed when a critical section
  contains an `await` (otherwise single-threaded asyncio code can't interleave there).
* Check-then-act across an await is the classic asyncio race.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable


async def bounded_map[T, R](fn: Callable[[T], Awaitable[R]], items: Iterable[T], *, limit: int) -> list[R]:
    raise NotImplementedError


class TokenBucket:
    def __init__(self, rate: float, capacity: int) -> None:
        raise NotImplementedError

    async def acquire(self, tokens: int = 1) -> None:
        raise NotImplementedError
