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
    if limit < 1:
        raise ValueError("limit must be >= 1")
    sem = asyncio.Semaphore(limit)

    async def run(item: T) -> R:
        async with sem:
            return await fn(item)

    return list(await asyncio.gather(*(run(i) for i in items)))


class TokenBucket:
    def __init__(self, rate: float, capacity: int) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be > 0 and capacity >= 1")
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._updated: float | None = None  # loop clock is only available inside a loop
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        if tokens > self.capacity:
            raise ValueError("tokens exceeds bucket capacity")
        loop = asyncio.get_running_loop()
        async with self._lock:  # FIFO: asyncio.Lock wakes waiters in arrival order
            while True:
                now = loop.time()
                if self._updated is not None:
                    elapsed = now - self._updated
                    self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # sleep exactly until enough tokens will have accrued; others wait on the lock
                await asyncio.sleep((tokens - self._tokens) / self.rate)
