"""
Exercise 17.4 - Client-side rate limiting (token bucket)           level: ★★☆
topics: http.rate_limiting, http.concurrency_limits, http.concurrent_calls

Problem
-------
LLM providers limit requests/min (RPM) and tokens/min (TPM). Instead of hammering the API
and eating 429s, throttle on the client with a token bucket.

TokenBucket(rate, capacity, *, clock=time.monotonic, sleep=asyncio.sleep)
  * rate: tokens added per second (float > 0); capacity: max tokens (> 0).
    Invalid values -> ValueError.
  * starts FULL (capacity tokens): allows an initial burst of `capacity`.
  * refill is continuous: tokens = min(capacity, tokens + elapsed * rate), computed lazily
    from `clock()` whenever you look at the bucket - no background task.
  * clock/sleep are injected so tests can use a fake clock (never call time.* or
    asyncio.sleep directly).

  available() -> float          current token count (after refill)
  try_acquire(cost=1) -> bool   take `cost` tokens now if available, else False (no waiting)
  async acquire(cost=1) -> float
      * wait until `cost` tokens are available, take them, return the total seconds slept
        (0.0 if no wait was needed)
      * sleep for exactly the time needed for the deficit: (cost - tokens) / rate, then
        re-check (loop - never assume one sleep was enough)
      * cost > capacity can never succeed -> ValueError immediately; cost <= 0 -> ValueError
      * FIFO fairness: concurrent acquirers are served in arrival order - a big request that
        is waiting must not be starved by a stream of small ones (hint: hold an asyncio.Lock
        while waiting; asyncio.Lock wakes waiters in FIFO order)

async run_rate_limited(calls, bucket, *, cost=1) -> list
  * `calls` is a list of zero-arg async callables; run them all concurrently, each one
    first doing `await bucket.acquire(cost)`; return results in input order

Interviewers look for
---------------------
* Token bucket (bursts up to capacity, average = rate) vs fixed window (allows 2x bursts at
  window edges) vs sliding window log (exact but O(n) memory) vs leaky bucket (smooth).
* Rate limit (requests per time) is not a concurrency limit (requests in flight) - you
  often need both (Semaphore for the latter).
* TPM limits: cost = estimated tokens of the request, not 1.
* Client-side limiting is per process; multiple workers need a shared limiter (Redis) or
  a per-worker share of the quota - and you still must handle 429s.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any


class TokenBucket:
    def __init__(
        self,
        rate: float,
        capacity: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be > 0")
        self.rate = rate
        self.capacity = capacity
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(capacity)
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def available(self) -> float:
        self._refill()
        return self._tokens

    def _check_cost(self, cost: float) -> None:
        if cost <= 0:
            raise ValueError("cost must be > 0")
        if cost > self.capacity:
            raise ValueError(f"cost {cost} exceeds bucket capacity {self.capacity}")

    def try_acquire(self, cost: float = 1) -> bool:
        self._check_cost(cost)
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    async def acquire(self, cost: float = 1) -> float:
        self._check_cost(cost)
        waited = 0.0
        async with self._lock:  # FIFO: the head of the queue waits; others queue behind it
            while True:
                self._refill()
                if self._tokens >= cost:
                    self._tokens -= cost
                    return waited
                delay = (cost - self._tokens) / self.rate
                await self._sleep(delay)
                waited += delay


async def run_rate_limited(
    calls: list[Callable[[], Awaitable[Any]]], bucket: TokenBucket, *, cost: float = 1
) -> list[Any]:
    async def one(call: Callable[[], Awaitable[Any]]) -> Any:
        await bucket.acquire(cost)
        return await call()

    return await asyncio.gather(*(one(c) for c in calls))
