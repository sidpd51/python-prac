"""
Exercise 02.3 - Sliding-window rate limiter and streaming stats     level: ★★☆
topics: data_structures.deque, data_structures.stack_queue, data_structures.complexity

Problem
-------
LLM providers enforce "N requests per T seconds". Protect yourself client-side.

1. class RateLimiter(limit: int, window: float)
     * limit <= 0 or window <= 0 -> ValueError (at construction)
     * allow(now: float) -> bool
         - sliding-window log: a request at `now` is allowed if fewer than `limit` ALLOWED
           requests happened in the half-open window (now - window, now]
         - allowed requests are recorded; rejected ones are not
         - `now` must be non-decreasing across calls, else ValueError
     * remaining(now: float) -> int : how many more requests would be allowed at `now`
       (does not record anything)
     * each call must be amortised O(1): evict expired timestamps from the left end

2. moving_average(values: Iterable[float], n: int) -> list[float]
     * for each input, the mean of the last min(i + 1, n) values (warm-up uses fewer)
     * O(1) per item: keep a running sum, don't re-sum the window
     * n <= 0 -> ValueError

3. tail(lines: Iterable[str], n: int) -> list[str]
     * the last n items of a possibly huge stream (like `tail -n`), O(n) memory
     * n == 0 -> []; n < 0 -> ValueError

Interviewers look for
---------------------
* deque: O(1) append/pop at BOTH ends. list.pop(0) / insert(0, x) are O(n) - using a list
  as a FIFO queue is the classic performance bug.
* deque(maxlen=n) silently discards from the opposite end - a bounded buffer for free.
* Sliding log vs fixed window (bursty at boundaries) vs token bucket (O(1) memory).
* Amortised analysis: each timestamp is appended once and popped at most once.
"""

from collections import deque
from collections.abc import Iterable


class RateLimiter:
    def __init__(self, limit: int, window: float) -> None:
        if limit <= 0 or window <= 0:
            raise ValueError("limit and window must be positive")
        self.limit = limit
        self.window = window
        self._log: deque[float] = deque()
        self._last = float("-inf")

    def allow(self, now: float) -> bool:
        _expire(self, now)
        if len(self._log) < self.limit:
            self._log.append(now)
            return True
        return False

    def remaining(self, now: float) -> int:
        _expire(self, now)
        return self.limit - len(self._log)


def _expire(limiter: RateLimiter, now: float) -> None:
    if now < limiter._last:
        raise ValueError("time went backwards")
    limiter._last = now
    cutoff = now - limiter.window
    log = limiter._log
    while log and log[0] <= cutoff:  # each timestamp is popped at most once
        log.popleft()


def moving_average(values: Iterable[float], n: int) -> list[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    window: deque[float] = deque()
    total = 0.0
    out: list[float] = []
    for v in values:
        window.append(v)
        total += v
        if len(window) > n:
            total -= window.popleft()
        out.append(total / len(window))
    return out


def tail(lines: Iterable[str], n: int) -> list[str]:
    if n < 0:
        raise ValueError("n must be >= 0")
    return list(deque(lines, maxlen=n))
