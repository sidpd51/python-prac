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

from collections.abc import Iterable


class RateLimiter:
    def __init__(self, limit: int, window: float) -> None:
        raise NotImplementedError

    def allow(self, now: float) -> bool:
        raise NotImplementedError

    def remaining(self, now: float) -> int:
        raise NotImplementedError


def moving_average(values: Iterable[float], n: int) -> list[float]:
    raise NotImplementedError


def tail(lines: Iterable[str], n: int) -> list[str]:
    raise NotImplementedError
