"""
Exercise 06.4 - @rate_limited(max_calls, period): sliding window  level: ★★☆
topics: decorators.with_arguments, decorators.wrappers, decorators.wrapping, decorators.wraps

Problem
-------
Your LLM provider allows N requests per minute. Write a decorator FACTORY

    @rate_limited(max_calls=60, period=60.0)
    def complete(prompt): ...

* Sliding window: a call is allowed if fewer than `max_calls` calls were ALLOWED in the last
  `period` seconds (a call at time t falls out of the window at t + period, i.e. when
  now - t >= period).
* Over the limit -> raise RateLimitExceeded (given below) WITHOUT calling the function; its
  `retry_after` is the number of seconds until the oldest call in the window expires.
  Rejected calls do not count. A call that raises DOES count (the provider saw it).
* Each decorated function gets its own window (two functions don't share a budget).
* `clock` (keyword-only, default time.monotonic) is injectable for tests - always read the
  time through it.
* max_calls < 1 or period <= 0 -> ValueError raised by rate_limited(...) itself.
* Must be thread-safe: 20 threads racing must never let more than max_calls through.
* Preserve metadata with functools.wraps.
(RateLimitExceeded is given - don't change it.)

Interviewers look for
---------------------
* Three layers: factory(config) -> decorator(func) -> wrapper(*args) - and what each closure
  captures. `@rate_limited` without parentheses would be a bug: why?
* Sliding window log (deque of timestamps) vs fixed window vs token bucket trade-offs.
* Check-then-act race: the check and the append must happen under one lock.
* monotonic clock, injectable for deterministic tests.
"""

import functools
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

__stub_keep__ = ["RateLimitExceeded"]


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limit exceeded, retry after {retry_after:.2f}s")
        self.retry_after = retry_after


def rate_limited(
    max_calls: int, period: float, *, clock: Callable[[], float] = time.monotonic
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    if period <= 0:
        raise ValueError("period must be > 0")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        calls: deque[float] = deque()  # one window per decorated function
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                now = clock()
                while calls and now - calls[0] >= period:
                    calls.popleft()
                if len(calls) >= max_calls:
                    raise RateLimitExceeded(calls[0] + period - now)
                calls.append(now)
            # call outside the lock: a slow request must not block other threads' checks
            return func(*args, **kwargs)

        return wrapper

    return decorator
