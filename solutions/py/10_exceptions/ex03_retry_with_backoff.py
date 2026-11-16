"""
Exercise 10.3 - Retrying transient LLM errors                    level: ★★☆
topics: exceptions.best_practices, exceptions.chaining, exceptions.logging,
        exceptions.raising, exceptions.try_except

Problem
-------
TransientError / PermanentError are given (think RateLimitError / AuthError).

1. class RetryError(Exception)
     __init__(self, attempts: int, last: BaseException)
     .attempts, .last ; str(err) == f"gave up after {attempts} attempts: {last!r}"

2. call_with_retry(fn, *, attempts=3, base_delay=0.5, max_delay=8.0,
                   sleep=time.sleep, logger=None) -> T
     * calls fn() (no args) until it returns; returns its value
     * attempts < 1 -> ValueError, before fn is ever called
     * ONLY TransientError is retried. Anything else (PermanentError, a ValueError bug,
       KeyboardInterrupt...) propagates immediately: the *same* exception object,
       unwrapped, after a single call, with no sleep
     * delay before retry k (k = 0 for the first retry):
         err.retry_after if it is not None, else base_delay * 2**k ; always capped at
         max_delay. sleep(delay) is called between attempts - never after the last one.
     * every failed attempt that will be retried is logged at WARNING on `logger`
       (default: logging.getLogger(__name__)), mentioning the attempt number and delay
     * when all attempts fail: log at ERROR, then
         raise RetryError(attempts, last_error) from last_error

Interviewers look for
---------------------
* Retry only what's retryable - retrying a 401 or a bug just burns quota and hides it.
* Never catch BaseException in a retry loop: Ctrl-C / task cancellation must stop it.
* `raise ... from last` keeps the real root cause in the traceback of the summary error.
* Injecting `sleep` (and a logger) makes timing logic testable without real waiting.
* Bounded exponential backoff + server-provided Retry-After; add jitter in production.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

__stub_keep__ = ["TransientError", "PermanentError"]


class TransientError(Exception):
    """Retryable failure, e.g. HTTP 429/503. retry_after is in seconds (server hint)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """Non-retryable failure, e.g. HTTP 401."""


class RetryError(Exception):
    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(f"gave up after {attempts} attempts: {last!r}")
        self.attempts = attempts
        self.last = last


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], object] = time.sleep,
    logger: logging.Logger | None = None,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    log = logger or logging.getLogger(__name__)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TransientError as e:  # everything else propagates untouched
            if attempt == attempts:
                log.error("attempt %d/%d failed: %s; giving up", attempt, attempts, e)
                raise RetryError(attempts, e) from e
            hint = e.retry_after
            delay = min(hint if hint is not None else base_delay * 2 ** (attempt - 1), max_delay)
            log.warning("attempt %d/%d failed: %s; retrying in %.2fs", attempt, attempts, e, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
