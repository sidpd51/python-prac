"""
Exercise 06.6 - @retry with backoff, for sync AND async functions   level: ★★★
topics: decorators.with_arguments, decorators.wraps, decorators.args_kwargs, decorators.higher_order

Problem
-------
Write `retry`, usable BOTH bare and with arguments:

    @retry
    def fetch(): ...                                      # defaults

    @retry(times=5, exceptions=(TimeoutError, ConnectionError), backoff=0.5)
    async def acomplete(prompt): ...                      # async works too

Signature:
    retry(func=None, /, *, times=3, exceptions=(Exception,), backoff=0.0,
          sleep=time.sleep, async_sleep=asyncio.sleep)

* `times` = total attempts (times=3 -> 1 call + up to 2 retries). times < 1 -> ValueError
  raised by retry(...) itself, before any function is decorated.
* Only exceptions matching `exceptions` (a class or a tuple of classes) are retried; any
  other exception propagates immediately with no further attempts.
* Before retry number k (k = 1, 2, ...) wait backoff * 2 ** (k - 1) seconds:
  backoff=0.5 -> waits 0.5, 1.0, 2.0 ... Don't sleep when backoff == 0 and never sleep
  after the final attempt.
* Sync functions sleep via `sleep(seconds)`; async functions via `await async_sleep(seconds)`
  (never block the event loop with time.sleep). Both are injectable for tests.
* If all attempts fail, re-raise the LAST exception itself (not a wrapper exception).
* The decorated async function must still be a coroutine function
  (inspect.iscoroutinefunction(...) is True), and metadata is preserved (functools.wraps).

Interviewers look for
---------------------
* The `func=None` trick: `@retry` passes the function positionally; `@retry(...)` passes
  only keywords and returns a decorator.
* A sync wrapper around an async function would return the coroutine un-awaited - retries
  would never happen. Detect with inspect.iscoroutinefunction and build an async wrapper.
* Exponential backoff (+ jitter in production), retrying only transient errors (timeouts,
  429, 5xx - not 400/401), idempotency of what you retry, and a cap on total time.
"""

import asyncio
import functools
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any


def retry(
    func: Callable[..., Any] | None = None,
    /,
    *,
    times: int = 3,
    exceptions: type[BaseException] | tuple[type[BaseException], ...] = (Exception,),
    backoff: float = 0.0,
    sleep: Callable[[float], Any] = time.sleep,
    async_sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> Any:
    if times < 1:
        raise ValueError("times must be >= 1")

    def delay(retry_number: int) -> float:
        return backoff * 2 ** (retry_number - 1)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                for attempt in range(1, times + 1):
                    try:
                        return await fn(*args, **kwargs)
                    except exceptions:
                        if attempt == times:
                            raise
                    if backoff > 0:
                        await async_sleep(delay(attempt))
                raise AssertionError("unreachable")

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
                if backoff > 0:
                    sleep(delay(attempt))
            raise AssertionError("unreachable")

        return wrapper

    if func is not None:  # used bare: @retry
        return decorator(func)
    return decorator
