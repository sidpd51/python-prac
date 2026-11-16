"""
Exercise 16.4 - Timeouts, fallbacks and a latency budget          level: ★★☆
topics: async.wait_for, async.asyncio_timeout, async.timeouts, async.timeout_cancellation, async.best_practices

Problem
-------
`call` arguments are zero-arg async callables (e.g. `lambda: llm.complete(prompt)`).

1. `async def call_with_fallback(call, *, timeout, fallback)` - use `asyncio.wait_for`.
   Return the call's result, or `fallback` if it takes longer than `timeout` seconds.
   Only the timeout is handled: any other exception from `call` propagates unchanged.
   (The timed-out call must be cancelled - wait_for does that for you.)

2. `async def retry_with_timeout(call, *, attempts, per_attempt_timeout, backoff=0.0)` -
   each attempt gets its own timeout. Retry on `TimeoutError` and `ConnectionError`; any
   other exception propagates immediately. Sleep `backoff * 2**k` seconds before retry
   number k+1 (k = 0, 1, ...). After `attempts` failed attempts re-raise the LAST error.
   attempts < 1 -> ValueError.

3. `async def answer_within_budget(question, *, retrieve, generate, budget) -> Answer` -
   a RAG request with ONE latency budget shared by both steps (use `asyncio.timeout`):
       docs = await retrieve(question)          # -> list[str]
       text = await generate(question, docs)    # -> str
   * finished in time -> Answer(text, docs, timed_out=False)
   * budget exceeded at any step -> Answer(FALLBACK_TEXT, docs retrieved so far
     (or []), timed_out=True)
   * if the *caller* cancels this coroutine, the CancelledError must propagate - do not
     turn an outer cancellation into a fallback answer.

Interviewers look for
---------------------
* A timeout is implemented as cancellation: the inner coroutine receives CancelledError at
  its current await, so its finally/cleanup runs - then the timeout surfaces as TimeoutError
  (builtin TimeoutError; asyncio.TimeoutError is an alias since 3.11).
* Per-call timeouts vs an overall deadline: three 2 s steps each with a 2 s timeout can take
  6 s; asyncio.timeout() around the whole block enforces a single budget.
* Never `except BaseException`/bare except around awaits: you'd swallow CancelledError.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

FALLBACK_TEXT = "Sorry, that took too long - please try again."


@dataclass
class Answer:
    text: str
    docs: list[str] = field(default_factory=list)
    timed_out: bool = False


async def call_with_fallback[T](call: Callable[[], Awaitable[T]], *, timeout: float, fallback: T) -> T:
    try:
        return await asyncio.wait_for(call(), timeout)
    except TimeoutError:
        return fallback


async def retry_with_timeout[T](
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    per_attempt_timeout: float,
    backoff: float = 0.0,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: Exception | None = None
    for k in range(attempts):
        if k and backoff:
            await asyncio.sleep(backoff * 2 ** (k - 1))
        try:
            return await asyncio.wait_for(call(), per_attempt_timeout)
        except (TimeoutError, ConnectionError) as e:
            last = e
    assert last is not None
    raise last


async def answer_within_budget(
    question: str,
    *,
    retrieve: Callable[[str], Awaitable[list[str]]],
    generate: Callable[[str, list[str]], Awaitable[str]],
    budget: float,
) -> Answer:
    docs: list[Any] = []
    try:
        async with asyncio.timeout(budget):
            docs = await retrieve(question)
            text = await generate(question, docs)
    except TimeoutError:
        # only OUR deadline becomes TimeoutError; an outer cancel stays CancelledError
        return Answer(FALLBACK_TEXT, docs, timed_out=True)
    return Answer(text, docs)
