"""
Exercise 16.3 - Partial failures in a batch of LLM calls         level: ★★☆
topics: async.gather_exceptions, async.return_exceptions, async.first_exception, async.task_cancel, async.cleanup_finally

Problem
-------
`embed` arguments below are async callables `embed(text) -> list[float]` that may raise
(rate limits, timeouts, bad input...).

1. `async def embed_partial(texts, embed) -> BatchReport` - call `embed` for every text
   concurrently with `asyncio.gather(..., return_exceptions=True)`. One bad text must not
   lose the other results:
   * `BatchReport.vectors[i]` is the vector for texts[i], or None if that call failed
   * `BatchReport.errors` maps index -> the exception instance raised for that text
   * only `Exception` subclasses count as per-item errors. If gather hands back a
     non-Exception BaseException (e.g. a CancelledError of a child), re-raise it - never
     report cancellation as "just a failed item".

2. `async def embed_with_retries(texts, embed, *, rounds) -> BatchReport` - like
   embed_partial, but after each round re-run ONLY the still-failed indices (concurrently),
   up to `rounds` total rounds (rounds >= 1, else ValueError). Errors in the report are the
   ones from the last attempt of each still-failing index.

3. `async def embed_all_or_nothing(texts, embed) -> list[list[float]]` - all-or-nothing
   semantics: start every call as a Task and wait with
   `asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)`. As soon as one call fails,
   cancel every still-pending task, wait until they have actually finished cancelling
   (so their `finally:` blocks ran), then raise the exception of the failed call (if several
   failed in the same wake-up, the one with the lowest index). On success return vectors in
   input order. Note plain `gather()` would propagate the first error but leave the
   siblings running in the background.

Interviewers look for
---------------------
* gather() without return_exceptions: the FIRST exception propagates to the awaiter, but the
  other awaitables are NOT cancelled - they keep running (and their errors may be "never
  retrieved").
* return_exceptions=True puts exception objects in the result list -> you must type-check
  every element.
* cancel() only *requests* cancellation; you must await the task to know it finished.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

Vector = list[float]
Embed = Callable[[str], Awaitable[Vector]]


@dataclass
class BatchReport:
    vectors: list[Vector | None]
    errors: dict[int, Exception] = field(default_factory=dict)


async def embed_partial(texts: Sequence[str], embed: Embed) -> BatchReport:
    results = await asyncio.gather(*(embed(t) for t in texts), return_exceptions=True)
    report = BatchReport(vectors=[None] * len(texts))
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            report.errors[i] = r
        elif isinstance(r, BaseException):
            raise r  # CancelledError / KeyboardInterrupt are not "item failures"
        else:
            report.vectors[i] = r
    return report


async def embed_with_retries(texts: Sequence[str], embed: Embed, *, rounds: int) -> BatchReport:
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    report = BatchReport(vectors=[None] * len(texts))
    todo = list(range(len(texts)))
    for _ in range(rounds):
        if not todo:
            break
        attempt = await embed_partial([texts[i] for i in todo], embed)
        still_failing: list[int] = []
        for pos, idx in enumerate(todo):
            if pos in attempt.errors:
                report.errors[idx] = attempt.errors[pos]
                still_failing.append(idx)
            else:
                report.vectors[idx] = attempt.vectors[pos]
                report.errors.pop(idx, None)
        todo = still_failing
    return report


async def embed_all_or_nothing(texts: Sequence[str], embed: Embed) -> list[Vector]:
    if not texts:
        return []
    tasks = [asyncio.create_task(embed(t)) for t in texts]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        # we were cancelled ourselves: don't leak the children
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    failed = [t for t in tasks if t in done and not t.cancelled() and t.exception() is not None]
    if failed:
        for t in pending:
            t.cancel()
        # wait for the cancellations to complete; return_exceptions so CancelledError
        # of the children doesn't propagate out of this gather
        await asyncio.gather(*pending, return_exceptions=True)
        exc = failed[0].exception()
        assert exc is not None
        raise exc
    return [t.result() for t in tasks]
