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
    raise NotImplementedError


async def embed_with_retries(texts: Sequence[str], embed: Embed, *, rounds: int) -> BatchReport:
    raise NotImplementedError


async def embed_all_or_nothing(texts: Sequence[str], embed: Embed) -> list[Vector]:
    raise NotImplementedError
