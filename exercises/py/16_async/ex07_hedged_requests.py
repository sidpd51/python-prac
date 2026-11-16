"""
Exercise 16.7 - Racing replicas, hedged requests, deadlines        level: ★★★
topics: async.wait, async.first_completed, async.all_completed, async.done_pending, async.task_cancelled, async.cancellederror

Problem
-------
`calls` / `call` are zero-arg async callables (e.g. the same prompt sent to different
LLM providers or replicas). All three functions use `asyncio.wait`.

1. `async def first_success(calls) -> tuple[int, T]` - start all calls at once and return
   `(index, result)` of the first one to SUCCEED (FIRST_COMPLETED in a loop: a call that
   fails is ignored while others are still running). Before returning, cancel all the
   losers and wait for them to finish cancelling. If every call fails raise
   `ExceptionGroup("all calls failed", errors)` with the errors in INDEX order.
   Empty `calls` -> ValueError.

2. `async def hedged(call, *, hedge_after, max_attempts=2) -> T` - "tail-latency hedging"
   (Google's *The Tail at Scale*): start one attempt; every time `hedge_after` seconds pass
   with no attempt finished, start one more copy (up to `max_attempts` running copies).
   Return the first successful result and cancel the rest (await them). If an attempt
   fails, start a replacement immediately if the attempt budget allows, otherwise keep
   waiting for the others; if all `max_attempts` attempts failed re-raise the last error.

3. `async def gather_within(calls, *, deadline) -> tuple[dict[int, T], list[int]]` - the
   "multi-retriever" pattern: run all calls, wait at most `deadline` seconds
   (ALL_COMPLETED with a timeout - note asyncio.wait does NOT raise on timeout), then
   cancel whatever is still pending. Return `(results, late)` where `results` maps index ->
   result for calls that finished successfully in time and `late` is the sorted list of
   indices that had to be cancelled. Calls that raised are left out of both (log-and-skip).

Interviewers look for
---------------------
* asyncio.wait returns two SETS (done, pending) of tasks - it never raises the tasks'
  exceptions and never cancels anything for you; mapping back to inputs is your job.
* Pass Tasks, not coroutines, to asyncio.wait (coroutines are rejected since 3.11).
* Cancel losers AND await them: "fire-and-forget cancel" leaves work running / connections
  open and produces "Task was destroyed but it is pending" warnings.
* Hedging trades a little extra load for a much better p99.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence

type Call[T] = Callable[[], Awaitable[T]]


async def first_success[T](calls: Sequence[Call[T]]) -> tuple[int, T]:
    raise NotImplementedError


async def hedged[T](call: Call[T], *, hedge_after: float, max_attempts: int = 2) -> T:
    raise NotImplementedError


async def gather_within[T](calls: Sequence[Call[T]], *, deadline: float) -> tuple[dict[int, T], list[int]]:
    raise NotImplementedError
