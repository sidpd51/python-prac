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


async def _cancel_all(tasks: set[asyncio.Task]) -> None:
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def first_success[T](calls: Sequence[Call[T]]) -> tuple[int, T]:
    if not calls:
        raise ValueError("need at least one call")
    index: dict[asyncio.Task, int] = {}
    for i, c in enumerate(calls):
        index[asyncio.create_task(c())] = i  # type: ignore[arg-type]
    pending: set[asyncio.Task] = set(index)
    errors: dict[int, Exception] = {}
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            winners = sorted((index[t], t) for t in done if t.exception() is None)
            for t in done:
                if (exc := t.exception()) is not None:
                    errors[index[t]] = exc  # type: ignore[assignment]
            if winners:
                i, task = winners[0]
                return i, task.result()
    finally:
        await _cancel_all(pending)  # also runs if WE are cancelled
    raise ExceptionGroup("all calls failed", [errors[i] for i in sorted(errors)])


async def hedged[T](call: Call[T], *, hedge_after: float, max_attempts: int = 2) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    running: set[asyncio.Task] = set()
    started = 0
    last_error: Exception | None = None

    def launch() -> None:
        nonlocal started
        running.add(asyncio.create_task(call()))  # type: ignore[arg-type]
        started += 1

    launch()
    try:
        while running:
            timeout = hedge_after if started < max_attempts else None
            done, _ = await asyncio.wait(running, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:  # hedge timer fired: nobody answered yet -> send a backup request
                launch()
                continue
            running -= done
            for t in done:
                if t.exception() is None:
                    return t.result()
                last_error = t.exception()  # type: ignore[assignment]
            if started < max_attempts:
                launch()
    finally:
        await _cancel_all(running)
    assert last_error is not None
    raise last_error


async def gather_within[T](calls: Sequence[Call[T]], *, deadline: float) -> tuple[dict[int, T], list[int]]:
    if not calls:
        return {}, []
    tasks = [asyncio.create_task(c()) for c in calls]  # type: ignore[arg-type]
    done, pending = await asyncio.wait(tasks, timeout=deadline, return_when=asyncio.ALL_COMPLETED)
    await _cancel_all(pending)
    results = {i: t.result() for i, t in enumerate(tasks) if t in done and t.exception() is None}
    late = sorted(i for i, t in enumerate(tasks) if t in pending)
    return results, late
