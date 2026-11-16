"""
Exercise 16.2 - Fix the blocked event loop (debug)                level: ★★☆
topics: async.blocking, async.sleep_vs_sleep, async.event_loop, async.event

Problem
-------
A teammate's FastAPI-style service "uses async" but requests still pile up. This is
their code - find why, then write the fixed versions below:

    async def count_tokens_all(texts):
        return [legacy_count_tokens(t) for t in texts]     # sync, ~50 ms each

    async def wait_until_ready(probe, *, interval, timeout):
        start = time.monotonic()
        while not await probe():
            if time.monotonic() - start > timeout:
                raise TimeoutError
            time.sleep(interval)

    MODEL_READY = False                                     # global flag + polling
    async def handle(prompt):
        while not MODEL_READY:
            await asyncio.sleep(0.001)
        ...

1. `async def count_tokens_all(texts) -> list[int]` - `legacy_count_tokens` (given) is a
   blocking, synchronous library call you cannot rewrite. Run the calls in worker threads
   (`asyncio.to_thread`) concurrently, so the event loop stays responsive and the total
   time is ~one call, not the sum. Results in input order.

2. `async def wait_until_ready(probe, *, interval, timeout) -> int` - `probe` is an async
   callable returning bool (e.g. a health check). Poll it every `interval` seconds WITHOUT
   blocking the loop. Return the number of probe calls it took. If it is not ready within
   `timeout` seconds raise `TimeoutError`.

3. `class WarmupGate` - replace the global-flag busy-poll with an `asyncio.Event`:
   * `open()` - mark the model as loaded; wakes every waiter at once
   * `is_open` - property, bool
   * `async wait(timeout=None) -> bool` - wait until open; return True, or False if
     `timeout` (seconds) elapses first (do not raise).

Interviewers look for
---------------------
* The event loop is single-threaded: any sync call that doesn't return quickly (time.sleep,
  requests.get, a heavy CPU loop, a sync DB driver) freezes EVERY coroutine on that loop.
* `asyncio.sleep` suspends only the current coroutine; `time.sleep` blocks the thread.
* `asyncio.to_thread` / `loop.run_in_executor` for blocking I/O; a process pool for heavy
  CPU work (the GIL still serialises pure-Python CPU work in threads).
* Event = broadcast "this happened" signal; no polling, no wasted wake-ups.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

TOKENIZE_DELAY = 0.05


def legacy_count_tokens(text: str) -> int:
    """A blocking sync library call (think: a C tokenizer or a sync HTTP client)."""
    time.sleep(TOKENIZE_DELAY)
    return len(text.split())


async def count_tokens_all(texts: Sequence[str]) -> list[int]:
    raise NotImplementedError


async def wait_until_ready(probe: Callable[[], Awaitable[bool]], *, interval: float, timeout: float) -> int:
    raise NotImplementedError


class WarmupGate:
    """Requests wait here until the model has finished loading."""

    def __init__(self) -> None:
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        raise NotImplementedError

    async def wait(self, timeout: float | None = None) -> bool:
        raise NotImplementedError
