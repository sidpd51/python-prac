"""
Exercise 16.1 - Sequential vs concurrent embedding calls          level: ★☆☆
topics: async.sequential_vs_concurrent, async.gather, async.gather_ordering, async.create_task, async.coroutine_objects

Problem
-------
`fake_embed(text, delay)` (given) simulates an I/O-bound embedding API call: it records
the text in the module-level list CALLS *when its body starts running*, sleeps `delay`
seconds with asyncio.sleep, and returns a tiny "vector" `[float(len(text))]`.

1. `async def embed_sequential(texts)` -> list of vectors, one `await fake_embed(t)` after
   another. Total time ~ sum of delays.

2. `async def embed_concurrent(texts)` -> same result, but all calls run concurrently with
   `asyncio.gather`. Total time ~ max delay. Results MUST be in input order even though the
   calls finish in a different order. `delays` is an optional list (same length as texts) of
   per-text delays passed to fake_embed; when None use the default delay. Empty input -> [].

3. `def start_prefetch(texts)` - a *plain* (non-async) function that must be called while an
   event loop is running. It schedules one Task per text with `asyncio.create_task`
   (name each task `"embed-<index>"`) and returns the list of tasks WITHOUT awaiting them.
   The caller awaits them later.

4. `def embed_blocking_entrypoint(texts)` - a plain function for scripts/CLIs that runs
   `embed_concurrent(texts)` on a fresh event loop and returns the vectors (asyncio.run).

Interviewers look for
---------------------
* Calling a coroutine function only creates a coroutine object - nothing runs until it is
  awaited or wrapped in a Task. Awaiting coroutines one by one is still sequential!
* gather() returns results in the order of its arguments, not completion order.
* create_task() schedules the coroutine on the loop immediately; it starts running at the
  caller's next suspension point (await) - not at the create_task line itself.
* asyncio.run() creates a loop, runs one entry coroutine, then cancels leftovers and closes
  the loop. It cannot be called from inside a running loop (RuntimeError).
"""

import asyncio
from collections.abc import Sequence

EMBED_DELAY = 0.05
CALLS: list[str] = []


async def fake_embed(text: str, delay: float = EMBED_DELAY) -> list[float]:
    """Pretend embedding API: I/O-bound, so it awaits instead of blocking."""
    CALLS.append(text)
    await asyncio.sleep(delay)
    return [float(len(text))]


async def embed_sequential(texts: Sequence[str]) -> list[list[float]]:
    raise NotImplementedError


async def embed_concurrent(texts: Sequence[str], delays: Sequence[float] | None = None) -> list[list[float]]:
    raise NotImplementedError


def start_prefetch(texts: Sequence[str]) -> list[asyncio.Task[list[float]]]:
    raise NotImplementedError


def embed_blocking_entrypoint(texts: Sequence[str]) -> list[list[float]]:
    raise NotImplementedError
