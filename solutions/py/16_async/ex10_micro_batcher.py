"""
Exercise 16.10 - Embedding micro-batcher (Condition + async with)  level: ★★★
topics: async.condition, async.async_context_managers, async.async_with, async.resource_management, async.graceful_shutdown

Problem
-------
Embedding APIs are far cheaper per text when called with a batch, but requests arrive one
at a time from many concurrent handlers. Build the classic server-side micro-batcher.

`class MicroBatcher(embed_batch, *, max_batch, max_wait)`
* `embed_batch(texts: list[str]) -> list[vector]` is async, one vector per text, in order.
* `async embed(text) -> vector` - called concurrently by many coroutines. Each call adds
  its text to a pending list and waits for ITS vector (an `asyncio.Future` per request is
  the usual trick).
* A single background flusher task (started by `__aenter__`) sends batches:
    - as soon as `max_batch` texts are pending -> flush immediately
    - otherwise flush whatever is pending `max_wait` seconds after the FIRST text of the
      batch arrived (a lone request must not wait forever)
    - never more than `max_batch` texts per call; record every batch size, in order, in
      the `batch_sizes` list attribute
  Use an `asyncio.Condition` to let `embed()` wake the flusher, and don't hold the
  condition's lock while awaiting `embed_batch` (new requests must keep queueing).
* If `embed_batch` raises, every caller in THAT batch gets the exception; the batcher keeps
  serving later batches.
* `async with MicroBatcher(...) as b:` - `__aenter__` starts the flusher and returns self;
  `__aexit__` calls `aclose()`.
* `async aclose()` - graceful: refuse new work, flush everything still pending (without
  waiting for max_wait), then stop the flusher task and await it. Idempotent.
  `embed()` after close -> RuntimeError.

Interviewers look for
---------------------
* Condition = Lock + "wait until predicate"; `await cond.wait_for(pred)` re-checks after
  every notify and re-acquires the lock before returning (also when cancelled/timed out).
* Size-or-time flushing is exactly how vLLM/TGI continuous batching and embedding
  services trade a few ms of latency for big throughput gains.
* Futures connect a producer-side result to an awaiting caller; guard against callers
  that were cancelled (`fut.done()`) before setting a result.
* Async context managers make start/stop symmetric and exception-safe.
"""

import asyncio
from collections.abc import Awaitable, Callable

Vector = list[float]


class MicroBatcher:
    def __init__(
        self,
        embed_batch: Callable[[list[str]], Awaitable[list[Vector]]],
        *,
        max_batch: int,
        max_wait: float,
    ) -> None:
        if max_batch < 1 or max_wait < 0:
            raise ValueError("max_batch must be >= 1 and max_wait >= 0")
        self._embed_batch = embed_batch
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.batch_sizes: list[int] = []
        self._pending: list[tuple[str, asyncio.Future[Vector]]] = []
        self._cond = asyncio.Condition()
        self._closing = False
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "MicroBatcher":
        self._task = asyncio.create_task(self._flusher(), name="micro-batcher")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def embed(self, text: str) -> Vector:
        if self._closing or self._task is None:
            raise RuntimeError("batcher is not running")
        fut: asyncio.Future[Vector] = asyncio.get_running_loop().create_future()
        async with self._cond:
            self._pending.append((text, fut))
            self._cond.notify()
        return await fut

    async def _next_batch(self) -> list[tuple[str, asyncio.Future[Vector]]] | None:
        loop = asyncio.get_running_loop()
        async with self._cond:
            await self._cond.wait_for(lambda: self._pending or self._closing)
            if not self._pending:
                return None  # closing and nothing left
            deadline = loop.time() + self.max_wait
            try:
                async with asyncio.timeout_at(deadline):
                    await self._cond.wait_for(lambda: len(self._pending) >= self.max_batch or self._closing)
            except TimeoutError:
                pass  # time's up: flush a partial batch
            batch = self._pending[: self.max_batch]
            del self._pending[: self.max_batch]
            return batch

    async def _flusher(self) -> None:
        while (batch := await self._next_batch()) is not None:
            self.batch_sizes.append(len(batch))
            try:
                vectors = await self._embed_batch([t for t, _ in batch])
            except Exception as e:  # noqa: BLE001 - deliver to the waiting callers
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            for (_, fut), vec in zip(batch, vectors, strict=True):
                if not fut.done():  # the caller may have been cancelled meanwhile
                    fut.set_result(vec)

    async def aclose(self) -> None:
        if self._task is None or self._task.done():
            self._closing = True
            return
        async with self._cond:
            self._closing = True
            self._cond.notify_all()
        await self._task  # flusher drains the backlog, then returns
