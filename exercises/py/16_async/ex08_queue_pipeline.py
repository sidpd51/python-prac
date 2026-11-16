"""
Exercise 16.8 - Ingestion pipeline: Queue, workers, graceful shutdown level: ★★★
topics: async.queue, async.graceful_shutdown, async.cancellation, async.cleanup_finally

Problem
-------
`embed(text)` arguments are async callables returning a vector; they may raise.

1. `async def ingest(source, embed, *, workers=3, maxsize=10) -> IngestReport`
   `source` is an ASYNC iterable of `(doc_id, text)` pairs (e.g. rows streamed from a DB).
   * one producer copies `source` into an `asyncio.Queue(maxsize)` - the bounded queue gives
     BACKPRESSURE: the producer must not run ahead of the workers by more than the queue size
   * `workers` worker tasks take items and call `embed`; success -> report.vectors[doc_id],
     an `Exception` -> report.failed[doc_id] = the exception (the worker keeps going)
   * when the source is exhausted and every item is processed, shut the workers down cleanly
     (sentinels, or `queue.join()` + cancel) and return. No task may outlive the call.
   * if the source itself raises, stop the workers and propagate that exception
   * workers < 1 or maxsize < 1 -> ValueError

2. `class IngestService(embed, *, workers=2)` - a long-running version with lifecycle:
   * `async start()` - spawn the worker tasks
   * `async submit(doc_id, text)` - enqueue (unbounded queue); RuntimeError if the service
     is not started or is stopping/stopped
   * `async stop(*, drain=True, timeout=None)`:
       - drain=True: stop accepting work, wait until everything already queued is processed,
         then stop the workers. If `timeout` is given and draining takes longer, fall back
         to a hard stop.
       - drain=False (hard stop): cancel the workers right away.
     After a hard stop, `cancelled` lists doc_ids whose embed call was interrupted mid-flight
     (record it in the worker's CancelledError handler, then RE-RAISE) and `dropped` lists
     queued-but-never-started doc_ids in queue order.
     When stop() returns, all worker tasks are finished.
   * attributes: `vectors: dict`, `failed: dict`, `cancelled: list`, `dropped: list`

Interviewers look for
---------------------
* Bounded queue = backpressure; an unbounded queue just moves the OOM somewhere else.
* `queue.task_done()` once per `get()` (in a finally!) or `join()` hangs forever.
* Shutdown options: sentinels (one per worker), `join()` then cancel, or `Queue.shutdown()`
  (3.13+). Graceful = stop intake -> drain -> stop workers -> await them; with a timeout
  that escalates to cancellation (what Kubernetes' SIGTERM -> SIGKILL grace period expects).
* A worker that swallows CancelledError can never be stopped.
"""

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass, field

Vector = list[float]
Embed = Callable[[str], Awaitable[Vector]]


@dataclass
class IngestReport:
    vectors: dict[str, Vector] = field(default_factory=dict)
    failed: dict[str, Exception] = field(default_factory=dict)


async def ingest(
    source: AsyncIterable[tuple[str, str]], embed: Embed, *, workers: int = 3, maxsize: int = 10
) -> IngestReport:
    raise NotImplementedError


class IngestService:
    def __init__(self, embed: Embed, *, workers: int = 2) -> None:
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def submit(self, doc_id: str, text: str) -> None:
        raise NotImplementedError

    async def stop(self, *, drain: bool = True, timeout: float | None = None) -> None:
        raise NotImplementedError
