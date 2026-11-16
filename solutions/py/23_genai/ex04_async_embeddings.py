"""
Exercise 23.4 - Concurrent batched embeddings with rate-limit retries   level: ★★★
topics: genai.concurrent_embeddings, genai.async_llm_calls, genai.rate_limits,
        genai.retry_backoff, genai.batching

Problem
-------
Embed a corpus through an API that takes up to `batch_size` texts per request, allows
only a few concurrent requests, and answers HTTP 429 when you go too fast.

await embed_texts(texts, embed_batch, *, batch_size=16, max_concurrency=4, max_retries=3,
                  base_delay=0.05, sleep=asyncio.sleep) -> list[list[float]]

  * embed_batch: an async callable  `await embed_batch(list_of_texts) -> list_of_vectors`
  * batch_size < 1 or max_concurrency < 1 -> ValueError;  empty texts -> [] (no calls)
  * split texts into consecutive batches of batch_size (last may be shorter)
  * run batches CONCURRENTLY, but never more than max_concurrency requests in flight
    (asyncio.Semaphore); with enough batches the limit should actually be reached
  * RateLimitError (given below) -> retry that batch, up to max_retries times (so at most
    max_retries + 1 calls per batch). Before retry number k (k = 1, 2, ...):
        await sleep(err.retry_after)          if the server sent retry_after
        await sleep(base_delay * 2 ** (k-1))  otherwise   (exponential backoff)
    Do the sleep WITHOUT holding a semaphore slot (other batches may use it meanwhile).
    Out of retries -> that RateLimitError propagates.
  * any other exception propagates immediately (no retry)
  * when one batch fails for good, the other still-running batches are CANCELLED (don't keep
    paying for requests whose result will be thrown away), and the original exception
    propagates (not wrapped in an ExceptionGroup)
  * a response whose length != batch length -> ValueError (API contract broken)
  * return one vector per input text, in INPUT order, regardless of completion order

Interviewers look for
---------------------
* Semaphore = bounded concurrency; gather over unbounded tasks = self-inflicted 429s.
* Results keyed by batch index, not by completion order.
* Exponential backoff (+ jitter in production, omitted here for determinism); honour the
  server's Retry-After; never retry non-transient errors (400, auth).
* Sleeping while holding the semaphore wastes a slot; cancelling siblings on failure saves money.
* Injected sleep -> fast, deterministic tests.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence

__stub_keep__ = ["RateLimitError"]

Vector = list[float]
EmbedBatch = Callable[[list[str]], Awaitable[list[Vector]]]


class RateLimitError(Exception):
    """HTTP 429. `retry_after` (seconds) is set when the server sent a Retry-After header."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


async def embed_texts(
    texts: Sequence[str],
    embed_batch: EmbedBatch,
    *,
    batch_size: int = 16,
    max_concurrency: int = 4,
    max_retries: int = 3,
    base_delay: float = 0.05,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> list[Vector]:
    if batch_size < 1 or max_concurrency < 1:
        raise ValueError("batch_size and max_concurrency must be >= 1")
    batches = [list(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(batch: list[str]) -> list[Vector]:
        for retry in range(max_retries + 1):
            try:
                async with semaphore:
                    vectors = await embed_batch(batch)
            except RateLimitError as err:
                if retry == max_retries:
                    raise
                delay = err.retry_after if err.retry_after is not None else base_delay * 2**retry
                await sleep(delay)  # outside the semaphore: the slot is free meanwhile
                continue
            if len(vectors) != len(batch):
                raise ValueError(f"expected {len(batch)} vectors, got {len(vectors)}")
            return vectors
        raise AssertionError("unreachable")

    tasks = [asyncio.create_task(run(b)) for b in batches]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)  # let cancellations finish
        raise
    return [vec for batch_vectors in results for vec in batch_vectors]
