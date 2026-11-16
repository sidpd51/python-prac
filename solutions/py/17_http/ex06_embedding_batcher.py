"""
Exercise 17.6 - Batched, concurrency-limited embedding with partial failures   level: ★★★
topics: http.batching, http.concurrency_limits, http.concurrent_calls, http.partial_failures,
        http.response_validation, http.get_post

Problem
-------
Embed 10,000 chunks for a RAG index: send them in batches, a few batches at a time, and
don't let one bad batch throw away 99% of the work. Models/dataclasses are given.

async embed_all(client, texts, *, model="text-embedding-3-small", batch_size=64,
                max_concurrency=4) -> EmbedResult
  * batch_size < 1 or max_concurrency < 1 -> ValueError (before any request)
  * no texts -> EmbedResult([], []) without any request
  * split `texts` into consecutive batches of `batch_size` (last one may be smaller);
    batch i covers input indices [i*batch_size, ...)
  * each batch: POST /embeddings  {"model": model, "input": [...batch texts...]}
  * all batches run concurrently, but never more than `max_concurrency` requests in flight
    (asyncio.Semaphore - acquire it around the request only)
  * the response is EmbeddingResponse JSON; `data[k].index` is the position WITHIN the batch
    and items may come back in any order -> put every vector at its global input position
  * a batch FAILS (its texts get None vectors, and one BatchFailure is recorded) when:
        - any httpx.HTTPError while sending               error = type(exc).__name__
        - status >= 400                                   error = f"HTTP {status}"
        - body isn't valid EmbeddingResponse JSON         error = "invalid response"
        - indices aren't exactly 0..len(batch)-1          error = "invalid response"
  * other batches are unaffected; EmbedResult.vectors has exactly len(texts) entries
    (vector or None) in input order; failures are sorted by batch_index
  * BatchFailure(batch_index, input_indices=[global indices], error=...)

Interviewers look for
---------------------
* Batching amortises per-request overhead and is how embedding APIs are priced/limited
  (max inputs per request, max tokens per request).
* gather/TaskGroup for concurrency, a Semaphore for the in-flight cap (a rate limiter would
  cap requests per SECOND - different thing).
* Reassembly by index, never by completion order or response list order.
* Partial failure as a first-class result: return successes + a failure report so the caller
  can retry just the failed batches. With a plain gather() the first exception propagates and
  the results of the batches that succeeded are lost (they keep running, unobserved) - handle
  errors per batch, or use return_exceptions=True / a TaskGroup with per-task try/except.
"""

import asyncio
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ValidationError


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingItem]


@dataclass(frozen=True)
class BatchFailure:
    batch_index: int
    input_indices: list[int]
    error: str


@dataclass
class EmbedResult:
    vectors: list[list[float] | None]
    failures: list[BatchFailure] = field(default_factory=list)


async def _embed_batch(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    model: str,
    batch: list[str],
) -> list[list[float]] | str:
    """Return vectors in batch order, or an error string."""
    try:
        async with sem:
            response = await client.post("/embeddings", json={"model": model, "input": batch})
    except httpx.HTTPError as exc:
        return type(exc).__name__
    if response.status_code >= 400:
        return f"HTTP {response.status_code}"
    try:
        parsed = EmbeddingResponse.model_validate_json(response.content)
    except ValidationError:
        return "invalid response"
    by_index = {item.index: item.embedding for item in parsed.data}
    if len(parsed.data) != len(batch) or set(by_index) != set(range(len(batch))):
        return "invalid response"
    return [by_index[i] for i in range(len(batch))]


async def embed_all(
    client: httpx.AsyncClient,
    texts: list[str],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
    max_concurrency: int = 4,
) -> EmbedResult:
    if batch_size < 1 or max_concurrency < 1:
        raise ValueError("batch_size and max_concurrency must be >= 1")
    if not texts:
        return EmbedResult([], [])
    sem = asyncio.Semaphore(max_concurrency)
    starts = range(0, len(texts), batch_size)
    outcomes = await asyncio.gather(
        *(_embed_batch(client, sem, model, texts[s : s + batch_size]) for s in starts)
    )
    result = EmbedResult(vectors=[None] * len(texts))
    for batch_index, (start, outcome) in enumerate(zip(starts, outcomes, strict=True)):
        indices = list(range(start, min(start + batch_size, len(texts))))
        if isinstance(outcome, str):
            result.failures.append(BatchFailure(batch_index, indices, outcome))
            continue
        for i, vector in zip(indices, outcome, strict=True):
            result.vectors[i] = vector
    return result
