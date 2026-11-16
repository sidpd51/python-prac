"""
Milestone M0.4b - production-grade async HTTP embedder (satisfies interfaces.Embedder)
topics: py.http.async_client, py.http.retries, py.http.backoff, py.http.retry_limits,
        py.http.concurrency_limits, py.http.batching, py.http.response_validation,
        py.async.semaphore, py.async.gather, py.genai.concurrent_embeddings, py.genai.rate_limits

Talks to an OpenAI-compatible endpoint:
    POST {url}   json={"model": model, "input": [texts...]}
    200 ->       {"data": [{"index": 0, "embedding": [...]}, ...]}   (data may be OUT OF ORDER)

Build
-----
HttpEmbedder(client, url, model, dim, *, api_key=None, batch_size=64, concurrency=4,
             max_retries=4, base_delay=0.5, max_delay=20.0, sleep=asyncio.sleep)
  * `client` is an injected httpx.AsyncClient (connection reuse; tests pass a MockTransport)
  * api_key (str | SecretStr | None) -> header "Authorization: Bearer <key>"
  * embed(texts): split into batches of batch_size, send at most `concurrency` requests at
    once (asyncio.Semaphore), reassemble rows in INPUT order (sort by "index" within a
    batch), L2-normalise, return float32 array (n, dim). Empty input -> (0, dim) array, no calls.
  * validate the response body with pydantic (EmbeddingResponse). Wrong number of
    embeddings or wrong dimension -> EmbeddingError.
  * retries: 429, 500, 502, 503, 504 and httpx.TimeoutException/ConnectError are retryable;
    up to max_retries RETRIES (so max_retries + 1 attempts). Delay = full jitter
    random.uniform(0, min(max_delay, base_delay * 2**attempt)), but if the response has a
    numeric Retry-After header, sleep exactly that many seconds instead.
    Use the injected `sleep` (tests make it instant).
  * non-retryable 4xx (400, 401, 403, 404, 422...) -> EmbeddingError immediately, with the
    status code on `.status_code`. Exhausted retries -> EmbeddingError chained (`from`) to
    the last error.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

import httpx
import numpy as np
from pydantic import BaseModel, SecretStr

log = logging.getLogger("docmind.embeddings")
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        raise NotImplementedError


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingItem]


class HttpEmbedder:
    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        model: str,
        dim: int,
        *,
        api_key: str | SecretStr | None = None,
        batch_size: int = 64,
        concurrency: int = 4,
        max_retries: int = 4,
        base_delay: float = 0.5,
        max_delay: float = 20.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        raise NotImplementedError

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError
