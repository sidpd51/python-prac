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
import random
from collections.abc import Awaitable, Callable, Sequence

import httpx
import numpy as np
from pydantic import BaseModel, SecretStr, ValidationError

log = logging.getLogger("docmind.embeddings")

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
        self.client = client
        self.url = url
        self.model = model
        self.dim = dim
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._sem = asyncio.Semaphore(concurrency)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
        mat = np.concatenate(results).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return np.divide(mat, norms, out=np.zeros_like(mat), where=norms > 0)

    async def _embed_batch(self, batch: Sequence[str]) -> np.ndarray:
        async with self._sem:
            body = await self._post_with_retries({"model": self.model, "input": list(batch)})
        try:
            parsed = EmbeddingResponse.model_validate(body)
        except ValidationError as e:
            raise EmbeddingError(f"malformed embedding response: {e.error_count()} errors") from e
        items = sorted(parsed.data, key=lambda d: d.index)
        if len(items) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(items)}")
        if any(len(it.embedding) != self.dim for it in items):
            raise EmbeddingError(f"expected dimension {self.dim}")
        return np.asarray([it.embedding for it in items], dtype=np.float32)

    async def _post_with_retries(self, payload: dict) -> object:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                resp = await self.client.post(self.url, json=payload, headers=self._headers)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in RETRYABLE_STATUS:
                    raise EmbeddingError(
                        f"embedding request failed: HTTP {resp.status_code}", resp.status_code
                    )
                last_exc = EmbeddingError(f"HTTP {resp.status_code}", resp.status_code)
                header = resp.headers.get("retry-after")
                if header is not None:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        retry_after = None
            if attempt == self.max_retries:
                break
            delay = (
                retry_after
                if retry_after is not None
                else random.uniform(0, min(self.max_delay, self.base_delay * 2**attempt))
            )
            log.warning("embedding attempt %d failed (%s); retrying in %.2fs", attempt + 1, last_exc, delay)
            await self._sleep(delay)
        status = last_exc.status_code if isinstance(last_exc, EmbeddingError) else None
        raise EmbeddingError(f"embedding failed after {self.max_retries + 1} attempts", status) from last_exc
