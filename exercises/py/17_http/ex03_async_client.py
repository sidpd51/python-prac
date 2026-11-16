"""
Exercise 17.3 - A reusable async API client                        level: ★★☆
topics: http.async_client, http.pooling, http.reuse, http.connect_timeouts,
        http.request_timeouts, http.response_validation, http.get_post, http.error_handling

Problem
-------
EmbeddingsClient wraps ONE httpx.AsyncClient for its whole lifetime:

    async with EmbeddingsClient("https://api.example.com/v1", api_key) as ec:
        resp = await ec.embed(["a", "b"])        # -> EmbeddingResponse (pydantic)

The response models and exceptions are given.

__init__(base_url, api_key, *, timeout=None, max_connections=10, transport=None)
  * only stores configuration - no client yet (constructors shouldn't do I/O)
  * self.timeout = timeout or DEFAULT_TIMEOUT   (30 s overall, 5 s to connect)
  * self.limits  = httpx.Limits(max_connections=max_connections,
                                max_keepalive_connections=max_connections)

async __aenter__ -> self
  * creates the httpx.AsyncClient (base_url, "Authorization: Bearer <key>" header,
    timeout, limits, transport). Entering twice without exiting -> RuntimeError.
async __aexit__
  * closes the client (aclose) - even if the body raised - and forgets it

client (property) -> httpx.AsyncClient
  * the open client; RuntimeError("client is not open") outside `async with`

async embed(texts, model="text-embedding-3-small", *, timeout=None) -> EmbeddingResponse
  * POST /embeddings {"model": model, "input": texts}
  * `timeout` (float seconds) overrides the client timeout for THIS request only
  * httpx.ConnectTimeout -> APITimeoutError(phase="connect");
    any other httpx.TimeoutException -> APITimeoutError(phase="read")  (chain the cause)
  * status >= 400 -> APIStatusError(status_code, body text)
  * body not valid JSON, or not matching EmbeddingResponse -> ResponseValidationError
    (chain the cause; keep the raw text in .raw)
  * len(data) != len(texts), or indices not exactly 0..n-1 -> ResponseValidationError
  * the returned response's data is sorted by index

async health() -> bool
  * GET /health; True iff status 200. Transport errors/timeouts -> False (never raise).

Interviewers look for
---------------------
* One AsyncClient per process/component: it owns a connection POOL; creating a client per
  request throws away keep-alive connections (new TCP + TLS handshake every call) and leaks
  sockets if not closed.
* httpx.Limits bounds pool size; pool timeout is how long you wait for a free connection.
* Timeout phases: connect (can we reach the host?), read (time between bytes), write, pool.
  A ConnectTimeout means the request was never sent - a ReadTimeout means it might have been
  processed (matters for retrying POSTs).
* Validate responses at the boundary: a 200 with a garbage body is still a failure.
"""

from typing import Self

import httpx
from pydantic import BaseModel

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingItem]
    usage: Usage


class APIStatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


class APITimeoutError(Exception):
    def __init__(self, phase: str) -> None:
        super().__init__(f"{phase} timeout")
        self.phase = phase


class ResponseValidationError(Exception):
    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class EmbeddingsClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: httpx.Timeout | None = None,
        max_connections: int = 10,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        raise NotImplementedError

    async def __aenter__(self) -> Self:
        raise NotImplementedError

    async def __aexit__(self, *exc_info: object) -> None:
        raise NotImplementedError

    @property
    def client(self) -> httpx.AsyncClient:
        raise NotImplementedError

    async def embed(
        self, texts: list[str], model: str = "text-embedding-3-small", *, timeout: float | None = None
    ) -> EmbeddingResponse:
        raise NotImplementedError

    async def health(self) -> bool:
        raise NotImplementedError
