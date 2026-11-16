"""
Exercise 17.1 - httpx basics: Client, AsyncClient, GET/POST        level: ★☆☆
topics: http.fundamentals, http.client, http.async_client, http.get_post, http.sync_vs_async,
        http.requests_vs_httpx, http.concurrent_calls

Problem
-------
A small client for an OpenAI-style API at API_BASE. No real network: every function takes
a client (or a transport) so tests can inject httpx.MockTransport.

1. make_client(api_key, *, transport=None) -> httpx.Client
     * base_url=API_BASE, headers: "Authorization: Bearer <key>", "User-Agent: USER_AGENT"
     * timeout = httpx.Timeout(10.0, connect=2.0)
     * pass `transport` through (None means the real network transport)

2. list_models(client) -> list[str]
     * GET /models  -> {"data": [{"id": "gpt-4o"}, ...]}  -> the ids, sorted
     * any non-2xx status -> httpx.HTTPStatusError (hint: response.raise_for_status())

3. search(client, query, top_k=5) -> list[dict]
     * GET /search with query parameters q=<query>, k=<top_k> - let httpx encode them
       (never build "?q=" + query by hand: "refund & returns" must arrive intact)
     * returns response.json()["results"]; non-2xx -> httpx.HTTPStatusError

4. create_embeddings(client, texts, model="text-embedding-3-small") -> list[list[float]]
     * POST /embeddings with JSON body {"model": model, "input": texts}
     * response {"data": [{"index": i, "embedding": [...]}, ...]} - return the vectors in
       INPUT order (sort by "index"; servers don't promise list order)
     * non-2xx -> httpx.HTTPStatusError

5. async fetch_documents(client: httpx.AsyncClient, doc_ids) -> list[dict | None]
     * GET /documents/{id} for every id CONCURRENTLY (not one after another)
     * results in the same order as doc_ids; a 404 gives None in that slot;
       any other non-2xx raises httpx.HTTPStatusError

Interviewers look for
---------------------
* A Client is a connection pool + shared config (base_url, headers, timeouts): create it
  once, reuse it; `httpx.get(...)` per call opens a new connection each time.
* requests is sync-only; httpx has the same API for sync (Client) and async (AsyncClient),
  plus HTTP/2, strict timeouts by default (5 s) and a pluggable transport (MockTransport).
* Sync I/O blocks the thread; with AsyncClient + gather, N requests take ~max(latency)
  instead of sum(latency).
* GET = safe/idempotent query (params in the URL); POST = create/compute (JSON body).
"""

from typing import Any

import httpx

API_BASE = "https://api.example.com/v1"
USER_AGENT = "rag-service/1.0"


def make_client(api_key: str, *, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    raise NotImplementedError


def list_models(client: httpx.Client) -> list[str]:
    raise NotImplementedError


def search(client: httpx.Client, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    raise NotImplementedError


def create_embeddings(
    client: httpx.Client, texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    raise NotImplementedError


async def fetch_documents(client: httpx.AsyncClient, doc_ids: list[str]) -> list[dict[str, Any] | None]:
    raise NotImplementedError
