import asyncio
import json
import time

import httpx
import pytest

from study.loader import load

ex = load(__file__, "ex01_http_basics")

BASE = "https://api.example.com/v1"


def sync_client(handler):
    return httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler))


@pytest.mark.topic("py.http.client", "py.http.fundamentals")
def test_make_client_config_and_headers():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    with ex.make_client("sk-test", transport=httpx.MockTransport(handler)) as client:
        assert isinstance(client, httpx.Client)
        assert client.timeout == httpx.Timeout(10.0, connect=2.0)
        client.get("/models")
    req = seen[0]
    assert str(req.url) == "https://api.example.com/v1/models"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert req.headers["user-agent"] == ex.USER_AGENT


@pytest.mark.topic("py.http.client", "py.http.get_post")
def test_list_models_sorted():
    def handler(request):
        assert request.method == "GET" and request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "gpt-4o"}, {"id": "embed-3"}, {"id": "o3"}]})

    assert ex.list_models(sync_client(handler)) == ["embed-3", "gpt-4o", "o3"]


@pytest.mark.topic("py.http.status_codes", "py.http.fundamentals")
@pytest.mark.parametrize("status", [401, 404, 500])
def test_list_models_raises_on_error_status(status):
    client = sync_client(lambda r: httpx.Response(status, json={"error": "nope"}))
    with pytest.raises(httpx.HTTPStatusError) as info:
        ex.list_models(client)
    assert info.value.response.status_code == status


@pytest.mark.topic("py.http.get_post", "py.http.fundamentals")
def test_search_encodes_query_params():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"results": [{"id": "d1", "score": 0.9}]})

    out = ex.search(sync_client(handler), "refund & returns?", top_k=3)
    assert out == [{"id": "d1", "score": 0.9}]
    params = seen[0].url.params
    assert params["q"] == "refund & returns?"
    assert params["k"] == "3"
    assert seen[0].method == "GET"


@pytest.mark.topic("py.http.get_post")
def test_create_embeddings_posts_json_and_reorders():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]},
        )

    vecs = ex.create_embeddings(sync_client(handler), ["a", "b"], model="m-small")
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/embeddings"
    assert req.headers["content-type"] == "application/json"
    assert json.loads(req.content) == {"model": "m-small", "input": ["a", "b"]}


@pytest.mark.topic("py.http.status_codes")
def test_create_embeddings_error():
    client = sync_client(lambda r: httpx.Response(400, json={"error": "bad input"}))
    with pytest.raises(httpx.HTTPStatusError):
        ex.create_embeddings(client, ["x"])


def async_client(handler):
    return httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(handler))


@pytest.mark.topic("py.http.async_client", "py.http.concurrent_calls", "py.http.sync_vs_async")
async def test_fetch_documents_concurrent_and_ordered():
    delays = {"a": 0.15, "b": 0.05, "c": 0.10, "d": 0.01}

    async def handler(request):
        doc_id = request.url.path.rsplit("/", 1)[-1]
        await asyncio.sleep(delays[doc_id])
        return httpx.Response(200, json={"id": doc_id})

    async with async_client(handler) as client:
        start = time.perf_counter()
        docs = await ex.fetch_documents(client, ["a", "b", "c", "d"])
        elapsed = time.perf_counter() - start
    assert [d["id"] for d in docs] == ["a", "b", "c", "d"], "input order, not completion order"
    assert elapsed < 0.25, f"took {elapsed:.2f}s - requests ran sequentially (sum = 0.31s)"


@pytest.mark.topic("py.http.async_client", "py.http.status_codes")
async def test_fetch_documents_404_is_none_other_errors_raise():
    async def handler(request):
        doc_id = request.url.path.rsplit("/", 1)[-1]
        if doc_id == "gone":
            return httpx.Response(404)
        if doc_id == "boom":
            return httpx.Response(503)
        return httpx.Response(200, json={"id": doc_id})

    async with async_client(handler) as client:
        assert await ex.fetch_documents(client, ["x", "gone", "y"]) == [{"id": "x"}, None, {"id": "y"}]
        assert await ex.fetch_documents(client, []) == []
        with pytest.raises(httpx.HTTPStatusError):
            await ex.fetch_documents(client, ["x", "boom"])


@pytest.mark.topic("py.http.requests_vs_httpx", "py.http.reuse")
def test_one_client_many_requests():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "m"}], "results": []})

    client = sync_client(handler)
    ex.list_models(client)
    ex.search(client, "q")
    ex.list_models(client)
    assert calls == ["/v1/models", "/v1/search", "/v1/models"]
    assert not client.is_closed, "functions must not close a client they were given"
