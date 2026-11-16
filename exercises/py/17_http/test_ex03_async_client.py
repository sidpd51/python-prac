import json

import httpx
import pytest

from study.loader import load

ex = load(__file__, "ex03_async_client")

BASE = "https://api.example.com/v1"


def ok_body(n, model="embed-small"):
    return {
        "model": model,
        "data": [{"index": i, "embedding": [float(i), 0.5]} for i in reversed(range(n))],
        "usage": {"prompt_tokens": 3 * n, "total_tokens": 3 * n},
    }


def mock(handler):
    return httpx.MockTransport(handler)


def embeddings_handler(seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json=ok_body(n))

    return handler


@pytest.mark.topic("py.http.async_client", "py.http.connect_timeouts", "py.http.pooling")
def test_init_stores_config_only():
    ec = ex.EmbeddingsClient(BASE, "sk-1")
    assert ec.timeout == httpx.Timeout(30.0, connect=5.0)
    assert ec.limits == httpx.Limits(max_connections=10, max_keepalive_connections=10)
    with pytest.raises(RuntimeError):
        _ = ec.client  # not opened yet
    custom = ex.EmbeddingsClient(BASE, "k", timeout=httpx.Timeout(3.0), max_connections=2)
    assert custom.timeout == httpx.Timeout(3.0)
    assert custom.limits.max_connections == 2


@pytest.mark.topic("py.http.get_post", "py.http.response_validation", "py.http.async_client")
async def test_embed_happy_path():
    seen = []
    async with ex.EmbeddingsClient(BASE, "sk-1", transport=mock(embeddings_handler(seen))) as ec:
        resp = await ec.embed(["a", "b", "c"], model="embed-small")
    assert isinstance(resp, ex.EmbeddingResponse)
    assert [item.index for item in resp.data] == [0, 1, 2]
    assert resp.data[2].embedding == [2.0, 0.5]
    assert resp.usage.total_tokens == 9
    req = seen[0]
    assert (req.method, str(req.url)) == ("POST", f"{BASE}/embeddings")
    assert req.headers["authorization"] == "Bearer sk-1"
    assert json.loads(req.content) == {"model": "embed-small", "input": ["a", "b", "c"]}


@pytest.mark.topic("py.http.reuse", "py.http.pooling")
async def test_single_client_reused_and_closed(monkeypatch):
    created = []

    class CountingClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            created.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", CountingClient)
    ec = ex.EmbeddingsClient(BASE, "k", transport=mock(embeddings_handler()))
    async with ec:
        first = ec.client
        for _ in range(3):
            await ec.embed(["x"])
        assert await ec.health() is True
        assert ec.client is first
    assert len(created) == 1, "one AsyncClient (= one connection pool) for the whole lifetime"
    assert created[0].is_closed
    with pytest.raises(RuntimeError):
        _ = ec.client


@pytest.mark.topic("py.http.reuse", "py.http.async_client")
async def test_closed_even_if_body_raises_and_no_double_enter():
    ec = ex.EmbeddingsClient(BASE, "k", transport=mock(embeddings_handler()))
    with pytest.raises(KeyError):
        async with ec:
            inner = ec.client
            raise KeyError("boom")
    assert inner.is_closed
    async with ec:
        with pytest.raises(RuntimeError):
            await ec.__aenter__()


@pytest.mark.topic("py.http.request_timeouts", "py.http.connect_timeouts")
async def test_per_request_timeout_override():
    seen = []
    async with ex.EmbeddingsClient(BASE, "k", transport=mock(embeddings_handler(seen))) as ec:
        await ec.embed(["a"])
        await ec.embed(["a"], timeout=120.0)
    assert seen[0].extensions["timeout"] == {"connect": 5.0, "read": 30.0, "write": 30.0, "pool": 30.0}
    assert seen[1].extensions["timeout"] == {"connect": 120.0, "read": 120.0, "write": 120.0, "pool": 120.0}


@pytest.mark.topic("py.http.connect_timeouts", "py.http.request_timeouts", "py.http.error_handling")
@pytest.mark.parametrize(
    ("raised", "phase"),
    [
        (httpx.ConnectTimeout("connect"), "connect"),
        (httpx.ReadTimeout("read"), "read"),
        (httpx.PoolTimeout("pool"), "read"),
    ],
)
async def test_timeout_phases(raised, phase):
    async with ex.EmbeddingsClient(BASE, "k", transport=mock(raiser(raised))) as ec:
        with pytest.raises(ex.APITimeoutError) as info:
            await ec.embed(["a"])
    assert info.value.phase == phase
    assert info.value.__cause__ is raised


@pytest.mark.topic("py.http.status_codes", "py.http.error_handling")
async def test_status_error():
    async with ex.EmbeddingsClient(
        BASE, "k", transport=mock(lambda r: httpx.Response(401, text="invalid api key"))
    ) as ec:
        with pytest.raises(ex.APIStatusError) as info:
            await ec.embed(["a"])
    assert info.value.status_code == 401
    assert info.value.body == "invalid api key"


@pytest.mark.topic("py.http.response_validation")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>Service Unavailable</html>"),
        httpx.Response(200, text='{"model": "m", "data": ['),
        httpx.Response(200, json={"model": "m", "data": [{"index": 0}], "usage": ok_body(1)["usage"]}),
        httpx.Response(200, json={**ok_body(1), "data": [{"index": 0, "embedding": ["NaN?", 1]}]}),
        httpx.Response(200, json=ok_body(1)),  # 1 vector for 2 inputs
        httpx.Response(200, json={**ok_body(2), "data": [{"index": 0, "embedding": [1.0]}] * 2}),  # dup
    ],
)
async def test_response_validation_errors(response):
    async with ex.EmbeddingsClient(BASE, "k", transport=mock(lambda r: response)) as ec:
        with pytest.raises(ex.ResponseValidationError) as info:
            await ec.embed(["a", "b"])
    assert info.value.raw == response.text


@pytest.mark.topic("py.http.response_validation")
async def test_invalid_json_cause_is_chained():
    async with ex.EmbeddingsClient(BASE, "k", transport=mock(lambda r: httpx.Response(200, text="{"))) as ec:
        with pytest.raises(ex.ResponseValidationError) as info:
            await ec.embed(["a"])
    assert info.value.__cause__ is not None


def raiser(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.topic("py.http.error_handling", "py.http.get_post")
@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (embeddings_handler(), True),
        (lambda r: httpx.Response(503), False),
        (raiser(httpx.ConnectError("refused")), False),
        (raiser(httpx.ReadTimeout("slow")), False),
    ],
)
async def test_health(handler, expected):
    async with ex.EmbeddingsClient(BASE, "k", transport=mock(handler)) as ec:
        assert await ec.health() is expected
