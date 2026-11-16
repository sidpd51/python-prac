import asyncio
import json

import httpx
import numpy as np
import pytest
from docmind.embeddings.hashing import HashingEmbedder
from docmind.embeddings.http import EmbeddingError, HttpEmbedder
from docmind.interfaces import Embedder

# ---------------------------------------------------------------- M0.4a hashing embedder


@pytest.mark.topic("py.numpy.vectorization", "py.genai.tokenization", "py.type_hints.protocol")
async def test_hashing_shape_norm_determinism():
    e = HashingEmbedder(64)
    assert isinstance(e, Embedder)
    v = await e.embed(["Hello world", "hello, WORLD!", "", "something else entirely"])
    assert v.shape == (4, 64) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v[[0, 1, 3]], axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(v[0], v[1])  # case/punctuation-insensitive
    assert not v[2].any()  # no tokens -> zero vector, no NaN
    assert (await e.embed([])).shape == (0, 64)


@pytest.mark.topic("py.numpy.matmul")
async def test_hashing_similarity_is_meaningful():
    e = HashingEmbedder(512)
    q, near, far = await e.embed(
        ["vector database index", "a vector index in a database", "banana bread recipe"]
    )
    assert q @ near > q @ far


@pytest.mark.topic("py.numpy.creation")
def test_hashing_validation():
    with pytest.raises(ValueError):
        HashingEmbedder(0)


# ---------------------------------------------------------------- M0.4b HTTP embedder

DIM = 3


class FakeEmbeddingAPI:
    """Scriptable OpenAI-compatible embedding endpoint for httpx.MockTransport."""

    def __init__(self, script=None, shuffle=False, delay=0.0):
        self.script = list(script or [])  # each item: status int, or an Exception to raise
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.shuffle = shuffle
        self.delay = delay
        self.headers = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.headers.append(dict(request.headers))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.script:
                step = self.script.pop(0)
                if isinstance(step, Exception):
                    raise step
                if isinstance(step, tuple):
                    status, headers = step
                    return httpx.Response(status, headers=headers, json={"error": "x"})
                if step != 200:
                    return httpx.Response(step, json={"error": "x"})
            texts = json.loads(request.content)["input"]
            data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(texts)]
            if self.shuffle:
                data.reverse()
            return httpx.Response(200, json={"data": data})
        finally:
            self.in_flight -= 1


def make(api: FakeEmbeddingAPI, **kw):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    client = httpx.AsyncClient(transport=httpx.MockTransport(api.handler))
    emb = HttpEmbedder(client, "http://fake/embed", "m", DIM, sleep=fake_sleep, **kw)
    return emb, sleeps


@pytest.mark.topic("py.http.batching", "py.http.async_client", "py.genai.concurrent_embeddings")
async def test_http_batches_and_order():
    api = FakeEmbeddingAPI(shuffle=True)
    emb, _ = make(api, batch_size=2, api_key="sk-1")
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    out = await emb.embed(texts)
    assert out.shape == (5, DIM) and out.dtype == np.float32
    assert api.calls == 3
    raw_first = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    np.testing.assert_allclose(out[0], raw_first, rtol=1e-5)
    assert np.argmax(out[:, 0]) == 4, "rows must come back in input order"
    assert api.headers[0]["authorization"] == "Bearer sk-1"
    assert (await emb.embed([])).shape == (0, DIM) and api.calls == 3


@pytest.mark.topic("py.http.concurrency_limits", "py.async.semaphore")
async def test_http_concurrency_limit():
    api = FakeEmbeddingAPI(delay=0.02)
    emb, _ = make(api, batch_size=1, concurrency=2)
    await emb.embed([str(i) for i in range(8)])
    assert api.calls == 8 and api.max_in_flight == 2


@pytest.mark.topic("py.http.retries", "py.http.backoff", "py.genai.retry_backoff")
async def test_http_retries_then_succeeds():
    api = FakeEmbeddingAPI(script=[503, httpx.ConnectTimeout("t"), 200])
    emb, sleeps = make(api, base_delay=1.0, max_delay=10.0)
    out = await emb.embed(["x"])
    assert out.shape == (1, DIM) and api.calls == 3
    assert len(sleeps) == 2
    assert 0 <= sleeps[0] <= 1.0 and 0 <= sleeps[1] <= 2.0, "full-jitter exponential backoff"


@pytest.mark.topic("py.http.rate_limiting", "py.genai.rate_limits")
async def test_http_honours_retry_after():
    api = FakeEmbeddingAPI(script=[(429, {"Retry-After": "7"}), 200])
    emb, sleeps = make(api)
    await emb.embed(["x"])
    assert sleeps == [7.0]


@pytest.mark.topic("py.http.retry_limits", "py.exceptions.chaining")
async def test_http_gives_up():
    api = FakeEmbeddingAPI(script=[500] * 10)
    emb, sleeps = make(api, max_retries=2)
    with pytest.raises(EmbeddingError) as info:
        await emb.embed(["x"])
    assert api.calls == 3 and len(sleeps) == 2
    assert info.value.__cause__ is not None


@pytest.mark.topic("py.http.error_handling", "py.http.status_codes")
@pytest.mark.parametrize("status", [400, 401, 422])
async def test_http_no_retry_on_client_errors(status):
    api = FakeEmbeddingAPI(script=[status])
    emb, sleeps = make(api)
    with pytest.raises(EmbeddingError) as info:
        await emb.embed(["x"])
    assert api.calls == 1 and sleeps == [] and info.value.status_code == status


@pytest.mark.topic("py.http.response_validation", "py.pydantic.model_validate")
async def test_http_validates_response():
    async def bad(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": "nope"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(bad))
    with pytest.raises(EmbeddingError):
        await HttpEmbedder(client, "http://f", "m", DIM).embed(["x"])

    async def wrong_dim(request):
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrong_dim))
    with pytest.raises(EmbeddingError):
        await HttpEmbedder(client, "http://f", "m", DIM).embed(["x"])
