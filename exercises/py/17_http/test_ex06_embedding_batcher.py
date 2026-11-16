import asyncio
import json

import httpx
import pytest

from study.loader import load

ex = load(__file__, "ex06_embedding_batcher")

BASE = "https://api.example.com/v1"


def vec(text):
    return [float(text[1:]), 1.0]  # "t17" -> [17.0, 1.0]: lets us check placement


class FakeEmbeddings:
    """Async MockTransport handler; records batches and peak concurrency."""

    def __init__(self, delay=0.01, fail=None):
        self.delay = delay
        self.fail = fail or {}  # text -> outcome for the batch containing it
        self.batches = []
        self.bodies = []
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        batch = body["input"]
        self.batches.append(batch)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            # later batches finish FIRST, so completion order != input order
            await asyncio.sleep(self.delay * (1 + 1 / (1 + int(batch[0][1:]))))
            for text, outcome in self.fail.items():
                if text in batch:
                    if isinstance(outcome, BaseException):
                        raise outcome
                    return outcome
            data = [{"index": i, "embedding": vec(t)} for i, t in enumerate(batch)]
            return httpx.Response(200, json={"data": data[::-1]})  # shuffled order
        finally:
            self.in_flight -= 1


def texts(n):
    return [f"t{i}" for i in range(n)]


async def run(handler, items, **kwargs):
    async with httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(handler)) as client:
        return await ex.embed_all(client, items, **kwargs)


@pytest.mark.topic("py.http.batching", "py.http.get_post")
async def test_batches_and_reassembly_in_input_order():
    fake = FakeEmbeddings()
    result = await run(fake, texts(10), batch_size=4, model="m-small")
    assert isinstance(result, ex.EmbedResult)
    assert result.failures == []
    assert result.vectors == [vec(t) for t in texts(10)]
    assert sorted(len(b) for b in fake.batches) == [2, 4, 4]
    assert sorted(fake.batches) == sorted([texts(10)[0:4], texts(10)[4:8], texts(10)[8:10]])
    assert all(b["model"] == "m-small" for b in fake.bodies)


@pytest.mark.topic("py.http.batching")
async def test_single_batch_and_exact_multiple():
    fake = FakeEmbeddings(delay=0)
    result = await run(fake, texts(3), batch_size=64)
    assert len(fake.batches) == 1 and result.vectors == [vec(t) for t in texts(3)]
    fake = FakeEmbeddings(delay=0)
    await run(fake, texts(8), batch_size=4)
    assert sorted(len(b) for b in fake.batches) == [4, 4]


@pytest.mark.topic("py.http.concurrency_limits", "py.http.concurrent_calls")
@pytest.mark.parametrize("limit", [1, 3])
async def test_concurrency_is_limited(limit):
    fake = FakeEmbeddings(delay=0.02)
    result = await run(fake, texts(20), batch_size=2, max_concurrency=limit)
    assert len(fake.batches) == 10
    assert fake.peak == limit, f"expected exactly {limit} requests in flight at peak, saw {fake.peak}"
    assert result.vectors == [vec(t) for t in texts(20)]


@pytest.mark.topic("py.http.concurrent_calls")
async def test_batches_run_concurrently():
    fake = FakeEmbeddings(delay=0.05)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await run(fake, texts(16), batch_size=2, max_concurrency=8)
    assert loop.time() - start < 0.3, "8 batches of ~0.06s should overlap"


@pytest.mark.topic("py.http.partial_failures", "py.http.response_validation", "py.http.status_codes")
async def test_partial_failures_reported_per_batch():
    fake = FakeEmbeddings(
        fail={
            "t5": httpx.Response(500, text="upstream exploded"),
            "t9": httpx.ConnectError("connection reset"),
            "t13": httpx.Response(200, text="<html>gateway</html>"),
            "t17": httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0]}]}),
            "t21": httpx.Response(200, json={"data": [{"index": i, "embedding": "x"} for i in range(4)]}),
        }
    )
    result = await run(fake, texts(26), batch_size=4)
    assert result.failures == [
        ex.BatchFailure(1, [4, 5, 6, 7], "HTTP 500"),
        ex.BatchFailure(2, [8, 9, 10, 11], "ConnectError"),
        ex.BatchFailure(3, [12, 13, 14, 15], "invalid response"),
        ex.BatchFailure(4, [16, 17, 18, 19], "invalid response"),
        ex.BatchFailure(5, [20, 21, 22, 23], "invalid response"),
    ]
    assert len(result.vectors) == 26
    failed = {i for f in result.failures for i in f.input_indices}
    for i, v in enumerate(result.vectors):
        assert v == (None if i in failed else vec(f"t{i}"))


@pytest.mark.topic("py.http.response_validation", "py.http.partial_failures")
async def test_duplicate_or_out_of_range_indices_are_invalid():
    dup = httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}] * 2})
    oob = httpx.Response(
        200, json={"data": [{"index": 0, "embedding": [1.0]}, {"index": 5, "embedding": [1.0]}]}
    )
    fake = FakeEmbeddings(delay=0, fail={"t0": dup, "t2": oob})
    result = await run(fake, texts(4), batch_size=2)
    assert [f.error for f in result.failures] == ["invalid response", "invalid response"]
    assert result.vectors == [None] * 4


@pytest.mark.topic("py.http.partial_failures")
async def test_timeouts_are_failures_too():
    fake = FakeEmbeddings(delay=0, fail={"t0": httpx.ReadTimeout("slow")})
    result = await run(fake, texts(3), batch_size=2)
    assert result.failures == [ex.BatchFailure(0, [0, 1], "ReadTimeout")]
    assert result.vectors[2] == vec("t2")


@pytest.mark.topic("py.http.batching")
async def test_empty_and_invalid_arguments():
    fake = FakeEmbeddings()
    result = await run(fake, [])
    assert (result.vectors, result.failures, fake.batches) == ([], [], [])
    for kwargs in ({"batch_size": 0}, {"max_concurrency": 0}):
        with pytest.raises(ValueError):
            await run(fake, texts(3), **kwargs)
    assert fake.batches == [], "validate before sending anything"
