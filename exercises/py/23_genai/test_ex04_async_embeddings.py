import asyncio

import pytest

from study.loader import load

ex = load(__file__, "ex04_async_embeddings")


def texts(n):
    return [f"t{i}" for i in range(n)]


def vec(text):
    return [float(text[1:]), float(len(text))]


class FakeEmbedder:
    """Fake embedding API: records calls, concurrency and cancellations.

    fail: {first_text_of_batch: [exception, exception, ...]} raised on successive calls.
    delay: seconds per call, or a function first_text -> seconds.
    """

    def __init__(self, *, fail=None, delay=0.0, wrong_length_for=None):
        self.fail = {k: list(v) for k, v in (fail or {}).items()}
        self.delay = delay
        self.wrong_length_for = wrong_length_for
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.peak = 0
        self.completed: list[str] = []
        self.cancelled: list[str] = []

    async def __call__(self, batch):
        key = batch[0]
        self.calls.append(list(batch))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay(key) if callable(self.delay) else self.delay)
            if self.fail.get(key):
                raise self.fail[key].pop(0)
            self.completed.append(key)
            if key == self.wrong_length_for:
                return [vec(t) for t in batch[:-1]]
            return [vec(t) for t in batch]
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise
        finally:
            self.in_flight -= 1


class RecordingSleep:
    def __init__(self, real=0.0):
        self.delays = []
        self.real = real

    async def __call__(self, seconds):
        self.delays.append(seconds)
        await asyncio.sleep(self.real)


@pytest.mark.topic("py.genai.batching", "py.genai.concurrent_embeddings")
async def test_batches_and_input_order():
    # later batches finish FIRST - results must still be in input order
    api = FakeEmbedder(delay=lambda key: 0.04 - 0.01 * (int(key[1:]) // 3))
    out = await ex.embed_texts(texts(10), api, batch_size=3, max_concurrency=4)
    assert out == [vec(t) for t in texts(10)]
    assert sorted(len(b) for b in api.calls) == [1, 3, 3, 3]
    assert sorted(api.calls) == sorted([texts(10)[i : i + 3] for i in range(0, 10, 3)])
    assert api.completed[0] != "t0", "test setup: batches should complete out of order"


@pytest.mark.topic("py.genai.concurrent_embeddings", "py.genai.async_llm_calls")
async def test_bounded_concurrency_is_reached_not_exceeded():
    api = FakeEmbedder(delay=0.02)
    loop = asyncio.get_running_loop()
    start = loop.time()
    out = await ex.embed_texts(texts(40), api, batch_size=2, max_concurrency=4)
    elapsed = loop.time() - start
    assert len(out) == 40
    assert api.peak == 4, f"peak in-flight requests was {api.peak}, expected exactly 4"
    assert elapsed < 0.3, "20 batches x 20ms with 4 in flight should take ~0.1s, not 0.4s"


@pytest.mark.topic("py.genai.retry_backoff", "py.genai.rate_limits")
async def test_exponential_backoff_then_success():
    api = FakeEmbedder(fail={"t0": [ex.RateLimitError(), ex.RateLimitError(), ex.RateLimitError()]})
    sleep = RecordingSleep()
    out = await ex.embed_texts(texts(2), api, batch_size=2, max_retries=3, base_delay=0.05, sleep=sleep)
    assert out == [vec("t0"), vec("t1")]
    assert sleep.delays == pytest.approx([0.05, 0.1, 0.2])
    assert len(api.calls) == 4


@pytest.mark.topic("py.genai.rate_limits")
async def test_retry_after_header_wins():
    api = FakeEmbedder(fail={"t0": [ex.RateLimitError(retry_after=1.5), ex.RateLimitError()]})
    sleep = RecordingSleep()
    await ex.embed_texts(texts(1), api, base_delay=0.01, sleep=sleep)
    assert sleep.delays == pytest.approx([1.5, 0.02])


@pytest.mark.topic("py.genai.retry_backoff")
async def test_gives_up_after_max_retries():
    api = FakeEmbedder(fail={"t0": [ex.RateLimitError(f"429 #{i}") for i in range(10)]})
    sleep = RecordingSleep()
    with pytest.raises(ex.RateLimitError, match="429 #2"):
        await ex.embed_texts(texts(1), api, max_retries=2, sleep=sleep)
    assert len(api.calls) == 3
    assert len(sleep.delays) == 2, "no sleep after the final failed attempt"


@pytest.mark.topic("py.genai.retry_backoff")
async def test_non_rate_limit_errors_are_not_retried():
    api = FakeEmbedder(fail={"t0": [PermissionError("401 bad key")]})
    sleep = RecordingSleep()
    with pytest.raises(PermissionError):
        await ex.embed_texts(texts(1), api, sleep=sleep)
    assert len(api.calls) == 1
    assert sleep.delays == []


@pytest.mark.topic("py.genai.concurrent_embeddings")
async def test_wrong_response_length_is_an_error():
    api = FakeEmbedder(wrong_length_for="t2")
    with pytest.raises(ValueError):
        await ex.embed_texts(texts(4), api, batch_size=2)


@pytest.mark.topic("py.genai.async_llm_calls", "py.genai.concurrent_embeddings")
async def test_failure_cancels_remaining_batches():
    slow = 0.3
    api = FakeEmbedder(
        fail={"t0": [PermissionError("401")]},
        delay=lambda key: 0.0 if key == "t0" else slow,
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(PermissionError):
        await ex.embed_texts(texts(6), api, batch_size=2, max_concurrency=3)
    assert loop.time() - start < slow, "should not wait for the slow batches to finish"
    assert sorted(api.cancelled) == ["t2", "t4"]
    assert api.completed == []


@pytest.mark.topic("py.genai.rate_limits", "py.genai.concurrent_embeddings")
async def test_backoff_sleep_does_not_hold_a_slot():
    # one slot only: while t0 backs off, t2's batch must be able to use the slot
    api = FakeEmbedder(fail={"t0": [ex.RateLimitError(retry_after=0.05)]})
    sleep = RecordingSleep(real=0.05)
    out = await ex.embed_texts(texts(4), api, batch_size=2, max_concurrency=1, sleep=sleep)
    assert out == [vec(t) for t in texts(4)]
    assert [b[0] for b in api.calls] == ["t0", "t2", "t0"]


@pytest.mark.topic("py.genai.batching")
async def test_empty_and_validation():
    api = FakeEmbedder()
    assert await ex.embed_texts([], api) == []
    assert api.calls == []
    with pytest.raises(ValueError):
        await ex.embed_texts(texts(3), api, batch_size=0)
    with pytest.raises(ValueError):
        await ex.embed_texts(texts(3), api, max_concurrency=0)
