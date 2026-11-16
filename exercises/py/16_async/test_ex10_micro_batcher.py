import asyncio
import contextlib
import functools
import time

import pytest

from study.loader import load

ex = load(__file__, "ex10_micro_batcher")


@contextlib.contextmanager
def raises_runtime_error():
    """pytest.raises(RuntimeError) would also swallow a stub's NotImplementedError (a subclass)."""
    try:
        yield
    except NotImplementedError:
        raise
    except RuntimeError:
        return
    pytest.fail("expected RuntimeError")


def deadline(seconds: float = 3.0):
    """Turn a hang in a buggy solution (lost task_done, unresolved future...) into a failure."""

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            async with asyncio.timeout(seconds):
                return await fn(*args, **kwargs)

        return wrapper

    return deco


class BatchAPI:
    def __init__(self, delay: float = 0.01, fail_on: str | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.batches: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        await asyncio.sleep(self.delay)
        if self.fail_on is not None and self.fail_on in texts:
            raise ConnectionError("embedding service 503")
        return [[float(len(t))] for t in texts]


def other_tasks() -> set[asyncio.Task]:
    return asyncio.all_tasks() - {asyncio.current_task()}


@pytest.mark.topic("py.async.condition", "py.async.async_with")
@deadline()
async def test_concurrent_requests_are_batched():
    api = BatchAPI()
    texts = [f"t{'x' * i}" for i in range(10)]
    async with ex.MicroBatcher(api, max_batch=4, max_wait=0.03) as b:
        vectors = await asyncio.gather(*(b.embed(t) for t in texts))
    assert vectors == [[float(len(t))] for t in texts]  # each caller got ITS vector
    assert b.batch_sizes == [4, 4, 2]
    assert [t for batch in api.batches for t in batch] == texts  # arrival order kept
    assert all(len(batch) <= 4 for batch in api.batches)


@pytest.mark.topic("py.async.condition", "py.async.timeouts")
@deadline()
async def test_lone_request_flushes_after_max_wait():
    api = BatchAPI(delay=0.0)
    async with ex.MicroBatcher(api, max_batch=8, max_wait=0.05) as b:
        start = time.perf_counter()
        assert await b.embed("solo") == [4.0]
        elapsed = time.perf_counter() - start
    assert 0.04 <= elapsed < 0.2
    assert b.batch_sizes == [1]


@pytest.mark.topic("py.async.condition")
@deadline()
async def test_full_batch_does_not_wait_for_timer():
    api = BatchAPI(delay=0.0)
    async with ex.MicroBatcher(api, max_batch=3, max_wait=1.0) as b:
        start = time.perf_counter()
        await asyncio.gather(*(b.embed(t) for t in ["a", "b", "c"]))
        assert time.perf_counter() - start < 0.2  # flushed on size, not after 1 s
    assert b.batch_sizes == [3]


@pytest.mark.topic("py.async.condition", "py.async.event_loop")
@deadline()
async def test_requests_arriving_over_time_share_a_window():
    api = BatchAPI(delay=0.0)

    async def later(b, text: str, delay: float):
        await asyncio.sleep(delay)
        return await b.embed(text)

    async with ex.MicroBatcher(api, max_batch=10, max_wait=0.08) as b:
        await asyncio.gather(
            later(b, "a", 0.0), later(b, "b", 0.02), later(b, "c", 0.04), later(b, "d", 0.15)
        )
    # a, b, c arrive within the first window; d comes after it was flushed
    assert api.batches == [["a", "b", "c"], ["d"]]


@pytest.mark.topic("py.async.condition", "py.async.gather_exceptions")
@deadline()
async def test_batch_failure_goes_to_its_callers_only():
    api = BatchAPI(fail_on="poison")
    async with ex.MicroBatcher(api, max_batch=2, max_wait=0.02) as b:
        results = await asyncio.gather(
            b.embed("ok1"), b.embed("poison"), b.embed("ok2"), b.embed("ok3"), return_exceptions=True
        )
        assert await b.embed("after") == [5.0]  # still serving
    assert isinstance(results[0], ConnectionError) and isinstance(results[1], ConnectionError)
    assert results[2:] == [[3.0], [3.0]]


@pytest.mark.topic(
    "py.async.graceful_shutdown", "py.async.resource_management", "py.async.async_context_managers"
)
@deadline()
async def test_aclose_flushes_pending_and_stops_flusher():
    api = BatchAPI(delay=0.0)
    b = ex.MicroBatcher(api, max_batch=10, max_wait=5.0)
    assert await b.__aenter__() is b
    pending = [asyncio.create_task(b.embed(t)) for t in ["x", "yy", "zzz"]]
    await asyncio.sleep(0.01)
    start = time.perf_counter()
    await b.aclose()
    assert time.perf_counter() - start < 0.2  # did not wait for max_wait
    assert [p.result() for p in pending] == [[1.0], [2.0], [3.0]]
    assert other_tasks() == set()  # flusher task finished
    with raises_runtime_error():
        await b.embed("late")
    await b.aclose()  # idempotent


@pytest.mark.topic("py.async.async_context_managers", "py.async.async_with", "py.async.resource_management")
@deadline()
async def test_context_manager_closes_on_error():
    api = BatchAPI(delay=0.0)
    with pytest.raises(KeyError):
        async with ex.MicroBatcher(api, max_batch=4, max_wait=0.01) as b:
            assert await b.embed("abc") == [3.0]
            raise KeyError("handler bug")
    assert other_tasks() == set()
    with raises_runtime_error():
        await b.embed("x")
