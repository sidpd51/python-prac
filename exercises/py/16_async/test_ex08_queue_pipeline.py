import asyncio
import contextlib
import functools
import time

import pytest

from study.loader import load

ex = load(__file__, "ex08_queue_pipeline")


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


class RateLimited(Exception):
    pass


class Embedder:
    def __init__(self, delay: float = 0.03, fail: frozenset[str] = frozenset()) -> None:
        self.delay = delay
        self.fail = fail
        self.in_flight = 0
        self.max_in_flight = 0
        self.completed = 0
        self.started: list[str] = []
        self.lead_seen: list[int] = []  # produced - completed, sampled at every call
        self.produced = 0

    async def __call__(self, text: str) -> list[float]:
        self.started.append(text)
        self.lead_seen.append(self.produced - self.completed)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if text in self.fail:
                raise RateLimited(text)
            return [float(len(text))]
        finally:
            self.in_flight -= 1
            self.completed += 1


def source(emb: Embedder, n: int, *, explode_at: int | None = None):
    async def gen():
        for i in range(n):
            if i == explode_at:
                raise ValueError("db connection lost")
            emb.produced += 1
            yield f"doc{i}", "x" * (i + 1)

    return gen()


def other_tasks() -> set[asyncio.Task]:
    return asyncio.all_tasks() - {asyncio.current_task()}


@pytest.mark.topic("py.async.queue", "py.async.sequential_vs_concurrent")
@deadline()
async def test_ingest_processes_everything_with_n_workers():
    emb = Embedder()
    start = time.perf_counter()
    report = await ex.ingest(source(emb, 9), emb, workers=3, maxsize=5)
    elapsed = time.perf_counter() - start
    assert report.vectors == {f"doc{i}": [float(i + 1)] for i in range(9)}
    assert report.failed == {}
    assert emb.max_in_flight == 3
    assert elapsed < 0.2  # 3 waves of 0.03 s, not 9
    assert other_tasks() == set()  # workers were shut down


@pytest.mark.topic("py.async.queue")
@deadline()
async def test_ingest_bounded_queue_gives_backpressure():
    emb = Embedder(delay=0.005)
    await ex.ingest(source(emb, 20), emb, workers=2, maxsize=1)
    # in-flight (2) + queued (1) + the one the producer is blocked on (1)
    assert max(emb.lead_seen) <= 4
    assert emb.completed == 20


@pytest.mark.topic("py.async.queue", "py.async.gather_exceptions")
@deadline()
async def test_ingest_item_failure_does_not_kill_worker():
    emb = Embedder(delay=0.005, fail=frozenset({"xx", "xxxx"}))
    report = await ex.ingest(source(emb, 5), emb, workers=1, maxsize=2)
    assert sorted(report.vectors) == ["doc0", "doc2", "doc4"]
    assert sorted(report.failed) == ["doc1", "doc3"]
    assert isinstance(report.failed["doc1"], RateLimited)


@pytest.mark.topic("py.async.queue", "py.async.graceful_shutdown")
@deadline()
async def test_ingest_source_error_propagates_and_cleans_up():
    emb = Embedder(delay=0.01)
    with pytest.raises(ValueError, match="db connection lost"):
        await ex.ingest(source(emb, 10, explode_at=4), emb, workers=2)
    await asyncio.sleep(0)
    assert other_tasks() == set()


@pytest.mark.topic("py.async.queue")
@deadline()
async def test_ingest_validation_and_empty():
    emb = Embedder()
    report = await ex.ingest(source(emb, 0), emb)
    assert report.vectors == {} and report.failed == {}
    with pytest.raises(ValueError):
        await ex.ingest(source(emb, 1), emb, workers=0)
    with pytest.raises(ValueError):
        await ex.ingest(source(emb, 1), emb, maxsize=0)


@pytest.mark.topic("py.async.graceful_shutdown", "py.async.queue")
@deadline()
async def test_service_drain_processes_backlog():
    emb = Embedder(delay=0.02)
    svc = ex.IngestService(emb, workers=2)
    with raises_runtime_error():
        await svc.submit("early", "x")  # not started
    await svc.start()
    for i in range(5):
        await svc.submit(f"d{i}", "abc")
    await svc.stop(drain=True)
    assert sorted(svc.vectors) == [f"d{i}" for i in range(5)]
    assert svc.cancelled == [] and svc.dropped == []
    assert other_tasks() == set()
    with raises_runtime_error():
        await svc.submit("late", "x")


@pytest.mark.topic("py.async.graceful_shutdown", "py.async.cancellation", "py.async.cancellederror")
@deadline()
async def test_service_hard_stop_cancels_in_flight_and_drops_queue():
    emb = Embedder(delay=0.2)
    svc = ex.IngestService(emb, workers=2)
    await svc.start()
    for i in range(6):
        await svc.submit(f"d{i}", "abc")
    await asyncio.sleep(0.02)
    start = time.perf_counter()
    await svc.stop(drain=False)
    assert time.perf_counter() - start < 0.1
    assert sorted(svc.cancelled) == ["d0", "d1"]
    assert svc.dropped == ["d2", "d3", "d4", "d5"]
    assert svc.vectors == {}
    assert other_tasks() == set()


@pytest.mark.topic("py.async.graceful_shutdown", "py.async.timeouts", "py.async.best_practices")
@deadline()
async def test_service_drain_timeout_escalates_to_hard_stop():
    emb = Embedder(delay=0.2)
    svc = ex.IngestService(emb, workers=1)
    await svc.start()
    for i in range(3):
        await svc.submit(f"d{i}", "abc")
    start = time.perf_counter()
    await svc.stop(drain=True, timeout=0.05)
    assert time.perf_counter() - start < 0.15
    assert svc.cancelled == ["d0"]
    assert svc.dropped == ["d1", "d2"]
    assert other_tasks() == set()
