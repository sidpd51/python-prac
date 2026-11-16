import threading
import time

import pytest

from study.loader import load

ex = load(__file__, "ex03_threadpool_io")


class ConcurrencyProbe:
    """Records how many calls overlap at the same time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def __enter__(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def __exit__(self, *exc):
        with self._lock:
            self.active -= 1


@pytest.mark.topic("py.internals.threadpool", "py.internals.cpu_vs_io")
def test_fetch_all_runs_concurrently_and_keeps_order():
    barrier = threading.Barrier(6, timeout=5)  # only passes if 6 calls overlap

    def fetch(url):
        barrier.wait()  # BrokenBarrierError if the calls ran one by one
        time.sleep(0.01 * (6 - int(url[-1])))  # later items finish first
        return url.upper()

    urls = [f"https://api/doc{i}" for i in range(6)]
    assert ex.fetch_all(urls, fetch, max_workers=6) == [u.upper() for u in urls]


@pytest.mark.topic("py.internals.threadpool", "py.internals.threading")
def test_fetch_all_bounded_and_propagates():
    probe = ConcurrencyProbe()

    def fetch(i):
        with probe:
            time.sleep(0.01)
            return i * 2

    assert ex.fetch_all(range(12), fetch, max_workers=3) == [i * 2 for i in range(12)]
    assert probe.max_active <= 3

    def bad(i):
        if i == 2:
            raise ConnectionError("503")
        return i

    with pytest.raises(ConnectionError):
        ex.fetch_all(range(5), bad, max_workers=2)


@pytest.mark.topic("py.internals.futures", "py.http.partial_failures")
def test_embed_settled_collects_errors_by_index():
    def embed(text):
        if "bad" in text:
            raise ValueError(f"cannot embed {text}")
        time.sleep(0.002 * (len(text) % 4))
        return [float(len(text))]

    texts = ["hello", "bad-1", "worlds", "bad-2", "x"]
    results, errors = ex.embed_settled(texts, embed, max_workers=4)
    assert results == [[5.0], None, [6.0], None, [1.0]]
    assert sorted(errors) == [1, 3]
    assert all(isinstance(e, ValueError) for e in errors.values())
    assert "bad-2" in str(errors[3])
    assert ex.embed_settled([], embed) == ([], {})


@pytest.mark.topic("py.internals.futures", "py.internals.threadpool")
def test_first_successful_does_not_wait_for_slow_calls():
    release = threading.Event()
    finished: list[str] = []

    def call(endpoint):
        if endpoint == "fast":
            return "answer-from-fast"
        release.wait(timeout=5)  # slow replicas block until the test releases them
        finished.append(endpoint)
        return f"answer-from-{endpoint}"

    try:
        result = ex.first_successful(["slow-a", "fast", "slow-b"], call, max_workers=3)
        assert result == "answer-from-fast"
        assert finished == [], "returned only after slow calls finished (did you use `with`?)"
    finally:
        release.set()


@pytest.mark.topic("py.internals.futures")
def test_first_successful_skips_failures_and_groups_errors():
    def flaky(endpoint):
        if endpoint.startswith("down"):
            raise TimeoutError(endpoint)
        time.sleep(0.02)
        return endpoint

    assert ex.first_successful(["down-1", "up", "down-2"], flaky) == "up"

    with pytest.raises(ExceptionGroup) as info:
        ex.first_successful(["down-1", "down-2"], flaky)
    assert len(info.value.exceptions) == 2
    assert all(isinstance(e, TimeoutError) for e in info.value.exceptions)

    with pytest.raises(ValueError):
        ex.first_successful([], flaky)


@pytest.mark.topic("py.internals.threading")
def test_token_meter_is_thread_safe():
    meter = ex.TokenMeter()
    barrier = threading.Barrier(8, timeout=5)

    def worker(n):
        barrier.wait()
        for _ in range(2000):
            meter.add("gpt-4o-mini", 3)
            meter.add(f"model-{n % 2}", 1)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    totals = meter.totals()
    assert totals == {"gpt-4o-mini": 48_000, "model-0": 8000, "model-1": 8000}
    totals["gpt-4o-mini"] = 0
    assert meter.totals()["gpt-4o-mini"] == 48_000, "totals() must return a copy"
