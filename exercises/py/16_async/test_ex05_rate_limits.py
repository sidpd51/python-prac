import asyncio
import time

import pytest

from study.loader import load

ex = load(__file__, "ex05_rate_limits")


class InFlight:
    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.current = 0
        self.max_seen = 0
        self.started: list[int] = []

    async def call(self, x: int) -> int:
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        self.started.append(x)
        try:
            await asyncio.sleep(self.delay)
            if x < 0:
                raise ValueError(x)
            return x * x
        finally:
            self.current -= 1


@pytest.mark.topic("py.async.semaphore", "py.async.io_bound")
async def test_bounded_map_limits_in_flight():
    tracker = InFlight()
    start = time.perf_counter()
    result = await ex.bounded_map(tracker.call, range(10), limit=3)
    elapsed = time.perf_counter() - start
    assert result == [x * x for x in range(10)]
    assert tracker.max_seen == 3  # never more than the limit ... but it did reach it
    # 10 items, 3 at a time -> 4 waves of 0.02 s (sequential would be 0.2 s)
    assert 0.07 <= elapsed < 0.18


@pytest.mark.topic("py.async.semaphore")
async def test_bounded_map_limit_one_is_sequential_and_order_kept():
    tracker = InFlight(delay=0.005)
    assert await ex.bounded_map(tracker.call, [3, 1, 2], limit=1) == [9, 1, 4]
    assert tracker.max_seen == 1
    assert await ex.bounded_map(tracker.call, [], limit=2) == []


@pytest.mark.topic("py.async.semaphore", "py.async.resource_management")
async def test_bounded_map_errors_release_permits():
    tracker = InFlight(delay=0.005)
    with pytest.raises(ValueError):
        await ex.bounded_map(tracker.call, [1, -1, 2], limit=2)
    await asyncio.sleep(0.03)
    assert tracker.current == 0
    with pytest.raises(ValueError):
        await ex.bounded_map(tracker.call, [1], limit=0)


@pytest.mark.topic("py.async.lock")
async def test_token_bucket_burst_then_rate():
    bucket = ex.TokenBucket(rate=25, capacity=2)  # one token per 0.04 s
    t0 = time.perf_counter()
    stamps: list[float] = []
    order: list[int] = []

    async def client(i: int) -> None:
        await bucket.acquire()
        stamps.append(time.perf_counter() - t0)
        order.append(i)

    await asyncio.gather(*(client(i) for i in range(6)))
    assert order == list(range(6))  # FIFO
    assert stamps[1] < 0.02  # the burst (capacity) is served immediately
    for k in range(2, 6):  # afterwards: never faster than the refill rate
        assert stamps[k] >= (k - 1) * 0.04 * 0.85
    assert stamps[-1] < 0.4


@pytest.mark.topic("py.async.lock")
async def test_token_bucket_multi_token_and_validation():
    bucket = ex.TokenBucket(rate=50, capacity=5)
    start = time.perf_counter()
    await bucket.acquire(5)
    await bucket.acquire(2)  # bucket empty -> needs 2 / 50 = 0.04 s
    elapsed = time.perf_counter() - start
    assert 0.03 <= elapsed < 0.2
    with pytest.raises(ValueError):
        await bucket.acquire(6)
    with pytest.raises(ValueError):
        ex.TokenBucket(rate=0, capacity=1)
    with pytest.raises(ValueError):
        ex.TokenBucket(rate=1, capacity=0)


@pytest.mark.topic("py.async.lock")
async def test_token_bucket_refills_but_caps_at_capacity():
    bucket = ex.TokenBucket(rate=100, capacity=2)
    await bucket.acquire(2)
    await asyncio.sleep(0.1)  # would be 10 tokens if uncapped
    start = time.perf_counter()
    await bucket.acquire(2)  # 2 available instantly
    assert time.perf_counter() - start < 0.015
    await bucket.acquire(2)  # empty again -> wait ~0.02 s
    assert time.perf_counter() - start >= 0.018
