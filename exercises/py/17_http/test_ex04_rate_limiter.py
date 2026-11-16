import asyncio
import time

import pytest

from study.loader import load

ex = load(__file__, "ex04_rate_limiter")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)  # still yield to the event loop like a real sleep


@pytest.fixture
def clock():
    return FakeClock()


def bucket(clock, rate, capacity):
    return ex.TokenBucket(rate, capacity, clock=clock, sleep=clock.sleep)


@pytest.mark.topic("py.http.rate_limiting")
@pytest.mark.parametrize(("rate", "capacity"), [(0, 1), (-1, 1), (1, 0), (1, -5)])
def test_invalid_config(rate, capacity):
    with pytest.raises(ValueError):
        ex.TokenBucket(rate, capacity)


@pytest.mark.topic("py.http.rate_limiting")
def test_burst_then_refill(clock):
    b = bucket(clock, rate=1, capacity=3)
    assert [b.try_acquire() for _ in range(4)] == [True, True, True, False]
    clock.now += 0.5
    assert b.available() == pytest.approx(0.5)
    assert b.try_acquire() is False
    clock.now += 0.5
    assert b.try_acquire() is True
    clock.now += 100
    assert b.available() == 3, "refill is capped at capacity"


@pytest.mark.topic("py.http.rate_limiting")
def test_cost_validation(clock):
    b = bucket(clock, rate=1, capacity=3)
    for cost in (0, -1, 4):
        with pytest.raises(ValueError):
            b.try_acquire(cost)


@pytest.mark.topic("py.http.rate_limiting")
async def test_acquire_waits_exactly_the_deficit(clock):
    b = bucket(clock, rate=2, capacity=2)
    assert await b.acquire() == 0.0
    assert await b.acquire() == 0.0
    assert clock.sleeps == []
    waited = await b.acquire()
    assert waited == pytest.approx(0.5)
    assert clock.now == pytest.approx(0.5)
    assert b.available() == pytest.approx(0.0)


@pytest.mark.topic("py.http.rate_limiting")
async def test_token_per_minute_costs(clock):
    tpm = bucket(clock, rate=1000 / 60, capacity=1000)  # 1000 tokens per minute
    assert await tpm.acquire(600) == 0.0
    waited = await tpm.acquire(600)  # needs 200 more tokens -> 12 s
    assert waited == pytest.approx(12.0)
    with pytest.raises(ValueError):
        await tpm.acquire(1500)  # can never fit: fail fast, don't wait forever
    assert clock.now == pytest.approx(12.0)


@pytest.mark.topic("py.http.rate_limiting", "py.http.concurrency_limits")
async def test_fifo_big_request_not_starved(clock):
    b = bucket(clock, rate=1, capacity=4)
    assert b.try_acquire(4)
    done = []

    async def take(name, cost):
        await b.acquire(cost)
        done.append((name, clock.now))

    big = asyncio.create_task(take("big", 4))
    await asyncio.sleep(0)  # big is now waiting
    smalls = [asyncio.create_task(take(f"s{i}", 1)) for i in range(3)]
    await asyncio.gather(big, *smalls)
    assert [name for name, _ in done] == ["big", "s0", "s1", "s2"]
    assert [t for _, t in done] == pytest.approx([4.0, 5.0, 6.0, 7.0])


@pytest.mark.topic("py.http.concurrent_calls", "py.http.rate_limiting")
async def test_run_rate_limited_order_and_pacing(clock):
    b = bucket(clock, rate=1, capacity=2)
    started = []

    def make(i):
        async def call():
            started.append((i, clock.now))
            return i * 10

        return call

    results = await ex.run_rate_limited([make(i) for i in range(5)], b)
    assert results == [0, 10, 20, 30, 40]
    assert sorted(t for _, t in started) == pytest.approx([0.0, 0.0, 1.0, 2.0, 3.0])


@pytest.mark.topic("py.http.rate_limiting")
async def test_run_rate_limited_cost_and_empty(clock):
    b = bucket(clock, rate=10, capacity=10)

    async def call():
        return "ok"

    assert await ex.run_rate_limited([], b) == []
    assert await ex.run_rate_limited([call] * 3, b, cost=5) == ["ok"] * 3
    assert clock.now == pytest.approx(0.5)


@pytest.mark.topic("py.http.rate_limiting")
async def test_defaults_use_real_clock():
    b = ex.TokenBucket(rate=50, capacity=1)
    start = time.perf_counter()
    for _ in range(3):
        await b.acquire()
    elapsed = time.perf_counter() - start
    assert 0.03 <= elapsed < 0.5
