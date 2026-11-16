import asyncio
import contextlib
import time

import pytest

from study.loader import load

ex = load(__file__, "ex02_unblock_the_loop")


class Heartbeat:
    """Counts how often the loop gets to run us; a blocked loop -> no beats."""

    def __init__(self, every: float = 0.005) -> None:
        self.every = every
        self.beats = 0
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.every)
            self.beats += 1

    async def __aenter__(self) -> "Heartbeat":
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


@pytest.mark.topic("py.async.blocking", "py.async.event_loop", "py.async.io_bound")
async def test_count_tokens_keeps_loop_responsive():
    texts = ["a b c", "one", "x y", "four words right here", "", "s t"]
    async with Heartbeat() as hb:
        start = time.perf_counter()
        counts = await ex.count_tokens_all(texts)
        elapsed = time.perf_counter() - start
    assert counts == [3, 1, 2, 4, 0, 2]
    # blocking version: ~0.30 s and 0 beats while it runs
    assert elapsed < 4 * ex.TOKENIZE_DELAY
    assert hb.beats >= 3


@pytest.mark.topic("py.async.blocking")
async def test_count_tokens_empty():
    assert await ex.count_tokens_all([]) == []


@pytest.mark.topic("py.async.sleep_vs_sleep", "py.async.asyncio_sleep")
async def test_wait_until_ready_polls_without_blocking():
    answers = iter([False, False, False, True])

    async def probe() -> bool:
        await asyncio.sleep(0)
        return next(answers)

    async with Heartbeat(every=0.002) as hb:
        calls = await ex.wait_until_ready(probe, interval=0.02, timeout=1.0)
    assert calls == 4
    assert hb.beats >= 5  # time.sleep(interval) would starve the heartbeat


@pytest.mark.topic("py.async.sleep_vs_sleep", "py.async.timeouts")
async def test_wait_until_ready_times_out():
    calls = 0

    async def never() -> bool:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)  # a real health check does I/O
        return False

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        await ex.wait_until_ready(never, interval=0.01, timeout=0.08)
    elapsed = time.perf_counter() - start
    assert 0.06 <= elapsed < 0.4
    assert calls >= 3


@pytest.mark.topic("py.async.event")
async def test_gate_releases_all_waiters_at_once():
    gate = ex.WarmupGate()
    assert gate.is_open is False
    waiters = [asyncio.create_task(gate.wait()) for _ in range(5)]
    await asyncio.sleep(0.02)
    assert not any(w.done() for w in waiters)  # parked, not spinning
    gate.open()
    assert gate.is_open is True
    assert await asyncio.gather(*waiters) == [True] * 5
    assert await gate.wait() is True  # already open -> returns immediately


@pytest.mark.topic("py.async.event", "py.async.timeouts")
async def test_gate_wait_timeout_returns_false():
    gate = ex.WarmupGate()
    start = time.perf_counter()
    assert await gate.wait(timeout=0.03) is False
    assert time.perf_counter() - start < 0.3
