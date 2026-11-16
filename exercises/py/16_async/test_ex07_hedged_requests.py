import asyncio
import functools
import time

import pytest

from study.loader import load

ex = load(__file__, "ex07_hedged_requests")


def deadline(seconds: float = 3.0):
    """Turn a hang in a buggy solution (lost task_done, unresolved future...) into a failure."""

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            async with asyncio.timeout(seconds):
                return await fn(*args, **kwargs)

        return wrapper

    return deco


class Provider(Exception):
    pass


class Script:
    """Each call takes the next (delay, outcome) step; outcome is a value or an exception."""

    def __init__(self, *steps: tuple[float, object]) -> None:
        self.steps = list(steps)
        self.started = 0
        self.cancelled: list[int] = []
        self.finished: list[int] = []

    def make(self, step: tuple[float, object], n: int):
        async def call() -> object:
            delay, outcome = step
            try:
                await asyncio.sleep(delay)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
            except asyncio.CancelledError:
                self.cancelled.append(n)
                raise
            finally:
                self.finished.append(n)

        return call

    def calls(self):
        """One independent call per step (for first_success / gather_within)."""
        return [self.make(step, n) for n, step in enumerate(self.steps)]

    async def __call__(self) -> object:
        """Same call repeated (for hedged): attempt k uses step k."""
        n = self.started
        self.started += 1
        return await self.make(self.steps[n], n)()


@pytest.mark.topic("py.async.wait", "py.async.first_completed", "py.async.task_cancel")
@deadline()
async def test_first_success_returns_fastest_and_cancels_losers():
    s = Script((0.2, "openai"), (0.02, "anthropic"), (0.2, "local"))
    start = time.perf_counter()
    assert await ex.first_success(s.calls()) == (1, "anthropic")
    assert time.perf_counter() - start < 0.15
    assert sorted(s.cancelled) == [0, 2]
    assert sorted(s.finished) == [0, 1, 2]  # losers awaited: cleanup already ran


@pytest.mark.topic("py.async.first_completed", "py.async.done_pending")
@deadline()
async def test_first_success_skips_fast_failures():
    s = Script((0.01, Provider("429")), (0.05, "ok"), (0.2, "slow"))
    assert await ex.first_success(s.calls()) == (1, "ok")
    assert s.cancelled == [2]


@pytest.mark.topic("py.async.first_completed", "py.async.gather_exceptions")
@deadline()
async def test_first_success_all_fail():
    s = Script((0.03, Provider("a")), (0.01, Provider("b")))
    with pytest.raises(ExceptionGroup) as info:
        await ex.first_success(s.calls())
    assert [str(e) for e in info.value.exceptions] == ["a", "b"]  # index order
    with pytest.raises(ValueError):
        await ex.first_success([])


@pytest.mark.topic("py.async.first_completed")
@deadline()
async def test_hedge_not_needed_when_fast():
    s = Script((0.01, "fast"), (0.01, "backup"))
    assert await ex.hedged(s, hedge_after=0.05) == "fast"
    assert s.started == 1


@pytest.mark.topic("py.async.first_completed", "py.async.timeouts", "py.async.cancellederror")
@deadline()
async def test_hedge_backup_wins_over_slow_primary():
    s = Script((0.3, "primary"), (0.01, "backup"))
    start = time.perf_counter()
    assert await ex.hedged(s, hedge_after=0.03) == "backup"
    elapsed = time.perf_counter() - start
    assert 0.035 <= elapsed < 0.2
    assert s.started == 2
    assert s.cancelled == [0]


@pytest.mark.topic("py.async.first_completed")
@deadline()
async def test_hedge_failure_triggers_immediate_replacement():
    s = Script((0.0, Provider("500")), (0.01, "retry-ok"))
    start = time.perf_counter()
    assert await ex.hedged(s, hedge_after=0.15) == "retry-ok"
    assert time.perf_counter() - start < 0.1  # did not wait for the hedge timer
    assert s.started == 2


@pytest.mark.topic("py.async.first_completed")
@deadline()
async def test_hedge_respects_max_attempts_and_reraises_last():
    s = Script((0.02, Provider("first")), (0.03, Provider("second")), (0.0, "never used"))
    with pytest.raises(Provider, match="second"):
        await ex.hedged(s, hedge_after=0.01, max_attempts=2)
    assert s.started == 2

    s3 = Script((0.1, "a"), (0.1, "b"), (0.1, "c"), (0.0, "d"))
    assert await ex.hedged(s3, hedge_after=0.01, max_attempts=3) == "a"
    assert s3.started == 3
    assert sorted(s3.cancelled) == [1, 2]


@pytest.mark.topic(
    "py.async.wait", "py.async.all_completed", "py.async.done_pending", "py.async.task_cancelled"
)
@deadline()
async def test_gather_within_deadline():
    s = Script((0.01, "bm25"), (0.2, "slow-vector"), (0.02, Provider("down")), (0.03, "dense"))
    start = time.perf_counter()
    results, late = await ex.gather_within(s.calls(), deadline=0.08)
    assert time.perf_counter() - start < 0.18
    assert results == {0: "bm25", 3: "dense"}
    assert late == [1]
    assert s.cancelled == [1]
    assert sorted(s.finished) == [0, 1, 2, 3]


@pytest.mark.topic("py.async.all_completed")
@deadline()
async def test_gather_within_all_in_time_returns_early():
    s = Script((0.01, "a"), (0.02, "b"))
    start = time.perf_counter()
    assert await ex.gather_within(s.calls(), deadline=1.0) == ({0: "a", 1: "b"}, [])
    assert time.perf_counter() - start < 0.2  # ALL_COMPLETED: returns as soon as all finish
    assert await ex.gather_within([], deadline=0.1) == ({}, [])
