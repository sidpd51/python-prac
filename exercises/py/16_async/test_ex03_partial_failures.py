import asyncio
import time

import pytest

from study.loader import load

ex = load(__file__, "ex03_partial_failures")


class RateLimited(Exception):
    pass


class FakeEmbedder:
    """embed(text): texts in `fail` raise RateLimited (the first `fail[text]` times)."""

    def __init__(self, fail: dict[str, int] | None = None, delays: dict[str, float] | None = None):
        self.fail = dict(fail or {})
        self.delays = delays or {}
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.cleaned_up: list[str] = []

    async def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        try:
            await asyncio.sleep(self.delays.get(text, 0.01))
            if self.fail.get(text, 0) > 0:
                self.fail[text] -= 1
                raise RateLimited(text)
            return [float(len(text))]
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        finally:
            self.cleaned_up.append(text)


@pytest.mark.topic("py.async.return_exceptions", "py.async.gather")
async def test_partial_keeps_good_results():
    embed = FakeEmbedder(fail={"bad": 99})
    report = await ex.embed_partial(["ok", "bad", "fine"], embed)
    assert report.vectors == [[2.0], None, [4.0]]
    assert list(report.errors) == [1]
    assert isinstance(report.errors[1], RateLimited)
    assert sorted(embed.calls) == ["bad", "fine", "ok"]


@pytest.mark.topic("py.async.return_exceptions")
async def test_partial_all_ok_and_empty():
    report = await ex.embed_partial(["a", "bb"], FakeEmbedder())
    assert report.vectors == [[1.0], [2.0]] and report.errors == {}
    empty = await ex.embed_partial([], FakeEmbedder())
    assert empty.vectors == [] and empty.errors == {}


@pytest.mark.topic("py.async.return_exceptions", "py.async.cancellederror")
async def test_partial_reraises_child_cancellation():
    async def embed(text: str) -> list[float]:
        if text == "boom":
            raise asyncio.CancelledError  # e.g. a child cancelled by someone else
        return [1.0]

    with pytest.raises(asyncio.CancelledError):
        await ex.embed_partial(["x", "boom"], embed)


@pytest.mark.topic("py.async.return_exceptions", "py.async.gather_exceptions")
async def test_retries_only_failed_indices():
    embed = FakeEmbedder(fail={"flaky": 1, "dead": 99})
    report = await ex.embed_with_retries(["ok", "flaky", "dead"], embed, rounds=3)
    assert report.vectors == [[2.0], [5.0], None]
    assert list(report.errors) == [2]
    assert embed.calls.count("ok") == 1  # succeeded first time: never re-sent
    assert embed.calls.count("flaky") == 2
    assert embed.calls.count("dead") == 3
    with pytest.raises(ValueError):
        await ex.embed_with_retries(["ok"], embed, rounds=0)


@pytest.mark.topic(
    "py.async.first_exception",
    "py.async.task_cancel",
    "py.async.cleanup_finally",
    "py.async.gather_exceptions",
)
async def test_all_or_nothing_cancels_siblings():
    embed = FakeEmbedder(fail={"bad": 1}, delays={"bad": 0.02, "slow1": 0.2, "slow2": 0.2})
    start = time.perf_counter()
    with pytest.raises(RateLimited):
        await ex.embed_all_or_nothing(["fast", "slow1", "bad", "slow2"], embed)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.15  # did not wait for the slow calls to finish
    assert sorted(embed.cancelled) == ["slow1", "slow2"]
    # cancellation was *awaited*: the siblings' finally blocks already ran
    assert sorted(embed.cleaned_up) == ["bad", "fast", "slow1", "slow2"]


@pytest.mark.topic("py.async.first_exception", "py.async.all_completed")
async def test_all_or_nothing_success_in_order():
    embed = FakeEmbedder(delays={"aaa": 0.05, "b": 0.01})
    assert await ex.embed_all_or_nothing(["aaa", "b", "cc"], embed) == [[3.0], [1.0], [2.0]]
    assert await ex.embed_all_or_nothing([], embed) == []
    assert embed.cancelled == []
