import asyncio
import time

import pytest

from study.loader import load

ex = load(__file__, "ex04_timeouts_and_budgets")


class SlowLLM:
    """complete(): sleeps `delay`, records whether it was cancelled and whether cleanup ran."""

    def __init__(self, delay: float, reply: str = "42") -> None:
        self.delay = delay
        self.reply = reply
        self.calls = 0
        self.cancelled = 0
        self.closed = 0

    async def complete(self) -> str:
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
            return self.reply
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.closed += 1  # e.g. release the HTTP connection


@pytest.mark.topic("py.async.wait_for", "py.async.timeouts")
async def test_fallback_fast_call_returns_result():
    llm = SlowLLM(0.01)
    assert await ex.call_with_fallback(llm.complete, timeout=0.5, fallback="fb") == "42"
    assert llm.cancelled == 0


@pytest.mark.topic("py.async.wait_for", "py.async.timeout_cancellation", "py.async.cleanup_finally")
async def test_fallback_on_timeout_cancels_the_call():
    llm = SlowLLM(0.2)
    start = time.perf_counter()
    assert await ex.call_with_fallback(llm.complete, timeout=0.03, fallback="fb") == "fb"
    assert time.perf_counter() - start < 0.15
    assert llm.cancelled == 1 and llm.closed == 1


@pytest.mark.topic("py.async.best_practices")
async def test_fallback_does_not_swallow_other_errors():
    async def broken() -> str:
        raise KeyError("model")

    with pytest.raises(KeyError):
        await ex.call_with_fallback(broken, timeout=0.5, fallback="fb")


@pytest.mark.topic("py.async.timeouts", "py.async.wait_for")
async def test_retry_until_success():
    delays = iter([0.2, 0.2, 0.0])
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(next(delays))
        return "ok"

    start = time.perf_counter()
    assert await ex.retry_with_timeout(call, attempts=3, per_attempt_timeout=0.03, backoff=0.01) == "ok"
    elapsed = time.perf_counter() - start
    assert calls == 3
    assert elapsed >= 0.03 * 2 + 0.01 + 0.02 - 0.01  # two timeouts + backoffs 0.01, 0.02
    assert elapsed < 0.4


@pytest.mark.topic("py.async.timeouts")
async def test_retry_reraises_last_and_respects_error_kinds():
    attempts: list[str] = []

    async def conn_then_timeout() -> str:
        attempts.append("x")
        if len(attempts) == 1:
            raise ConnectionError("reset")
        await asyncio.sleep(0.2)
        return "late"

    with pytest.raises(TimeoutError):
        await ex.retry_with_timeout(conn_then_timeout, attempts=2, per_attempt_timeout=0.02)
    assert len(attempts) == 2

    async def bad_request() -> str:
        attempts.append("bad")
        raise ValueError("400")

    attempts.clear()
    with pytest.raises(ValueError):
        await ex.retry_with_timeout(bad_request, attempts=5, per_attempt_timeout=0.1)
    assert attempts == ["bad"]  # not retried
    with pytest.raises(ValueError):
        await ex.retry_with_timeout(bad_request, attempts=0, per_attempt_timeout=0.1)


def _rag(retrieve_delay: float, generate_delay: float, log: list[str]):
    async def retrieve(q: str) -> list[str]:
        await asyncio.sleep(retrieve_delay)
        return [f"doc about {q}"]

    async def generate(q: str, docs: list[str]) -> str:
        try:
            await asyncio.sleep(generate_delay)
            return f"answer to {q} using {len(docs)} docs"
        finally:
            log.append("generate-cleanup")

    return retrieve, generate


@pytest.mark.topic("py.async.asyncio_timeout")
async def test_budget_met():
    log: list[str] = []
    retrieve, generate = _rag(0.01, 0.01, log)
    ans = await ex.answer_within_budget("cats", retrieve=retrieve, generate=generate, budget=0.5)
    assert ans == ex.Answer("answer to cats using 1 docs", ["doc about cats"], False)


@pytest.mark.topic("py.async.asyncio_timeout", "py.async.timeout_cancellation", "py.async.timeouts")
async def test_budget_is_shared_across_steps():
    log: list[str] = []
    # each step alone fits in 0.1 s, together they don't -> one shared budget must fire
    retrieve, generate = _rag(0.06, 0.06, log)
    start = time.perf_counter()
    ans = await ex.answer_within_budget("cats", retrieve=retrieve, generate=generate, budget=0.09)
    assert time.perf_counter() - start < 0.2
    assert ans.timed_out is True
    assert ans.text == ex.FALLBACK_TEXT
    assert ans.docs == ["doc about cats"]  # partial progress kept
    assert log == ["generate-cleanup"]  # generate was cancelled, its finally ran


@pytest.mark.topic("py.async.asyncio_timeout")
async def test_budget_exceeded_in_retrieval():
    log: list[str] = []
    retrieve, generate = _rag(0.2, 0.01, log)
    ans = await ex.answer_within_budget("q", retrieve=retrieve, generate=generate, budget=0.03)
    assert ans == ex.Answer(ex.FALLBACK_TEXT, [], True)
    assert log == []  # never got to generate


@pytest.mark.topic("py.async.best_practices", "py.async.cancellederror", "py.async.task_cancelled")
async def test_outer_cancellation_propagates():
    log: list[str] = []
    retrieve, generate = _rag(0.01, 0.3, log)
    task = asyncio.create_task(ex.answer_within_budget("q", retrieve=retrieve, generate=generate, budget=1.0))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert log == ["generate-cleanup"]
