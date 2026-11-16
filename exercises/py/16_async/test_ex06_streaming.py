import asyncio
import inspect

import pytest

from study.loader import load

ex = load(__file__, "ex06_streaming")


async def agen(obj):
    """Return obj as an async iterator.

    An unsolved stub of an async generator is a plain `async def` (no yield), so calling it
    gives a coroutine; awaiting it surfaces the stub's NotImplementedError (-> TODO).
    """
    if inspect.isawaitable(obj):
        await obj
    return obj


@pytest.fixture(autouse=True)
def _clear():
    ex.STREAMS_CLOSED.clear()


@pytest.mark.topic("py.async.async_generators", "py.async.async_for")
async def test_stream_tokens_yields_words():
    stream = await agen(ex.stream_tokens("the quick  brown fox"))
    assert inspect.isasyncgen(stream)
    assert [t async for t in stream] == ["the", "quick", "brown", "fox"]
    assert ex.STREAMS_CLOSED == ["the quick  brown fox"]


@pytest.mark.topic("py.async.async_generators", "py.async.suspension")
async def test_stream_tokens_is_lazy_and_closes_early():
    stream = await agen(ex.stream_tokens("a b c d", delay=0.001))
    assert await anext(stream) == "a"
    assert ex.STREAMS_CLOSED == []  # suspended, not finished
    await stream.aclose()
    assert ex.STREAMS_CLOSED == ["a b c d"]


@pytest.mark.topic("py.async.async_generators", "py.async.cancellation", "py.async.cleanup_finally")
async def test_stream_cleanup_on_cancel():
    async def consume() -> None:
        stream = await agen(ex.stream_tokens("w " * 100, delay=0.01))
        async for _ in stream:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.035)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert len(ex.STREAMS_CLOSED) == 1


@pytest.mark.topic("py.async.async_for", "py.async.resource_management")
async def test_collect_until_stop_sequence_closes_stream():
    stream = await agen(ex.stream_tokens("Answer: Paris <END> ignored tail"))
    assert await ex.collect_until(stream, "<END>") == "Answer: Paris"
    assert ex.STREAMS_CLOSED == ["Answer: Paris <END> ignored tail"]  # closed right away


@pytest.mark.topic("py.async.async_for")
async def test_collect_until_plain_async_iterable():
    class Tokens:  # async iterable without aclose()
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for t in ["x", "y"]:
                yield t

    assert await ex.collect_until(Tokens(), "STOP") == "x y"


def make_api(pages: dict[str | None, ex.Page]):
    fetched: list[str | None] = []

    async def fetch_page(cursor: str | None) -> ex.Page:
        fetched.append(cursor)
        await asyncio.sleep(0)
        return pages[cursor]

    return fetch_page, fetched


PAGES = {
    None: ex.Page([ex.Hit("a", 0.9), ex.Hit("b", 0.4)], "p2"),
    "p2": ex.Page([], "p3"),
    "p3": ex.Page([ex.Hit("c", 0.7)], None),
}


@pytest.mark.topic("py.async.async_iterators", "py.async.async_for")
async def test_paginator_iterates_all_pages_lazily():
    fetch_page, fetched = make_api(PAGES)
    pager = ex.Paginator(fetch_page)
    assert fetched == []  # nothing fetched on construction
    assert pager.__aiter__() is pager
    first = await anext(pager)
    assert first == ex.Hit("a", 0.9)
    assert fetched == [None]  # only the first page so far
    rest = [h.id async for h in pager]
    assert rest == ["b", "c"]
    assert fetched == [None, "p2", "p3"]
    with pytest.raises(StopAsyncIteration):
        await anext(pager)
    assert fetched == [None, "p2", "p3"]  # no extra fetch after the last page


@pytest.mark.topic("py.async.async_iterators")
async def test_paginator_single_empty_page():
    fetch_page, _ = make_api({None: ex.Page([], None)})
    assert [h async for h in ex.Paginator(fetch_page)] == []


@pytest.mark.topic("py.async.async_comprehensions")
async def test_relevant_ids():
    fetch_page, _ = make_api(PAGES)
    assert await ex.relevant_ids(ex.Paginator(fetch_page), 0.5) == ["a", "c"]

    async def hits():
        yield ex.Hit("z", 0.5)
        yield ex.Hit("y", 0.49)

    assert await ex.relevant_ids(hits(), 0.5) == ["z"]


def _delayed(value: str, delay: float, log: list[str]):
    async def call() -> str:
        try:
            await asyncio.sleep(delay)
            return value
        except asyncio.CancelledError:
            log.append(f"cancelled {value}")
            raise

    return call


@pytest.mark.topic("py.async.as_completed", "py.async.completion_order")
async def test_as_they_finish_completion_order_with_index():
    log: list[str] = []
    calls = [_delayed("slow", 0.09, log), _delayed("fast", 0.01, log), _delayed("mid", 0.05, log)]
    stream = await agen(ex.as_they_finish(calls))
    got = [pair async for pair in stream]
    assert got == [(1, "fast"), (2, "mid"), (0, "slow")]
    assert log == []


@pytest.mark.topic("py.async.as_completed", "py.async.task_cancel", "py.async.cleanup_finally")
async def test_as_they_finish_early_stop_cancels_rest():
    log: list[str] = []
    calls = [_delayed("slow", 0.2, log), _delayed("fast", 0.01, log), _delayed("slower", 0.2, log)]
    stream = await agen(ex.as_they_finish(calls))
    assert await anext(stream) == (1, "fast")
    await stream.aclose()
    assert sorted(log) == ["cancelled slow", "cancelled slower"]
