"""
Exercise 16.6 - Streaming: token streams, pagination, as_completed level: ★★☆
topics: async.async_generators, async.async_iterators, async.async_for, async.async_comprehensions, async.as_completed, async.completion_order

Problem
-------
1. `async def stream_tokens(text, *, delay=0.005)` - an ASYNC GENERATOR that fakes an LLM
   streaming response: yield each whitespace-separated word of `text`, awaiting
   `asyncio.sleep(delay)` before each one. When the generator finishes OR is closed early
   (consumer breaks + aclose(), or is cancelled), append `text` to the module-level list
   STREAMS_CLOSED exactly once (try/finally).

2. `async def collect_until(stream, stop) -> str` - consume an async iterable of tokens
   with `async for`, joining them with single spaces, stopping BEFORE the first token equal
   to `stop` (a "stop sequence"). After stopping early you must close the stream: call
   `await stream.aclose()` if it has that method (don't rely on garbage collection).

3. `class Paginator(fetch_page)` - an async ITERATOR class (implement `__aiter__` and
   `__anext__`, no async generator) over a paginated vector-search API.
   `fetch_page(cursor)` is async, takes `None` for the first page and returns a `Page`
   (given). Iterate over the individual hits across all pages, fetching each page lazily
   (only when the previous page's hits are used up). Stop when a page has
   `next_cursor is None` and its hits are consumed. Empty pages in the middle are allowed.

4. `async def relevant_ids(hits, min_score) -> list[str]` - use an ASYNC COMPREHENSION to
   return the `id` of every hit from the async iterable `hits` with score >= min_score.

5. `async def as_they_finish(calls)` - an async generator: run the zero-arg async callables
   in `calls` concurrently and yield `(index, result)` pairs in COMPLETION order using
   `asyncio.as_completed`. If the consumer stops early, cancel the calls that are still
   running (finally) - no orphaned tasks.

Interviewers look for
---------------------
* An async generator is `async def` + `yield`; consume with `async for`, it has
  `aclose()` / `asend()` / `athrow()`.
* Async iterator protocol: `__aiter__` (plain def, returns self) + `async __anext__` raising
  StopAsyncIteration.
* Early `break` from `async for` does NOT run the generator's finally right away - close it
  explicitly (or `contextlib.aclosing`), or the cleanup happens at GC / loop shutdown.
* as_completed yields futures in completion order; you lose the input index unless you
  carry it in the result.
"""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field

STREAMS_CLOSED: list[str] = []


@dataclass(frozen=True)
class Hit:
    id: str
    score: float


@dataclass(frozen=True)
class Page:
    hits: list[Hit] = field(default_factory=list)
    next_cursor: str | None = None


async def stream_tokens(text: str, *, delay: float = 0.005) -> AsyncIterator[str]:
    try:
        for word in text.split():
            await asyncio.sleep(delay)
            yield word
    finally:
        STREAMS_CLOSED.append(text)


async def collect_until(stream: AsyncIterable[str], stop: str) -> str:
    out: list[str] = []
    try:
        async for tok in stream:
            if tok == stop:
                break
            out.append(tok)
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()
    return " ".join(out)


class Paginator:
    def __init__(self, fetch_page: Callable[[str | None], Awaitable[Page]]) -> None:
        self._fetch = fetch_page
        self._buffer: list[Hit] = []
        self._cursor: str | None = None
        self._started = False
        self._exhausted = False

    def __aiter__(self) -> "Paginator":
        return self

    async def __anext__(self) -> Hit:
        while not self._buffer:
            if self._exhausted:
                raise StopAsyncIteration
            page = await self._fetch(self._cursor if self._started else None)
            self._started = True
            self._buffer = list(page.hits)
            self._cursor = page.next_cursor
            self._exhausted = page.next_cursor is None
        return self._buffer.pop(0)


async def relevant_ids(hits: AsyncIterable[Hit], min_score: float) -> list[str]:
    return [h.id async for h in hits if h.score >= min_score]


async def as_they_finish[T](calls: Sequence[Callable[[], Awaitable[T]]]) -> AsyncIterator[tuple[int, T]]:
    async def tagged(i: int, call: Callable[[], Awaitable[T]]) -> tuple[int, T]:
        return i, await call()

    tasks = [asyncio.create_task(tagged(i, c)) for i, c in enumerate(calls)]
    try:
        for fut in asyncio.as_completed(tasks):
            yield await fut
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
