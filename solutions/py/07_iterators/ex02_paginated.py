"""
Exercise 07.2 - A lazy, cursor-paginated API iterator             level: ★★☆
topics: iterators.custom, iterators.dunder_iter, iterators.dunder_next, iterators.protocol, iterators.stopiteration

Problem
-------
A vector-DB / document API returns results in pages:

    client.list_documents(cursor=None, page_size=100)
        -> {"items": [doc, ...], "next_cursor": "c2" | None}

Write `DocumentPager(client, page_size=100)`, a hand-written ITERATOR CLASS (implement
__iter__ and __next__; no generator functions or yield anywhere) that yields every document
across all pages, in order.

* Lazy: constructing the pager makes NO request. The first page is fetched on the first
  next(); the next page only when the current page's items are used up.
* Empty pages that still have a next_cursor happen in real APIs - skip them and keep going.
* Stop when a page has next_cursor None and its items are used up. After that, every
  further next() raises StopIteration again WITHOUT calling the client.
* __iter__ returns self (it's an iterator, so it's one-pass).
* `pages_fetched` attribute: number of requests made so far.
* page_size < 1 -> ValueError in __init__.

FakeClient (given, don't change) simulates the API and records its calls.

Interviewers look for
---------------------
* The iterator protocol: __iter__ returns self, __next__ returns the next item or raises
  StopIteration - and keeps raising it once exhausted.
* State machine thinking: buffer, cursor, "done" flag.
* Laziness = no wasted requests (cost, rate limits) when the consumer stops early.
* In real code you'd write this as a generator - know both.
"""

from collections import deque
from typing import Any

__stub_keep__ = ["FakeClient"]


class FakeClient:
    """In-memory stand-in for a paginated HTTP API."""

    def __init__(self, pages: list[list[Any]]) -> None:
        self._pages = pages
        self.calls: list[tuple[str | None, int]] = []

    def list_documents(self, cursor: str | None = None, page_size: int = 100) -> dict:
        self.calls.append((cursor, page_size))
        index = 0 if cursor is None else int(cursor)
        items = self._pages[index] if index < len(self._pages) else []
        has_more = index + 1 < len(self._pages)
        return {"items": list(items), "next_cursor": str(index + 1) if has_more else None}


class DocumentPager:
    def __init__(self, client: Any, page_size: int = 100) -> None:
        if page_size < 1:
            raise ValueError("page_size must be >= 1")
        self._client = client
        self._page_size = page_size
        self._buffer: deque[Any] = deque()
        self._cursor: str | None = None
        self._last_page_seen = False
        self.pages_fetched = 0

    def __iter__(self) -> "DocumentPager":
        return self

    def __next__(self) -> Any:
        while not self._buffer:
            if self._last_page_seen:
                raise StopIteration
            page = self._client.list_documents(cursor=self._cursor, page_size=self._page_size)
            self.pages_fetched += 1
            self._buffer.extend(page["items"])
            self._cursor = page["next_cursor"]
            self._last_page_seen = self._cursor is None
        return self._buffer.popleft()
