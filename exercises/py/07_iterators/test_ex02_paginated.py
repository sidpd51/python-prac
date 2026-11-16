import ast
import inspect
import textwrap
from itertools import islice

import pytest

from study.loader import load

ex = load(__file__, "ex02_paginated")

PAGES = [["d1", "d2"], ["d3", "d4"], ["d5"]]


@pytest.mark.topic("py.iterators.custom", "py.iterators.protocol")
def test_yields_all_documents_in_order():
    client = ex.FakeClient(PAGES)
    pager = ex.DocumentPager(client, page_size=2)
    assert list(pager) == ["d1", "d2", "d3", "d4", "d5"]
    assert client.calls == [(None, 2), ("1", 2), ("2", 2)]
    assert pager.pages_fetched == 3


@pytest.mark.topic("py.iterators.dunder_iter", "py.iterators.protocol")
def test_is_an_iterator():
    pager = ex.DocumentPager(ex.FakeClient(PAGES))
    assert iter(pager) is pager
    assert hasattr(pager, "__next__")


@pytest.mark.topic("py.iterators.dunder_next")
def test_lazy_fetching():
    client = ex.FakeClient(PAGES)
    pager = ex.DocumentPager(client)
    assert client.calls == [], "no request at construction"
    assert next(pager) == "d1"
    assert len(client.calls) == 1
    assert next(pager) == "d2"
    assert len(client.calls) == 1, "second page fetched too early"
    assert next(pager) == "d3"
    assert len(client.calls) == 2


@pytest.mark.topic("py.iterators.dunder_next")
def test_stops_early_without_extra_requests():
    client = ex.FakeClient([["a", "b"], ["c"], ["d"]])
    assert list(islice(ex.DocumentPager(client), 2)) == ["a", "b"]
    assert len(client.calls) == 1


@pytest.mark.topic("py.iterators.dunder_next")
def test_skips_empty_pages():
    client = ex.FakeClient([[], ["x"], [], [], ["y"]])
    assert list(ex.DocumentPager(client)) == ["x", "y"]
    assert len(client.calls) == 5
    assert list(ex.DocumentPager(ex.FakeClient([[]]))) == []


@pytest.mark.topic("py.iterators.stopiteration", "py.iterators.exhaustion")
def test_stays_exhausted_without_more_requests():
    client = ex.FakeClient([["only"]])
    pager = ex.DocumentPager(client)
    assert list(pager) == ["only"]
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(pager)
    assert len(client.calls) == 1
    assert list(pager) == []


@pytest.mark.topic("py.iterators.custom")
def test_validation_and_no_generators():
    with pytest.raises(ValueError):
        ex.DocumentPager(ex.FakeClient(PAGES), page_size=0)
    src = textwrap.dedent(inspect.getsource(ex.DocumentPager))
    assert not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(ast.parse(src))), (
        "write the protocol by hand - no yield"
    )
