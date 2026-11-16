import ast
import inspect
import io
import textwrap

import pytest

from study.loader import load

ex = load(__file__, "ex01_for_internals")


class Spy:
    """An iterable that counts iter() and next() calls."""

    def __init__(self, items):
        self.items = list(items)
        self.iter_calls = 0
        self.next_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return _SpyIterator(self)


class _SpyIterator:
    def __init__(self, spy):
        self.spy = spy
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.spy.next_calls += 1
        if self.i >= len(self.spy.items):
            raise StopIteration
        self.i += 1
        return self.spy.items[self.i - 1]


@pytest.mark.topic("py.iterators.for_internals", "py.iterators.iter", "py.iterators.next")
def test_for_each_processes_all_and_counts():
    seen = []
    spy = Spy(["doc1", "doc2", "doc3"])
    assert ex.for_each(spy, seen.append) == 3
    assert seen == ["doc1", "doc2", "doc3"]
    assert spy.iter_calls == 1
    assert spy.next_calls == 4  # 3 items + the call that raised StopIteration


@pytest.mark.topic("py.iterators.for_internals")
def test_for_each_empty_and_no_for_statement():
    assert ex.for_each([], print) == 0
    tree = ast.parse(textwrap.dedent(inspect.getsource(ex.for_each)))
    banned = (ast.For, ast.comprehension)
    assert not any(isinstance(n, banned) for n in ast.walk(tree)), "don't use for/comprehensions"


@pytest.mark.topic("py.iterators.stopiteration", "py.iterators.for_internals")
def test_stopiteration_from_callback_propagates():
    def fn(item):
        if item == 2:
            raise StopIteration("from the body")

    with pytest.raises(StopIteration, match="from the body"):
        ex.for_each([1, 2, 3], fn)


@pytest.mark.topic("py.iterators.next")
def test_first():
    assert ex.first(["a", "b"]) == "a"
    assert ex.first([], default=None) is None
    assert ex.first(iter([]), "fallback") == "fallback"
    with pytest.raises(ValueError):
        ex.first([])


@pytest.mark.topic("py.iterators.next", "py.iterators.one_pass")
def test_first_consumes_only_one_item():
    it = iter(range(5))
    assert ex.first(it) == 0
    assert next(it) == 1
    assert ex.first(x for x in "xyz") == "x"


@pytest.mark.topic("py.iterators.iter", "py.iterators.stopiteration")
def test_read_records_two_arg_iter():
    stream = io.BytesIO(b"abcdefgh12")
    chunks = ex.read_records(stream, 4)
    assert stream.tell() == 0, "must be lazy: nothing read before next()"
    assert iter(chunks) is chunks
    assert list(chunks) == [b"abcd", b"efgh", b"12"]
    assert list(ex.read_records(io.BytesIO(b""), 3)) == []
    with pytest.raises(ValueError):
        ex.read_records(io.BytesIO(b"x"), 0)
