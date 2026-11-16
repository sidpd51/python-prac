import ast
import inspect
import textwrap

import pytest

from study.loader import load

ex = load(__file__, "ex03_reiterable")

ROWS = [{"q": "2+2", "a": "4"}, {"q": "capital of France", "a": "Paris"}, {"q": "h2o", "a": "water"}]


class OldStyleSequence:
    """Only __getitem__ - still iterable via the legacy sequence protocol."""

    def __getitem__(self, i):
        if i >= 3:
            raise IndexError
        return i * 10


@pytest.mark.topic("py.iterators.dunder_next", "py.iterators.custom")
def test_dataset_iterator():
    it = ex.DatasetIterator(["a", "b"])
    assert iter(it) is it
    assert next(it) == "a"
    assert next(it) == "b"
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


@pytest.mark.topic("py.iterators.iterable_vs_iterator", "py.iterators.dunder_iter")
def test_dataset_is_reiterable():
    ds = ex.EvalDataset(ROWS)
    assert len(ds) == 3
    first_run = [r["a"] for r in ds]
    second_run = [r["a"] for r in ds]
    assert first_run == second_run == ["4", "Paris", "water"]


@pytest.mark.topic("py.iterators.iterable_vs_iterator", "py.iterators.iter")
def test_each_iter_call_returns_fresh_independent_iterator():
    ds = ex.EvalDataset(ROWS)
    it1, it2 = iter(ds), iter(ds)
    assert it1 is not it2
    assert isinstance(it1, ex.DatasetIterator)
    next(it1)
    assert next(it2) is ROWS[0]
    pairs = [(a["a"], b["a"]) for a in ds for b in ds if a is not b]
    assert len(pairs) == 6, "nested loops over the same dataset must work"


@pytest.mark.topic("py.iterators.one_pass", "py.iterators.exhaustion")
def test_dataset_from_generator_is_still_reiterable():
    gen = (row for row in ROWS)
    ds = ex.EvalDataset(gen)
    assert list(ds) == ROWS
    assert list(ds) == ROWS
    assert list(gen) == [], "the generator itself was consumed once, by the dataset"


@pytest.mark.topic("py.iterators.dunder_iter")
def test_iter_uses_iterator_class_not_yield():
    src = textwrap.dedent(inspect.getsource(ex.EvalDataset))
    tree = ast.parse(src)
    assert not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(tree))
    assert len(ex.EvalDataset([])) == 0
    assert list(ex.EvalDataset([])) == []


@pytest.mark.topic("py.iterators.iterable_vs_iterator", "py.iterators.iter")
def test_is_iterable():
    for obj in ([], "abc", {}, range(3), iter([]), ex.EvalDataset([]), OldStyleSequence()):
        assert ex.is_iterable(obj), obj
    for obj in (42, None, 3.14, object()):
        assert not ex.is_iterable(obj), obj


@pytest.mark.topic("py.iterators.iterable_vs_iterator", "py.iterators.protocol")
def test_is_iterator():
    ds = ex.EvalDataset(ROWS)
    assert ex.is_iterator(iter(ds))
    assert ex.is_iterator(iter([1]))
    assert ex.is_iterator(x for x in "ab")
    with open(__file__) as f:
        assert ex.is_iterator(f)  # files are their own iterators
    for obj in (ds, [1], "ab", range(3), OldStyleSequence(), 5):
        assert not ex.is_iterator(obj), obj
