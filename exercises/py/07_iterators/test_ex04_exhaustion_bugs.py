import itertools

import pytest

from study.loader import load

ex = load(__file__, "ex04_exhaustion_bugs")


class OnePass:
    """An iterator that fails loudly if anyone tries to traverse it again after exhaustion."""

    def __init__(self, items):
        self._it = iter(items)
        self.exhausted = False
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.exhausted:
            raise AssertionError("second pass over an exhausted iterator")
        try:
            item = next(self._it)
        except StopIteration:
            self.exhausted = True
            raise
        self.pulled += 1
        return item


@pytest.mark.topic("py.iterators.exhaustion", "py.iterators.one_pass")
def test_summarize_scores_on_generator():
    scores = (s for s in [0.5, 0.75, 1.0, 0.25])
    assert ex.summarize_scores(scores) == {"count": 4, "mean": 0.625, "max": 1.0}


@pytest.mark.topic("py.iterators.one_pass")
def test_summarize_scores_single_pass_and_lists():
    src = OnePass([3.0, 1.0, 2.0])
    assert ex.summarize_scores(src) == {"count": 3, "mean": 2.0, "max": 3.0}
    assert ex.summarize_scores([-5.0, -1.0]) == {"count": 2, "mean": -3.0, "max": -1.0}


@pytest.mark.topic("py.iterators.exhaustion")
def test_summarize_scores_empty():
    with pytest.raises(ValueError):
        ex.summarize_scores(iter([]))


@pytest.mark.topic("py.iterators.exhaustion", "py.iterators.one_pass")
def test_filter_new_with_iterator_seen_ids():
    seen = OnePass(["doc-3", "doc-1"])
    candidates = ["doc-1", "doc-2", "doc-3", "doc-4", "doc-2"]
    assert ex.filter_new(candidates, seen) == ["doc-2", "doc-4", "doc-2"]
    assert seen.exhausted


@pytest.mark.topic("py.iterators.exhaustion")
def test_filter_new_generators_everywhere():
    candidates = (f"id{i}" for i in range(6))
    seen = map(str.lower, ["ID4", "ID0", "ID2"])
    assert ex.filter_new(candidates, seen) == ["id1", "id3", "id5"]
    assert ex.filter_new([], iter([])) == []


@pytest.mark.topic("py.iterators.next", "py.iterators.stopiteration")
def test_peek_keeps_first_item():
    hits = map(str.upper, ["a", "b", "c"])
    first, it = ex.peek(hits)
    assert first == "A"
    assert list(it) == ["A", "B", "C"]


@pytest.mark.topic("py.iterators.next", "py.iterators.exhaustion")
def test_peek_empty():
    first, it = ex.peek(iter([]), default="FALLBACK")
    assert first == "FALLBACK"
    assert list(it) == []
    assert ex.peek([])[0] is None


@pytest.mark.topic("py.iterators.next", "py.iterators.one_pass")
def test_peek_is_lazy_and_handles_infinite():
    src = OnePass(itertools.count())
    first, it = ex.peek(src)
    assert first == 0
    assert src.pulled == 1, "peek must pull exactly one item"
    assert list(itertools.islice(it, 4)) == [0, 1, 2, 3]
    first, it = ex.peek(None for _ in range(2))
    assert first is None
    assert list(it) == [None, None], "a None item is not the same as 'empty'"
