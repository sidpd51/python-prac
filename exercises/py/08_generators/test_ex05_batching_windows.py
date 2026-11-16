import itertools

import pytest

from study.loader import load

ex = load(__file__, "ex05_batching_windows")


@pytest.mark.topic("py.generators.memory_efficiency", "py.genai.batching")
def test_batched():
    assert list(ex.batched(range(7), 3)) == [(0, 1, 2), (3, 4, 5), (6,)]
    assert list(ex.batched([], 3)) == []


@pytest.mark.topic("py.generators.memory_efficiency")
def test_batched_infinite_and_eager_validation():
    assert next(ex.batched(itertools.count(), 2)) == (0, 1)
    with pytest.raises(ValueError):
        ex.batched([1], 0)


@pytest.mark.topic("py.generators.list_vs_generator")
def test_sliding_window():
    assert list(ex.sliding_window("abcd", 2)) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert list(ex.sliding_window("ab", 3)) == []
    assert next(ex.sliding_window(itertools.count(), 3)) == (0, 1, 2)


@pytest.mark.topic("py.genai.batching", "py.genai.tokenization")
def test_token_budget_batches():
    texts = ["aaaa", "bb", "cccccc", "d", "eeeeeeeeeeee", "ff"]
    batches = list(ex.token_budget_batches(texts, max_tokens=7))
    assert batches == [["aaaa", "bb"], ["cccccc", "d"], ["eeeeeeeeeeee"], ["ff"]]
    assert list(ex.token_budget_batches([], 10)) == []


@pytest.mark.topic("py.genai.batching")
def test_token_budget_custom_counter():
    words = lambda s: len(s.split())  # noqa: E731
    texts = ["one two", "three", "four five six", "seven"]
    assert list(ex.token_budget_batches(texts, 3, words)) == [
        ["one two", "three"],
        ["four five six"],
        ["seven"],
    ]
