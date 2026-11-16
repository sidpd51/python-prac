import itertools

import pytest

from study.loader import load

ex = load(__file__, "ex04_yield_from")


@pytest.mark.topic("py.generators.yield_from")
def test_flatten_mixed():
    data = [1, [2, (3, [4])], "ab", iter([5]), {"k": 1}, b"xy", (), [[[]]]]
    assert list(ex.flatten(data)) == [1, 2, 3, 4, "ab", 5, {"k": 1}, b"xy"]


@pytest.mark.topic("py.generators.yield_from", "py.generators.lazy_evaluation")
def test_flatten_lazy_on_infinite():
    nested = [[0], itertools.count(1)]
    assert list(itertools.islice(ex.flatten(nested), 5)) == [0, 1, 2, 3, 4]


@pytest.mark.topic("py.generators.yield_from")
def test_counted_return_value():
    g = ex.counted("xyz")
    assert list(itertools.islice(g, 3)) == ["x", "y", "z"]
    with pytest.raises(StopIteration) as info:
        next(g)
    assert info.value.value == 3


@pytest.mark.topic("py.generators.yield_from", "py.generators.chaining")
def test_concat_with_counts():
    g = ex.concat_with_counts("ab", [1, 2, 3], [])
    items = []
    while True:
        try:
            items.append(next(g))
        except StopIteration as stop:
            result = stop.value
            break
    assert items == ["a", "b", 1, 2, 3]
    assert result == [2, 3, 0]
