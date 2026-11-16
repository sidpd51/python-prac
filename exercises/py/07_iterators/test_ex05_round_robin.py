import ast
import inspect
import itertools
import textwrap

import pytest

from study.loader import load

ex = load(__file__, "ex05_round_robin")


class Counting:
    def __init__(self, items):
        self._it = iter(items)
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._it)
        self.pulled += 1
        return item


@pytest.mark.topic("py.iterators.custom", "py.iterators.protocol")
def test_rotator_cycles_forever():
    rot = ex.KeyRotator(["k1", "k2", "k3"])
    assert iter(rot) is rot
    assert list(itertools.islice(rot, 7)) == ["k1", "k2", "k3", "k1", "k2", "k3", "k1"]


@pytest.mark.topic("py.iterators.custom")
def test_rotator_validation():
    with pytest.raises(ValueError):
        ex.KeyRotator([])
    with pytest.raises(ValueError):
        ex.KeyRotator(["a", "b", "a"])
    rot = ex.KeyRotator(["a", "b"])
    with pytest.raises(KeyError):
        rot.disable("zzz")
    rot.disable("a")
    with pytest.raises(KeyError):
        rot.disable("a")


@pytest.mark.topic("py.iterators.dunder_next")
def test_disable_upcoming_key():
    rot = ex.KeyRotator(["k1", "k2", "k3"])
    assert next(rot) == "k1"
    rot.disable("k2")
    assert rot.active == ("k1", "k3")
    assert [next(rot) for _ in range(4)] == ["k3", "k1", "k3", "k1"]


@pytest.mark.topic("py.iterators.dunder_next")
def test_disable_just_returned_key():
    rot = ex.KeyRotator(["k1", "k2", "k3"])
    assert next(rot) == "k1"
    rot.disable("k1")
    assert [next(rot) for _ in range(3)] == ["k2", "k3", "k2"]


@pytest.mark.topic("py.iterators.dunder_next")
def test_disable_last_key_wraps_around():
    rot = ex.KeyRotator(["k1", "k2", "k3"])
    assert [next(rot) for _ in range(3)] == ["k1", "k2", "k3"]
    rot.disable("k3")
    assert next(rot) == "k1"
    rot = ex.KeyRotator(["k1", "k2", "k3"])
    next(rot)
    next(rot)  # k2 returned, k3 is next
    rot.disable("k3")
    assert next(rot) == "k1"


@pytest.mark.topic("py.iterators.stopiteration", "py.iterators.exhaustion")
def test_all_disabled_stops_for_good():
    rot = ex.KeyRotator(["a", "b"])
    next(rot)
    rot.disable("a")
    rot.disable("b")
    assert rot.active == ()
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(rot)
    assert list(rot) == []


@pytest.mark.topic("py.iterators.custom", "py.iterators.protocol")
def test_round_robin_interleaves():
    rr = ex.RoundRobin("ABC", "D", "EF")
    assert iter(rr) is rr
    assert list(rr) == ["A", "D", "E", "B", "F", "C"]
    assert list(ex.RoundRobin()) == []
    assert list(ex.RoundRobin([], [1], [])) == [1]


@pytest.mark.topic("py.iterators.dunder_next")
def test_round_robin_is_lazy():
    a, b = Counting(itertools.count()), Counting(["x", "y"])
    rr = ex.RoundRobin(a, b)
    assert a.pulled == b.pulled == 0
    assert list(itertools.islice(rr, 3)) == [0, "x", 1]
    assert (a.pulled, b.pulled) == (2, 1)
    assert list(itertools.islice(rr, 4)) == ["y", 2, 3, 4], "infinite source keeps going"


@pytest.mark.topic("py.iterators.stopiteration", "py.iterators.exhaustion")
def test_round_robin_stays_exhausted():
    rr = ex.RoundRobin([1], [2])
    assert list(rr) == [1, 2]
    with pytest.raises(StopIteration):
        next(rr)
    with pytest.raises(StopIteration):
        next(rr)


@pytest.mark.topic("py.iterators.custom")
def test_no_generators_or_cycle():
    assert list(ex.RoundRobin([1])) == [1]
    for cls in (ex.KeyRotator, ex.RoundRobin):
        src = textwrap.dedent(inspect.getsource(cls))
        nodes = list(ast.walk(ast.parse(src)))
        assert not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in nodes)
        assert "cycle" not in src
