import bisect
import math
import random

import pytest

from study.loader import load

ex = load(__file__, "ex05_binary_search")


@pytest.mark.topic("py.data_structures.binary_search")
@pytest.mark.parametrize(
    ("xs", "x", "lo", "hi"),
    [
        ([], 5, 0, 0),
        ([1], 0, 0, 0),
        ([1], 1, 0, 1),
        ([1], 2, 1, 1),
        ([1, 2, 2, 2, 3], 2, 1, 4),
        ([1, 2, 2, 2, 3], 2.5, 4, 4),
        ([1, 3, 5, 7], 0, 0, 0),
        ([1, 3, 5, 7], 8, 4, 4),
        ([5, 5, 5], 5, 0, 3),
    ],
)
def test_bounds_cases(xs, x, lo, hi):
    assert ex.lower_bound(xs, x) == lo
    assert ex.upper_bound(xs, x) == hi


@pytest.mark.topic("py.data_structures.binary_search", "py.data_structures.searching")
def test_bounds_match_bisect_randomised():
    rng = random.Random(7)
    for _ in range(300):
        xs = sorted(rng.randint(0, 20) for _ in range(rng.randint(0, 30)))
        x = rng.randint(-2, 22)
        assert ex.lower_bound(xs, x) == bisect.bisect_left(xs, x), (xs, x)
        assert ex.upper_bound(xs, x) == bisect.bisect_right(xs, x), (xs, x)
        assert ex.upper_bound(xs, x) - ex.lower_bound(xs, x) == xs.count(x)


@pytest.mark.topic("py.data_structures.complexity", "py.data_structures.binary_search")
def test_bounds_are_logarithmic():
    class CountingList(list):
        reads = 0

        def __getitem__(self, i):
            CountingList.reads += 1
            return super().__getitem__(i)

    xs = CountingList(range(1_000_000))
    assert ex.lower_bound(xs, 765_432) == 765_432
    assert CountingList.reads <= 21, "not a binary search (or it copies/scans the list)"


@pytest.mark.topic("py.data_structures.binary_search", "py.data_structures.complexity")
@pytest.mark.parametrize(("lo", "hi", "threshold"), [(0, 1_000_000, 777_777), (0, 1, 0), (5, 9, 7)])
def test_first_true_finds_threshold_in_log_calls(lo, hi, threshold):
    calls = []

    def pred(i):
        calls.append(i)
        return i >= threshold

    assert ex.first_true(lo, hi, pred) == threshold
    assert len(calls) <= math.ceil(math.log2(hi - lo)) + 1
    assert all(lo <= c < hi for c in calls), "pred called outside [lo, hi)"


@pytest.mark.topic("py.data_structures.binary_search")
def test_first_true_none_all_empty():
    assert ex.first_true(0, 100, lambda i: False) == 100
    assert ex.first_true(0, 100, lambda i: True) == 0
    assert ex.first_true(10, 20, lambda i: True) == 10

    def boom(i):
        raise AssertionError("pred must not be called on an empty range")

    assert ex.first_true(5, 5, boom) == 5
    assert ex.first_true(7, 3, boom) == 3


@pytest.mark.topic("py.data_structures.searching", "py.data_structures.binary_search")
def test_first_true_context_window_use_case():
    # smallest number of retrieved chunks whose total tokens exceed the budget
    chunk_tokens = [120, 340, 90, 410, 220, 300]
    prefix = [sum(chunk_tokens[:n]) for n in range(len(chunk_tokens) + 1)]
    n = ex.first_true(0, len(prefix), lambda k: prefix[k] > 900)
    assert n == 4  # 120+340+90+410 = 960 > 900


@pytest.mark.topic("py.data_structures.searching", "py.data_structures.linear_search")
def test_events_between_half_open():
    ts = [1.0, 2.0, 2.0, 3.5, 5.0, 5.0, 8.0]
    assert ex.events_between(ts, 2.0, 5.0) == [2.0, 2.0, 3.5]
    assert ex.events_between(ts, 0, 100) == ts
    assert ex.events_between(ts, 5.0, 5.0) == []
    assert ex.events_between(ts, 6.0, 2.0) == []
    assert ex.events_between(ts, 8.0, 9.0) == [8.0]
    assert ex.events_between([], 0, 1) == []
    # agrees with a linear scan
    assert ex.events_between(ts, 1.5, 8.0) == [t for t in ts if 1.5 <= t < 8.0]
