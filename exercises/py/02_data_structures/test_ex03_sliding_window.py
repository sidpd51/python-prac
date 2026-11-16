import time

import pytest

from study.loader import load

ex = load(__file__, "ex03_sliding_window")


@pytest.mark.topic("py.data_structures.deque", "py.data_structures.stack_queue")
def test_rate_limiter_basic_window():
    rl = ex.RateLimiter(limit=3, window=10.0)
    assert [rl.allow(t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert rl.allow(9.99) is False
    assert rl.allow(10.0) is True  # t=0 expired: window is (0, 10]
    assert rl.allow(10.5) is False
    assert rl.allow(11.0) is True  # t=1 expired


@pytest.mark.topic("py.data_structures.deque")
def test_rejected_requests_are_not_recorded():
    rl = ex.RateLimiter(limit=1, window=5.0)
    assert rl.allow(0) is True
    for t in (1, 2, 3, 4):
        assert rl.allow(t) is False
    assert rl.allow(5) is True  # only t=0 counted, rejected ones did not extend the block


@pytest.mark.topic("py.data_structures.deque", "py.data_structures.stack_queue")
def test_remaining_does_not_record():
    rl = ex.RateLimiter(limit=2, window=1.0)
    assert rl.remaining(0) == 2
    assert rl.remaining(0) == 2
    rl.allow(0)
    assert rl.remaining(0.5) == 1
    rl.allow(0.5)
    assert rl.remaining(0.9) == 0
    assert rl.remaining(1.0) == 1
    assert rl.remaining(2.0) == 2


@pytest.mark.topic("py.data_structures.deque")
def test_rate_limiter_validation():
    rl = ex.RateLimiter(2, 1.0)
    rl.allow(5)
    with pytest.raises(ValueError):
        rl.allow(4)
    for bad in [(0, 1.0), (-1, 1.0), (1, 0), (1, -2.0)]:
        with pytest.raises(ValueError):
            ex.RateLimiter(*bad)


@pytest.mark.topic("py.data_structures.complexity", "py.data_structures.deque")
def test_rate_limiter_is_fast_with_large_limit():
    rl = ex.RateLimiter(limit=5_000, window=10_000.0)
    start = time.perf_counter()
    allowed = sum(rl.allow(t * 1.0) for t in range(30_000))
    elapsed = time.perf_counter() - start
    # blocks of 5k ticks alternate: allowed, rejected (window full), allowed (old ones expired)...
    assert allowed == 15_000
    assert elapsed < 1.0, "allow() is O(window) - rebuilding/scanning the log on every call?"


@pytest.mark.topic("py.data_structures.deque", "py.data_structures.complexity")
def test_moving_average():
    assert ex.moving_average([2, 4, 6, 8], 2) == pytest.approx([2, 3, 5, 7])
    assert ex.moving_average([1, 2, 3], 5) == pytest.approx([1, 1.5, 2])
    assert ex.moving_average([], 3) == []
    assert ex.moving_average(iter([10.0, 0.0]), 1) == pytest.approx([10.0, 0.0])
    with pytest.raises(ValueError):
        ex.moving_average([1], 0)


@pytest.mark.topic("py.data_structures.deque", "py.data_structures.stack_queue")
def test_tail_streams():
    consumed = []

    def lines():
        for i in range(100_000):
            consumed.append(i)
            yield f"line {i}"

    assert ex.tail(lines(), 3) == ["line 99997", "line 99998", "line 99999"]
    assert len(consumed) == 100_000
    assert ex.tail(["a", "b"], 5) == ["a", "b"]
    assert ex.tail(["a", "b"], 0) == []
    with pytest.raises(ValueError):
        ex.tail(["a"], -1)
