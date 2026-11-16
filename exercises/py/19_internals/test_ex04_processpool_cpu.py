import math
import sys
import sysconfig

import pytest

from study.loader import load

ex = load(__file__, "ex04_processpool_cpu")


@pytest.mark.topic("py.internals.gil")
def test_gil_status():
    status = ex.gil_status()
    assert set(status) == {"free_threaded_build", "gil_enabled"}
    assert all(isinstance(v, bool) for v in status.values())
    assert status["free_threaded_build"] == (sysconfig.get_config_var("Py_GIL_DISABLED") == 1)
    if not status["free_threaded_build"]:
        assert status["gil_enabled"], "a default build always has the GIL"
    if hasattr(sys, "_is_gil_enabled"):
        assert status["gil_enabled"] == sys._is_gil_enabled()


@pytest.mark.topic("py.internals.multiprocessing", "py.internals.cpu_vs_io")
@pytest.mark.parametrize(
    ("n", "parts", "expected"),
    [
        (10, 3, [range(0, 4), range(4, 7), range(7, 10)]),
        (9, 3, [range(0, 3), range(3, 6), range(6, 9)]),
        (2, 5, [range(0, 1), range(1, 2)]),
        (0, 3, []),
        (5, 1, [range(0, 5)]),
    ],
)
def test_split_ranges(n, parts, expected):
    assert ex.split_ranges(n, parts) == expected


@pytest.mark.topic("py.internals.multiprocessing")
def test_split_ranges_properties_and_validation():
    for n in range(0, 40):
        for parts in range(1, 7):
            chunks = ex.split_ranges(n, parts)
            assert [i for r in chunks for i in r] == list(range(n))
            sizes = [len(r) for r in chunks]
            assert all(sizes) and (not sizes or max(sizes) - min(sizes) <= 1)
            assert sizes == sorted(sizes, reverse=True)
    for bad in [(10, 0), (-1, 2)]:
        with pytest.raises(ValueError):
            ex.split_ranges(*bad)


@pytest.mark.topic("py.internals.processpool", "py.internals.cpu_vs_io")
def test_parallel_sum_of_squares():
    n = 200_000
    assert ex.parallel_sum_of_squares(n, workers=2) == sum(i * i for i in range(n))
    assert ex.parallel_sum_of_squares(0) == 0
    assert ex.parallel_sum_of_squares(3, workers=8) == 5


@pytest.mark.topic("py.internals.processpool")
def test_run_cpu_bound_order_and_errors():
    items = [5, 0, 10, 3]
    assert ex.run_cpu_bound(math.factorial, items, max_workers=2) == [120, 1, 3628800, 6]
    assert ex.run_cpu_bound(math.isqrt, range(50), max_workers=2, chunksize=10) == [
        math.isqrt(i) for i in range(50)
    ]
    with pytest.raises(ValueError):  # raised in the child, re-raised here
        ex.run_cpu_bound(math.factorial, [3, -1], max_workers=2)


@pytest.mark.topic("py.internals.processpool", "py.internals.concurrency_models")
def test_run_cpu_bound_rejects_unpicklable_before_starting():
    def local_square(x):
        return x * x

    with pytest.raises(TypeError) as info:
        ex.run_cpu_bound(lambda x: x * 2, [1, 2])
    assert "picklable" in str(info.value)
    assert info.value.__cause__ is not None, "chain the original pickling error"
    with pytest.raises(TypeError):
        ex.run_cpu_bound(local_square, [1, 2])
