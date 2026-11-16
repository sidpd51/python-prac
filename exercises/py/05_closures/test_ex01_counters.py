import pytest

from study.loader import load

ex = load(__file__, "ex01_counters")


def _cells(fn):
    return {
        name: cell.cell_contents
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True)
    }


@pytest.mark.topic("py.closures.closures", "py.closures.nested_functions")
def test_counter_sequence():
    next_id = ex.make_counter(100, 10)
    assert [next_id(), next_id(), next_id()] == [100, 110, 120]


@pytest.mark.topic("py.closures.closures")
def test_counter_defaults_and_negative_step():
    c = ex.make_counter()
    assert [c(), c(), c()] == [0, 1, 2]
    down = ex.make_counter(3, -1)
    assert [down(), down(), down(), down()] == [3, 2, 1, 0]


@pytest.mark.topic("py.closures.closures", "py.closures.free_variables")
def test_counters_are_independent():
    a = ex.make_counter()
    b = ex.make_counter()
    a()
    a()
    assert b() == 0
    assert a() == 2
    assert a.__closure__ is not None, "make_counter must return a closure"


@pytest.mark.topic("py.closures.closures")
def test_counter_validates_step_eagerly():
    with pytest.raises(ValueError):
        ex.make_counter(0, 0)


@pytest.mark.topic("py.closures.nonlocal")
def test_averager_running_mean():
    avg = ex.make_averager()
    assert avg(10) == 10.0
    assert avg(20) == 15.0
    assert avg(30) == 20.0
    other = ex.make_averager()
    assert other(1) == 1.0


@pytest.mark.topic("py.closures.nonlocal", "py.closures.inspection")
def test_averager_state_lives_in_cells_not_a_list():
    avg = ex.make_averager()
    avg(2)
    avg(4)
    cells = _cells(avg)
    assert {"total", "count"} <= set(cells), "use free variables named total and count"
    assert cells["count"] == 2
    assert cells["total"] == 6
    assert not any(isinstance(v, (list, dict)) for v in cells.values()), "use nonlocal"
