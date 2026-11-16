import inspect

import pytest

from study.loader import load

ex = load(__file__, "ex03_coroutine_averager")


@pytest.mark.topic("py.generators.send", "py.generators.resumption")
def test_send_running_average():
    avg = ex.averager()
    assert avg.send(10) == 10.0
    assert avg.send(20) == 15.0
    assert avg.send(30) == 20.0


@pytest.mark.topic("py.generators.send")
def test_send_none_is_ignored():
    avg = ex.averager()
    avg.send(4)
    assert avg.send(None) == 4.0
    assert avg.send(8) == 6.0


@pytest.mark.topic("py.generators.throw")
def test_throw_resets():
    avg = ex.averager()
    avg.send(100)
    assert avg.throw(ex.Reset) == 0.0
    assert avg.send(2) == 2.0


@pytest.mark.topic("py.generators.close")
def test_close_records_summary():
    ex.CLOSED_SUMMARIES.clear()
    avg = ex.averager()
    avg.send(1)
    avg.send(3)
    avg.close()
    assert ex.CLOSED_SUMMARIES == [(2, 2.0)]
    assert inspect.getgeneratorstate(avg) == inspect.GEN_CLOSED
    with pytest.raises(StopIteration):
        avg.send(1)


@pytest.mark.topic("py.generators.suspension", "py.decorators.wraps")
def test_primed_decorator():
    assert ex.averager.__name__ == "averager"
    assert inspect.getgeneratorstate(ex.averager()) == inspect.GEN_SUSPENDED

    @ex.primed
    def echo():
        received = None
        while True:
            received = yield received

    e = echo()
    assert e.send("hi") == "hi"
