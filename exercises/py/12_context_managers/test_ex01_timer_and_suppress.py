import pytest

from study.loader import load

ex = load(__file__, "ex01_timer_and_suppress")


def fake_clock(*ticks):
    it = iter(ticks)
    return lambda: next(it)


@pytest.mark.topic("py.context_managers.protocol", "py.context_managers.enter")
def test_timer_measures_and_binds_self():
    t = ex.Timer("embed", clock=fake_clock(10.0, 12.5))
    assert t.elapsed is None
    with t as bound:
        assert bound is t
        assert t.elapsed is None
    assert t.elapsed == 2.5


@pytest.mark.topic("py.context_managers.exit", "py.context_managers.custom")
def test_timer_reports_on_exception_and_does_not_swallow():
    seen = []
    t = ex.Timer("llm-call", clock=fake_clock(1.0, 4.0), on_exit=lambda n, e: seen.append((n, e)))
    with pytest.raises(TimeoutError), t:
        raise TimeoutError("model too slow")
    assert seen == [("llm-call", 3.0)]
    assert t.elapsed == 3.0


@pytest.mark.topic("py.context_managers.custom")
def test_timer_is_reusable_and_defaults_work():
    t = ex.Timer(clock=fake_clock(0.0, 1.0, 5.0, 5.5))
    with t:
        pass
    assert t.elapsed == 1.0
    with t:
        assert t.elapsed is None, "reset on re-entry"
    assert t.elapsed == 0.5
    with ex.Timer() as real:  # default perf_counter
        sum(range(1000))
    assert real.name == "block"
    assert real.elapsed >= 0


@pytest.mark.topic("py.context_managers.exit")
def test_suppress_swallows_listed_types():
    with ex.suppress_errors(KeyError, ValueError) as s:
        {}["missing"]
    assert isinstance(s.error, KeyError)
    with ex.suppress_errors(LookupError) as s:
        [][1]  # IndexError is a LookupError
    assert isinstance(s.error, IndexError)


@pytest.mark.topic("py.context_managers.exit")
def test_suppress_lets_others_through():
    s = ex.suppress_errors(KeyError)
    with pytest.raises(ZeroDivisionError), s:
        _ = 1 / 0
    assert s.error is None
    with s:
        pass
    assert s.error is None


@pytest.mark.topic("py.context_managers.exit", "py.context_managers.protocol")
def test_suppress_exit_return_values():
    s = ex.suppress_errors(KeyError)
    s.__enter__()
    assert not s.__exit__(None, None, None)
    err = KeyError("k")
    assert s.__exit__(KeyError, err, None) is True
    assert s.error is err
    assert not s.__exit__(ValueError, ValueError("v"), None)


@pytest.mark.topic("py.context_managers.custom")
def test_suppress_resets_between_uses_and_code_after_error_is_skipped():
    s = ex.suppress_errors(ValueError)
    ran = []
    with s:
        int("x")
        ran.append("after")  # never runs: the block is exited at the raise
    assert ran == []
    with s:
        ran.append("ok")
    assert s.error is None
    assert ran == ["ok"]
