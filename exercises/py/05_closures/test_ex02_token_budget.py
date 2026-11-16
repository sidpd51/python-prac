import pytest

from study.loader import load

ex = load(__file__, "ex02_token_budget")


@pytest.mark.topic("py.closures.closures", "py.closures.nonlocal")
def test_track_returns_remaining():
    t = ex.make_usage_tracker(1000)
    assert t.track(100, 50) == 850
    assert t.track(300, 0) == 550
    assert t.remaining() == 550


@pytest.mark.topic("py.closures.free_variables")
def test_sibling_closures_share_state():
    t = ex.make_usage_tracker(500)
    t.track(10, 20)
    t.track(5, 5)
    assert t.report() == {
        "calls": 2,
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "remaining": 460,
    }
    for fn in (t.track, t.remaining, t.report):
        assert fn.__closure__, "each function must be a closure over the shared state"


@pytest.mark.topic("py.closures.closures")
def test_trackers_are_independent():
    a = ex.make_usage_tracker(100)
    b = ex.make_usage_tracker(100)
    a.track(60, 0)
    assert b.remaining() == 100
    assert a.remaining() == 40


@pytest.mark.topic("py.closures.nonlocal")
def test_exact_budget_allowed_then_exceeded_rejected_without_mutation():
    t = ex.make_usage_tracker(100)
    assert t.track(70, 30) == 0
    with pytest.raises(ex.BudgetExceeded):
        t.track(1, 0)
    assert t.report() == {"calls": 1, "prompt_tokens": 70, "completion_tokens": 30, "remaining": 0}


@pytest.mark.topic("py.closures.nonlocal")
def test_rejected_call_mid_session_keeps_state():
    t = ex.make_usage_tracker(100)
    t.track(50, 0)
    with pytest.raises(ex.BudgetExceeded):
        t.track(40, 20)
    assert t.remaining() == 50
    assert t.track(40, 10) == 0


@pytest.mark.topic("py.closures.closures")
def test_validation():
    with pytest.raises(ValueError):
        ex.make_usage_tracker(-1)
    t = ex.make_usage_tracker(10)
    with pytest.raises(ValueError):
        t.track(-1, 0)
    with pytest.raises(ValueError):
        t.track(0, -5)
    assert t.report()["calls"] == 0


@pytest.mark.topic("py.closures.closures")
def test_report_is_a_fresh_copy():
    t = ex.make_usage_tracker(10)
    r = t.report()
    r["remaining"] = 999
    assert t.report()["remaining"] == 10
