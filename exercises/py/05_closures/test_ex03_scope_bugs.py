import inspect

import pytest

from study.loader import load

ex = load(__file__, "ex03_scope_bugs")


@pytest.fixture
def fresh_counters():
    ex.REQUEST_COUNTS.clear()
    ex.REQUEST_TOTAL = 0
    yield
    ex.REQUEST_COUNTS.clear()
    ex.REQUEST_TOTAL = 0


@pytest.mark.topic("py.closures.local_global", "py.closures.unboundlocalerror")
def test_record_request_updates_globals(fresh_counters):
    assert ex.record_request("a") == 1
    assert ex.record_request("b") == 2
    assert ex.record_request("a") == 3
    assert ex.REQUEST_TOTAL == 3
    assert ex.REQUEST_COUNTS == {"a": 2, "b": 1}


@pytest.mark.topic("py.closures.unboundlocalerror", "py.closures.local_global")
def test_pick_model_short_and_long():
    assert ex.pick_model(100) == ex.DEFAULT_MODEL
    assert ex.pick_model(ex.LONG_CONTEXT_THRESHOLD) == ex.DEFAULT_MODEL
    assert ex.pick_model(ex.LONG_CONTEXT_THRESHOLD + 1) == ex.LONG_CONTEXT_MODEL


@pytest.mark.topic("py.closures.local_global")
def test_pick_model_does_not_touch_globals():
    before = ex.DEFAULT_MODEL
    ex.pick_model(10**6)
    assert ex.pick_model(1) == before == ex.DEFAULT_MODEL


@pytest.mark.topic("py.closures.closures", "py.closures.lexical_scoping")
def test_formatters_bind_their_own_template():
    fs = ex.make_formatters(["Q: {text}", "Summarize: {text}", "{text}!"])
    assert [f("hi") for f in fs] == ["Q: hi", "Summarize: hi", "hi!"]


@pytest.mark.topic("py.closures.closures", "py.closures.free_variables")
def test_formatters_take_one_param_and_are_closures():
    fs = ex.make_formatters(["<{text}>"])
    assert len(inspect.signature(fs[0]).parameters) == 1, "no default-argument trick"
    assert fs[0].__closure__, "the template should be captured in a closure cell"
    assert ex.make_formatters([]) == []
