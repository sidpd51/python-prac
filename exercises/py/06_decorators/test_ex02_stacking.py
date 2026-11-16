import pytest

from study.loader import load

ex = load(__file__, "ex02_stacking")


@pytest.mark.topic("py.decorators.stacking", "py.decorators.syntax")
def test_stacking_order_definition_and_call():
    log = []

    @ex.trace(log, "auth")
    @ex.trace(log, "cache")
    def answer(q):
        log.append("body")
        return q.upper()

    assert log == ["cache:decorate", "auth:decorate"]
    log.clear()
    assert answer("hi") == "HI"
    assert log == ["auth:before", "cache:before", "body", "cache:after", "auth:after"]
    assert answer.__name__ == "answer"


@pytest.mark.topic("py.decorators.stacking")
def test_after_not_logged_on_exception():
    log = []

    @ex.trace(log, "outer")
    def boom():
        raise RuntimeError("x")

    log.clear()
    with pytest.raises(RuntimeError):
        boom()
    assert log == ["outer:before"]


@pytest.mark.topic("py.decorators.stacking", "py.decorators.higher_order")
def test_apply_decorators_matches_at_syntax():
    log_a, log_b = [], []

    @ex.trace(log_a, "d1")
    @ex.trace(log_a, "d2")
    @ex.trace(log_a, "d3")
    def f(x):
        return x + 1

    def g(x):
        return x + 1

    g2 = ex.apply_decorators(g, ex.trace(log_b, "d1"), ex.trace(log_b, "d2"), ex.trace(log_b, "d3"))
    assert log_a == log_b == ["d3:decorate", "d2:decorate", "d1:decorate"]
    log_a.clear()
    log_b.clear()
    assert f(1) == g2(1) == 2
    assert log_a == log_b


@pytest.mark.topic("py.decorators.first_class")
def test_apply_no_decorators_returns_same_object():
    def f():
        return 1

    assert ex.apply_decorators(f) is f


@pytest.mark.topic("py.decorators.first_class", "py.decorators.higher_order")
def test_compose_right_to_left():
    def strip(s):
        return s.strip()

    def lower(s):
        return s.lower()

    def tokens(s):
        return s.split()

    normalize = ex.compose(tokens, lower, strip)
    assert normalize("  Hello World  ") == ["hello", "world"]
    assert ex.compose(len)("abc") == 3


@pytest.mark.topic("py.decorators.higher_order")
def test_compose_innermost_gets_all_args():
    def build(system, user, *, sep="\n"):
        return f"{system}{sep}{user}"

    count_lines = ex.compose(len, str.splitlines, build)
    assert count_lines("sys", "usr") == 2
    assert count_lines("sys", "usr", sep=" ") == 1
    with pytest.raises(TypeError):
        ex.compose()
