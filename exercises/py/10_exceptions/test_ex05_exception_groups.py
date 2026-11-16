import pytest

from study.loader import load

ex = load(__file__, "ex05_exception_groups")


def ok(v):
    return lambda: v


def boom(exc):
    def task():
        raise exc

    return task


@pytest.mark.topic("py.exceptions.multiple")
def test_run_all_success():
    assert ex.run_all({"a": ok(1), "b": ok(2)}) == {"a": 1, "b": 2}
    assert ex.run_all({}) == {}


@pytest.mark.topic("py.exceptions.multiple", "py.exceptions.contextual")
def test_run_all_groups_every_failure():
    calls = []

    def tracked(name, exc=None):
        def task():
            calls.append(name)
            if exc:
                raise exc
            return name

        return task

    e1, e2 = TimeoutError("slow"), ValueError("empty")
    tasks = {
        "b1": tracked("b1", e1),
        "b2": tracked("b2"),
        "b3": tracked("b3", e2),
    }
    with pytest.raises(ExceptionGroup) as info:
        ex.run_all(tasks)
    eg = info.value
    assert calls == ["b1", "b2", "b3"], "keep running after a failure"
    assert eg.message == "2 of 3 tasks failed"
    assert list(eg.exceptions) == [e1, e2]
    assert e1.__notes__ == ["task=b1"]
    assert e2.__notes__ == ["task=b3"]


@pytest.mark.topic("py.exceptions.hierarchy")
def test_run_all_base_exception_stops_immediately():
    calls = []

    def second():
        calls.append("second")

    with pytest.raises(KeyboardInterrupt):
        ex.run_all({"first": boom(KeyboardInterrupt()), "second": second})
    assert calls == []


@pytest.mark.topic("py.exceptions.contextual")
def test_task_name():
    e = ValueError("x")
    assert ex.task_name(e) is None
    e.add_note("something else")
    e.add_note("task=chunk-7")
    assert ex.task_name(e) == "chunk-7"


@pytest.mark.topic("py.exceptions.multiple")
def test_flatten_nested():
    a, b, c, d = ValueError("a"), KeyError("b"), TypeError("c"), OSError("d")
    eg = ExceptionGroup("outer", [a, ExceptionGroup("inner", [b, ExceptionGroup("deep", [c])]), d])
    assert ex.flatten(eg) == [a, b, c, d]
    assert ex.flatten(a) == [a]


def _group(**errors):
    excs = []
    for name, e in errors.items():
        e.add_note(f"task={name}")
        excs.append(e)
    return ExceptionGroup("batch failed", excs)


@pytest.mark.topic("py.exceptions.multiple", "py.exceptions.best_practices")
def test_triage_sorts_both_kinds_in_one_go():
    eg = _group(
        a=ex.TransientError("429"),
        b=ValueError("empty"),
        c=ex.TransientError("503"),
    )
    t = ex.triage(boom(eg))
    assert t == ex.Triage(retry=["a", "c"], invalid=["b"])


@pytest.mark.topic("py.exceptions.multiple")
def test_triage_naked_exception_and_no_error():
    e = ex.TransientError("429")
    e.add_note("task=solo")
    assert ex.triage(boom(e)) == ex.Triage(retry=["solo"])
    assert ex.triage(ok(None)) == ex.Triage()


@pytest.mark.topic("py.exceptions.multiple", "py.exceptions.hierarchy")
def test_triage_unmatched_remainder_propagates_as_group():
    auth = PermissionError("401")
    eg = _group(a=ex.TransientError("429"), b=auth)
    with pytest.raises(ExceptionGroup) as info:
        ex.triage(boom(eg))
    assert ex.flatten(info.value) == [auth]


@pytest.mark.topic("py.exceptions.multiple")
def test_triage_nested_groups():
    inner = _group(x=ValueError("bad"), y=ex.TransientError("slow"))
    outer = ExceptionGroup("outer", [inner, _group(z=ex.TransientError("slow"))])
    assert ex.triage(boom(outer)) == ex.Triage(retry=["y", "z"], invalid=["x"])


@pytest.mark.topic("py.exceptions.multiple")
def test_run_all_then_triage_end_to_end():
    tasks = {"ok": ok(1), "rl": boom(ex.TransientError("429")), "bad": boom(ValueError("x"))}
    assert ex.triage(lambda: ex.run_all(tasks)) == ex.Triage(retry=["rl"], invalid=["bad"])
