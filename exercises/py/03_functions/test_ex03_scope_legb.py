import pytest

from study.loader import load

ex = load(__file__, "ex03_scope_legb")


@pytest.mark.topic("py.functions.legb", "py.closures.nonlocal")
def test_meter_accumulates():
    record, total = ex.make_meter()
    assert total() == 0
    assert record(120) == 120
    assert record(30) == 150
    assert total() == 150


@pytest.mark.topic("py.functions.legb", "py.closures.closures")
def test_meters_are_independent_and_not_global():
    r1, t1 = ex.make_meter()
    r2, t2 = ex.make_meter()
    r1(10)
    r2(1)
    r1(5)
    assert (t1(), t2()) == (15, 1)
    module_ints = {k: v for k, v in vars(ex).items() if isinstance(v, int) and not k.startswith("__")}
    assert module_ints == {}, f"state leaked into module globals: {module_ints}"


@pytest.mark.topic("py.functions.legb", "py.closures.unboundlocalerror")
def test_counter():
    inc = ex.make_counter()
    assert [inc(), inc(), inc()] == [1, 2, 3]
    other = ex.make_counter(start=10)
    assert other() == 11
    assert inc() == 4


@pytest.mark.topic("py.functions.legb", "py.functions.first_class")
def test_formatters_bind_each_template():
    fmts = ex.make_formatters(["Q: {}", "A: {}", "[{}]"])
    assert [f("x") for f in fmts] == ["Q: x", "A: x", "[x]"]
    assert ex.make_formatters([]) == []


@pytest.mark.topic("py.functions.legb", "py.functions.return_values")
def test_resolve_order():
    local_ns = {"x": "L"}
    enclosing = [{"x": "E1", "y": "E1"}, {"y": "E2", "z": "E2"}]
    global_ns = {"x": "G", "z": "G", "g": "G", "len": "shadowed"}
    assert ex.resolve("x", local_ns, enclosing, global_ns) == ("local", "L")
    assert ex.resolve("y", local_ns, enclosing, global_ns) == ("enclosing", "E1")  # innermost
    assert ex.resolve("z", local_ns, enclosing, global_ns) == ("enclosing", "E2")
    assert ex.resolve("g", local_ns, enclosing, global_ns) == ("global", "G")
    assert ex.resolve("len", local_ns, enclosing, global_ns) == ("global", "shadowed")
    assert ex.resolve("len", {}, [], {}) == ("builtin", len)
    assert ex.resolve("ValueError", {}, [], {}) == ("builtin", ValueError)


@pytest.mark.topic("py.functions.legb")
def test_resolve_none_values_and_missing():
    assert ex.resolve("x", {"x": None}, [], {"x": 1}) == ("local", None)
    with pytest.raises(NameError, match="nope"):
        ex.resolve("nope", {}, [{}], {})
