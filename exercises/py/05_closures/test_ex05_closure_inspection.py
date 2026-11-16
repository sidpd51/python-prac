import pytest

from study.loader import load

ex = load(__file__, "ex05_closure_inspection")


def make_callback(model, temperature):
    retries = 3

    def on_call(prompt):
        return f"{model}@{temperature}:{prompt} (retries={retries})"

    return on_call


def make_pair():
    value = 0

    def get():
        return value

    def set_(v):
        nonlocal value
        value = v

    return get, set_


def make_with_deleted():
    secret = "token"
    other = 1

    def f():
        return secret, other  # noqa: F821 - `secret` is deleted on purpose

    del secret
    return f


def plain(x):
    return x


@pytest.mark.topic("py.closures.inspection", "py.closures.co_freevars")
def test_closure_vars_reads_cells():
    cb = make_callback("gpt-x", 0.2)
    assert ex.closure_vars(cb) == {"model": "gpt-x", "retries": 3, "temperature": 0.2}
    assert list(ex.closure_vars(cb)) == list(cb.__code__.co_freevars)


@pytest.mark.topic("py.closures.inspection")
def test_closure_vars_plain_function_and_empty_cell():
    assert ex.closure_vars(plain) == {}
    assert ex.closure_vars(make_with_deleted()) == {"other": 1}


@pytest.mark.topic("py.closures.inspection", "py.closures.free_variables")
def test_closure_vars_sees_live_value():
    get, set_ = make_pair()
    set_(42)
    assert ex.closure_vars(get) == {"value": 42}


@pytest.mark.topic("py.closures.inspection", "py.closures.lexical_scoping")
def test_set_closure_var_rebinds_shared_cell():
    get, set_ = make_pair()
    ex.set_closure_var(get, "value", "patched")
    assert get() == "patched"
    cb = make_callback("m", 0.0)
    ex.set_closure_var(cb, "model", "mock-llm")
    assert cb("hi") == "mock-llm@0.0:hi (retries=3)"


@pytest.mark.topic("py.closures.co_freevars")
def test_set_closure_var_unknown_name():
    cb = make_callback("m", 0.0)
    with pytest.raises(KeyError, match="prompt"):
        ex.set_closure_var(cb, "prompt", "x")  # a parameter, not a free variable
    with pytest.raises(KeyError, match="x"):
        ex.set_closure_var(plain, "x", 1)


@pytest.mark.topic("py.closures.free_variables", "py.closures.inspection")
def test_shares_state():
    get1, set1 = make_pair()
    get2, set2 = make_pair()
    assert ex.shares_state(get1, set1) is True
    assert ex.shares_state(get1, get2) is False
    a = make_callback("same", 1)
    b = make_callback("same", 1)
    assert ex.shares_state(a, b) is False, "equal values in different cells are not shared"
    assert ex.shares_state(plain, get1) is False
    assert ex.shares_state(plain, plain) is False
