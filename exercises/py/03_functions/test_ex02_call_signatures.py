import pytest

from study.loader import load

ex = load(__file__, "ex02_call_signatures")

MSGS = [{"role": "user", "content": "hi"}]


@pytest.mark.topic("py.functions.positional_args", "py.functions.keyword_args")
def test_build_request_basic():
    assert ex.build_request("small", MSGS) == {"model": "small", "messages": MSGS, "temperature": 0.0}
    out = ex.build_request("small", iter(MSGS), temperature=0.3, max_tokens=256)
    assert out == {"model": "small", "messages": MSGS, "temperature": 0.3, "max_tokens": 256}


@pytest.mark.topic("py.functions.kwargs")
def test_build_request_extra_options():
    out = ex.build_request("m", MSGS, top_p=0.9, stop=["\n"], seed=None)
    assert out == {"model": "m", "messages": MSGS, "temperature": 0.0, "top_p": 0.9, "stop": ["\n"]}


@pytest.mark.topic("py.functions.pos_kw_only", "py.functions.kwargs")
def test_build_request_signature_rules():
    assert ex.build_request("m", MSGS)["model"] == "m"
    with pytest.raises(TypeError):
        ex.build_request("m", MSGS, 0.5)  # temperature is keyword-only
    with pytest.raises(TypeError):
        ex.build_request("m", MSGS, model="other")  # would override via **extra
    with pytest.raises(TypeError):
        ex.build_request("m", MSGS, messages=[])


@pytest.mark.topic("py.functions.args", "py.functions.kwargs", "py.functions.unpacking")
def test_call_with_retry_forwards_args_and_retries():
    calls = []

    def flaky(a, b, *, scale=1):
        calls.append((a, b, scale))
        if len(calls) < 3:
            raise TimeoutError("slow upstream")
        return (a + b) * scale

    assert ex.call_with_retry(flaky, 1, 2, scale=10) == 30
    assert calls == [(1, 2, 10)] * 3


@pytest.mark.topic("py.functions.args")
def test_call_with_retry_gives_up_and_reraises_last():
    errors = iter([ConnectionError("1"), ConnectionError("2"), ConnectionError("3"), ConnectionError("4")])

    def down():
        raise next(errors)

    with pytest.raises(ConnectionError, match="3"):
        ex.call_with_retry(down)  # 1 try + 2 retries


@pytest.mark.topic("py.functions.keyword_args")
def test_call_with_retry_only_retries_listed_errors():
    calls = []

    def bad_request():
        calls.append(1)
        raise ValueError("invalid prompt")

    with pytest.raises(ValueError):
        ex.call_with_retry(bad_request, retries=5)
    assert len(calls) == 1

    custom = []

    def rate_limited():
        custom.append(1)
        if len(custom) == 1:
            raise KeyError("429")
        return "ok"

    assert ex.call_with_retry(rate_limited, retry_on=(KeyError,), retries=1) == "ok"


@pytest.mark.topic("py.functions.keyword_args")
def test_call_with_retry_zero_and_negative_retries():
    calls = []

    def once():
        calls.append(1)
        raise TimeoutError

    with pytest.raises(TimeoutError):
        ex.call_with_retry(once, retries=0)
    assert len(calls) == 1
    with pytest.raises(ValueError):
        ex.call_with_retry(once, retries=-1)
    assert len(calls) == 1, "fn must not be called when retries is invalid"


@pytest.mark.topic("py.functions.pos_kw_only")
def test_call_with_retry_fn_name_does_not_collide():
    def apply(fn, value):
        return fn(value)

    assert ex.call_with_retry(apply, fn=str.upper, value="ok") == "OK"
    assert ex.call_with_retry(apply, str.upper, "x") == "X"


@pytest.mark.topic("py.functions.unpacking", "py.functions.args", "py.functions.kwargs")
def test_bind():
    def f(*args, **kwargs):
        return args, kwargs

    g = ex.bind(f, 1, x=2)
    assert g(3, y=4) == ((1, 3), {"x": 2, "y": 4})
    assert g() == ((1,), {"x": 2})
    assert g(x=99) == ((1,), {"x": 99})  # call-time kwargs win
    assert ex.bind(f)() == ((), {})


@pytest.mark.topic("py.functions.kwargs")
def test_bind_captures_at_bind_time():
    def f(**kwargs):
        return kwargs

    opts = {"temperature": 0.1}
    g = ex.bind(f, **opts)
    opts["temperature"] = 0.9
    assert g() == {"temperature": 0.1}
    result = g()
    result["mutated"] = True
    assert g() == {"temperature": 0.1}

    def h(fn):
        return fn

    assert ex.bind(h, fn=1)() == 1  # bind's own fn is positional-only too
