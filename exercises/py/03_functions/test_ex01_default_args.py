import time

import pytest

from study.loader import load

ex = load(__file__, "ex01_default_args")


@pytest.mark.topic("py.functions.default_args")
def test_new_conversation_calls_are_independent():
    a = ex.new_conversation("You are terse.")
    b = ex.new_conversation("You are verbose.")
    assert a == [{"role": "system", "content": "You are terse."}]
    assert b == [{"role": "system", "content": "You are verbose."}]
    a.append({"role": "user", "content": "hi"})
    assert ex.new_conversation("x") == [{"role": "system", "content": "x"}]


@pytest.mark.topic("py.functions.default_args", "py.functional.side_effects")
def test_new_conversation_does_not_mutate_input():
    history = [{"role": "user", "content": "hi"}]
    out = ex.new_conversation("sys", history)
    assert out == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert history == [{"role": "user", "content": "hi"}]
    assert out is not history
    assert ex.new_conversation("sys", []) == [{"role": "system", "content": "sys"}]


@pytest.mark.topic("py.functions.default_args", "py.functions.keyword_args")
def test_make_event_uses_clock_at_call_time():
    ticks = iter([100.0, 200.0])
    e1 = ex.make_event("start", clock=lambda: next(ticks))
    e2 = ex.make_event("stop", {"ok": True}, clock=lambda: next(ticks))
    assert e1 == {"name": "start", "payload": {}, "created_at": 100.0}
    assert e2 == {"name": "stop", "payload": {"ok": True}, "created_at": 200.0}


@pytest.mark.topic("py.functions.default_args")
def test_make_event_default_clock_is_not_frozen():
    e1 = ex.make_event("a")
    time.sleep(0.02)
    e2 = ex.make_event("b")
    assert e2["created_at"] > e1["created_at"]
    assert abs(e2["created_at"] - time.time()) < 5


@pytest.mark.topic("py.functions.default_args", "py.fundamentals.mutability")
def test_make_event_payloads_are_never_shared():
    payload = {"user": "u1"}
    e = ex.make_event("login", payload, clock=lambda: 0.0)
    assert payload == {"user": "u1"}, "caller's payload was mutated"
    e["payload"]["extra"] = 1
    assert payload == {"user": "u1"}
    d1 = ex.make_event("x", clock=lambda: 0.0)
    d1["payload"]["leak"] = True
    assert ex.make_event("y", clock=lambda: 0.0)["payload"] == {}


@pytest.mark.topic("py.functions.keyword_args", "py.functions.pos_kw_only")
def test_make_event_clock_is_keyword_only():
    assert ex.make_event("a", None, clock=lambda: 1.0)["created_at"] == 1.0
    with pytest.raises(TypeError):
        ex.make_event("a", None, lambda: 1.0)


@pytest.mark.topic("py.functions.return_values", "py.functions.unpacking")
@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("anthropic/claude-x:2024-10", ("anthropic", "claude-x", "2024-10")),
        ("openai/gpt-4o", ("openai", "gpt-4o", None)),
        ("local/llama:8b:q4", ("local", "llama", "8b:q4")),
        ("hf/org/model", ("hf", "org/model", None)),
    ],
)
def test_parse_model_id_returns_tuple(model_id, expected):
    out = ex.parse_model_id(model_id)
    assert out == expected
    provider, name, version = out  # unpackable
    assert isinstance(out, tuple)


@pytest.mark.topic("py.functions.return_values")
@pytest.mark.parametrize("bad", ["gpt-4o", "/gpt", "openai/", "openai/:v1", ""])
def test_parse_model_id_invalid(bad):
    with pytest.raises(ValueError):
        ex.parse_model_id(bad)
