from typing import Literal, NotRequired, get_args, get_origin, get_type_hints

import pytest

from study.loader import load

ex = load(__file__, "ex01_chat_messages")


@pytest.mark.topic("py.type_hints.literal", "py.type_hints.functions")
def test_make_message():
    assert ex.make_message("user", "hi") == {"role": "user", "content": "hi"}
    msg = ex.make_message("tool", "42", name="calculator")
    assert msg == {"role": "tool", "content": "42", "name": "calculator"}
    assert type(msg) is dict, "a TypedDict instance is just a dict at runtime"


@pytest.mark.topic("py.type_hints.literal", "py.type_hints.basic")
def test_make_message_validates_at_runtime():
    with pytest.raises(ValueError):
        ex.make_message("robot", "hi")  # annotations alone would let this through
    with pytest.raises(TypeError):
        ex.make_message("user", 123)
    # the valid roles come from the Literal itself
    assert get_args(ex.Role) == ("system", "user", "assistant", "tool")
    assert get_origin(ex.Role) is Literal


@pytest.mark.topic("py.type_hints.typeddict", "py.type_hints.optional")
def test_typeddict_shape():
    ex.make_message("user", "x")
    assert ex.ChatMessage.__required_keys__ == {"role", "content"}
    assert ex.ChatMessage.__optional_keys__ == {"name"}
    hints = get_type_hints(ex.ChatMessage, include_extras=True)
    assert hints["content"] is str
    assert hints["name"] == NotRequired[str]


@pytest.mark.topic("py.type_hints.typeddict", "py.type_hints.collections")
def test_validate_messages_ok_and_copies():
    raw = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi", "name": "sid"},
    ]
    out = ex.validate_messages(raw)
    assert out == raw
    out[0]["content"] = "changed"
    assert raw[0]["content"] == "You are terse.", "return new dicts"


@pytest.mark.topic("py.type_hints.typeddict")
@pytest.mark.parametrize(
    ("bad", "index"),
    [
        ([{"role": "user"}], 0),
        ([{"role": "user", "content": "a"}, {"content": "b"}], 1),
        ([{"role": "user", "content": "a", "extra": 1}], 0),
        ([{"role": "admin", "content": "a"}], 0),
        ([{"role": "user", "content": ["a"]}], 0),
        ([{"role": "user", "content": "a", "name": None}], 0),
        ([{"role": "user", "content": "a"}, "not a dict"], 1),
    ],
)
def test_validate_messages_errors(bad, index):
    with pytest.raises(ValueError, match=rf"^message {index}: "):
        ex.validate_messages(bad)


@pytest.mark.topic("py.type_hints.collections", "py.type_hints.returns")
def test_tokens_by_role():
    msgs = [
        ex.make_message("system", "be nice"),
        ex.make_message("user", "hello there"),
        ex.make_message("user", "again"),
    ]
    assert ex.tokens_by_role(msgs) == {"system": 7, "user": 16}
    assert ex.tokens_by_role(msgs, count=lambda s: len(s.split())) == {"system": 2, "user": 3}
    assert ex.tokens_by_role([]) == {}


@pytest.mark.topic("py.type_hints.optional", "py.type_hints.returns")
def test_last_message():
    msgs = [
        ex.make_message("user", "q1"),
        ex.make_message("assistant", "a1"),
        ex.make_message("user", "q2"),
    ]
    assert ex.last_message(msgs) == {"role": "user", "content": "q2"}
    assert ex.last_message(msgs, "assistant")["content"] == "a1"
    assert ex.last_message(msgs, "tool") is None
    assert ex.last_message([]) is None


@pytest.mark.topic("py.type_hints.functions", "py.type_hints.returns")
def test_signatures_are_annotated():
    ex.make_message("user", "x")
    hints = get_type_hints(ex.last_message)
    assert hints["return"] == ex.ChatMessage | None
    assert get_type_hints(ex.tokens_by_role)["return"] == dict[ex.Role, int]
