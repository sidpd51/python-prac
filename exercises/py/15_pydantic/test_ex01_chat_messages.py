import json

import pytest
from pydantic import ValidationError

from study.loader import load

ex = load(__file__, "ex01_chat_messages")


def _call(call_id="call_1", name="search", args='{"q": "pydantic"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


@pytest.mark.topic("py.pydantic.custom_validators", "py.pydantic.validation")
@pytest.mark.parametrize("raw", ["user", " User ", "USER\n"])
def test_role_is_normalised_before_literal_check(raw):
    assert ex.Message(role=raw, content="hi").role == "user"


@pytest.mark.topic("py.pydantic.custom_validators", "py.pydantic.validation")
@pytest.mark.parametrize("raw", ["robot", 3, None])
def test_bad_roles_still_rejected(raw):
    with pytest.raises(ValidationError) as info:
        ex.Message(role=raw, content="hi")
    assert info.value.errors()[0]["loc"] == ("role",)


@pytest.mark.topic("py.pydantic.custom_validators")
def test_content_is_stripped_and_blank_becomes_none():
    assert ex.Message(role="user", content="  hello \n").content == "hello"
    msg = ex.Message(role="assistant", content="   ", tool_calls=[_call()])
    assert msg.content is None


@pytest.mark.topic("py.pydantic.required_optional", "py.pydantic.validation")
def test_role_is_required():
    with pytest.raises(ValidationError) as info:
        ex.Message(content="hi")
    assert [e["type"] for e in info.value.errors()] == ["missing"]


@pytest.mark.topic("py.pydantic.model_validator")
@pytest.mark.parametrize(
    "payload",
    [
        {"role": "tool", "content": "42"},  # no tool_call_id
        {"role": "tool", "tool_call_id": "c1", "content": " "},  # blank content -> None
        {"role": "user", "content": "x", "tool_calls": [_call()]},
        {"role": "user", "content": "x", "tool_call_id": "c1"},
        {"role": "assistant"},
        {"role": "assistant", "content": "  "},
        {"role": "system", "content": None},
        {"role": "user"},
    ],
)
def test_cross_field_rules(payload):
    with pytest.raises(ValidationError):
        ex.Message.model_validate(payload)


@pytest.mark.topic("py.pydantic.model_validator", "py.pydantic.nested", "py.pydantic.basemodel")
def test_valid_messages_and_nested_tool_calls():
    msg = ex.Message.model_validate({"role": "assistant", "tool_calls": [_call()]})
    assert isinstance(msg.tool_calls[0], ex.ToolCall)
    assert isinstance(msg.tool_calls[0].function, ex.FunctionCall)
    assert msg.tool_calls[0].function.name == "search"
    tool = ex.Message(role="tool", tool_call_id="call_1", content="results...")
    assert tool.content == "results..."


@pytest.mark.topic("py.pydantic.nested", "py.pydantic.validation")
def test_nested_error_location():
    bad = {"role": "assistant", "tool_calls": [_call(), {"id": "c2", "function": {"name": "x"}}]}
    with pytest.raises(ValidationError) as info:
        ex.Message.model_validate(bad)
    locs = [e["loc"] for e in info.value.errors()]
    assert ("tool_calls", 1, "function", "arguments") in locs


@pytest.mark.topic("py.pydantic.models")
def test_parsed_arguments():
    tc = ex.ToolCall.model_validate(_call(args='{"q": "rag", "k": 3}'))
    assert tc.parsed_arguments() == {"q": "rag", "k": 3}
    with pytest.raises(ValueError):
        ex.ToolCall.model_validate(_call(args="[1, 2]")).parsed_arguments()
    with pytest.raises(json.JSONDecodeError):
        ex.ToolCall.model_validate(_call(args="{q: rag")).parsed_arguments()


def _conv(*msgs):
    return ex.Conversation.model_validate({"messages": list(msgs)})


SYS = {"role": "system", "content": "be terse"}
USER = {"role": "user", "content": "weather in Paris?"}
ASSIST_CALL = {"role": "assistant", "tool_calls": [_call("call_9", "weather", '{"city": "Paris"}')]}
TOOL = {"role": "tool", "tool_call_id": "call_9", "content": "18C"}
ASSIST = {"role": "assistant", "content": "18C and sunny."}


@pytest.mark.topic("py.pydantic.model_validator", "py.pydantic.nested")
def test_conversation_valid_and_last_text():
    conv = _conv(SYS, USER, ASSIST_CALL, TOOL, ASSIST)
    assert len(conv.messages) == 5
    assert conv.last_assistant_text() == "18C and sunny."
    assert _conv(USER, ASSIST_CALL).last_assistant_text() is None


@pytest.mark.topic("py.pydantic.model_validator")
@pytest.mark.parametrize(
    "msgs",
    [
        [USER, SYS],  # system not first
        [USER, TOOL],  # tool answer without a prior call
        [USER, TOOL, ASSIST_CALL],  # call issued AFTER the tool message
    ],
)
def test_conversation_order_rules(msgs):
    with pytest.raises(ValidationError):
        _conv(*msgs)


@pytest.mark.topic("py.pydantic.model_validator")
def test_empty_conversation_rejected():
    with pytest.raises(ValidationError):
        ex.Conversation(messages=[])
