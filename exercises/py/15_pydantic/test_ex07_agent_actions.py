import json
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from study.loader import load

ex = load(__file__, "ex07_agent_actions")

MODELS = (ex.SearchAction, ex.CalculatorAction, ex.FinalAnswer)


@pytest.fixture
def adapter():
    return ex.build_action_adapter(*MODELS)


@pytest.mark.topic("py.pydantic.validation", "py.pydantic.nested")
def test_dispatch_on_tag(adapter):
    a = adapter.validate_python({"type": "search", "query": "refunds"})
    assert isinstance(a, ex.SearchAction) and a.top_k == 5
    b = adapter.validate_json('{"type": "final_answer", "answer": "42", "citations": ["d1"]}')
    assert isinstance(b, ex.FinalAnswer) and b.citations == ["d1"]
    c = adapter.validate_python({"type": "calculator", "expression": "2+2"})
    assert isinstance(c, ex.CalculatorAction)


@pytest.mark.topic("py.pydantic.validation")
def test_discriminated_errors_are_precise(adapter):
    with pytest.raises(ValidationError) as info:
        adapter.validate_python({"type": "search", "query": "q", "top_k": 99})
    errors = info.value.errors()
    assert len(errors) == 1, "only the selected member should report errors"
    assert errors[0]["loc"] == ("search", "top_k")


@pytest.mark.topic("py.pydantic.validation")
@pytest.mark.parametrize(
    ("payload", "etype"),
    [({"type": "browse", "url": "x"}, "union_tag_invalid"), ({"query": "q"}, "union_tag_not_found")],
)
def test_bad_or_missing_tag(adapter, payload, etype):
    with pytest.raises(ValidationError) as info:
        adapter.validate_python(payload)
    assert info.value.errors()[0]["type"] == etype


@pytest.mark.topic("py.pydantic.models")
def test_build_adapter_argument_checks():
    class NoType(BaseModel):
        x: int

    class StrType(BaseModel):
        type: str

    class Search2(BaseModel):
        type: Literal["search"]

    with pytest.raises(ValueError):
        ex.build_action_adapter()
    with pytest.raises(TypeError):
        ex.build_action_adapter(ex.SearchAction, NoType)
    with pytest.raises(TypeError):
        ex.build_action_adapter(ex.SearchAction, StrType)
    with pytest.raises(ValueError):
        ex.build_action_adapter(ex.SearchAction, Search2)
    single = ex.build_action_adapter(ex.FinalAnswer)
    assert isinstance(single.validate_python({"type": "final_answer", "answer": "a"}), ex.FinalAnswer)


@pytest.mark.topic("py.pydantic.json_schema")
def test_schema_has_discriminator(adapter):
    schema = adapter.json_schema()
    assert schema["discriminator"]["propertyName"] == "type"
    assert len(schema["oneOf"]) == 3


@pytest.mark.topic("py.pydantic.api_validation", "py.pydantic.model_validate")
def test_parse_actions_partial(adapter):
    items = [
        {"type": "search", "query": "a"},
        {"type": "search", "query": ""},
        {"type": "calculator", "expression": "1+1"},
        "not even a dict",
        {"type": "final_answer", "answer": "done"},
    ]
    actions, failures = ex.parse_actions(items, adapter)
    assert [type(a).__name__ for a in actions] == ["SearchAction", "CalculatorAction", "FinalAnswer"]
    assert [i for i, _ in failures] == [1, 3]
    assert all(isinstance(e, ValidationError) for _, e in failures)


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, history):
        self.calls.append(json.loads(json.dumps(history)))  # snapshot
        return self.replies.pop(0)


BASE = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "2+2?"}]


@pytest.mark.topic("py.pydantic.api_validation")
def test_next_action_first_try(adapter):
    llm = ScriptedLLM('{"type": "calculator", "expression": "2+2"}')
    action = ex.next_action(llm, BASE, adapter)
    assert action == ex.CalculatorAction(type="calculator", expression="2+2")
    assert llm.calls == [BASE]


@pytest.mark.topic("py.pydantic.api_validation", "py.pydantic.validation")
def test_next_action_repairs_with_feedback(adapter):
    messages = list(BASE)
    llm = ScriptedLLM(
        "Sure! I'll calculate.",
        '{"type": "search", "query": "", "top_k": 0}',
        '{"type": "final_answer", "answer": "4"}',
    )
    action = ex.next_action(llm, messages, adapter)
    assert isinstance(action, ex.FinalAnswer)
    assert messages == BASE, "caller's history must not be mutated"
    assert len(llm.calls) == 3
    second, third = llm.calls[1], llm.calls[2]
    assert second[:2] == BASE
    assert second[2] == {"role": "assistant", "content": "Sure! I'll calculate."}
    assert second[3]["role"] == "user"
    assert second[3]["content"].startswith(ex.FEEDBACK_PREFIX + "(root): Invalid JSON")
    assert third[:4] == second
    assert third[5]["content"] == (
        ex.FEEDBACK_PREFIX
        + "search.query: String should have at least 1 character; "
        + "search.top_k: Input should be greater than or equal to 1"
    )


@pytest.mark.topic("py.pydantic.api_validation")
def test_next_action_gives_up(adapter):
    llm = ScriptedLLM('{"type": "dance"}', '{"type": "dance"}', "{}")
    with pytest.raises(ex.AgentOutputError) as info:
        ex.next_action(llm, BASE, adapter, max_attempts=2)
    assert info.value.attempts == 2
    assert isinstance(info.value.last_error, ValidationError)
    assert info.value.__cause__ is info.value.last_error
    assert len(llm.calls) == 2


@pytest.mark.topic("py.pydantic.api_validation")
def test_next_action_rejects_zero_attempts(adapter):
    with pytest.raises(ValueError):
        ex.next_action(ScriptedLLM(), BASE, adapter, max_attempts=0)
