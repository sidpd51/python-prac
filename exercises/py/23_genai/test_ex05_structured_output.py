import json

import pydantic
import pytest

from study.loader import load

ex = load(__file__, "ex05_structured_output")

VALID = {"answer": "Paris", "confidence": 0.9, "citations": [{"source_id": "doc-3", "quote": "Paris is..."}]}


class ScriptedLLM:
    """Fake chat LLM: returns scripted replies and records every conversation it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.conversations = []

    def __call__(self, messages):
        self.conversations.append(messages)
        return self.replies.pop(0)


@pytest.mark.topic("py.genai.structured_output", "py.genai.regex")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('Sure!\n```json\n{"a": 1}\n```\nHope it helps', '{"a": 1}'),
        ('```\n  {"a": 1}  \n```', '{"a": 1}'),
        ('```JSON \n{"a": 1}\n```', '{"a": 1}'),
        ('Result: {"a": {"b": 2}} - done', '{"a": {"b": 2}}'),
        ('first\n```json\n{"x": 1}\n```\nsecond\n```json\n{"y": 2}\n```', '{"x": 1}'),
        ('Here: {"text": "use ``` fences"}', '{"text": "use ``` fences"}'),
    ],
)
def test_extract_json(text, expected):
    assert ex.extract_json(text) == expected


@pytest.mark.topic("py.genai.structured_output")
@pytest.mark.parametrize("text", ["no json here", "", "} backwards {"])
def test_extract_json_none_found(text):
    with pytest.raises(ValueError):
        ex.extract_json(text)


@pytest.mark.topic("py.genai.structured_output", "py.genai.structured_data")
def test_parse_answer_valid():
    ans = ex.parse_answer("Here you go:\n```json\n" + json.dumps(VALID) + "\n```")
    assert isinstance(ans, ex.Answer)
    assert ans.answer == "Paris" and ans.confidence == 0.9
    assert ans.citations[0].source_id == "doc-3"


@pytest.mark.topic("py.genai.structured_output", "py.genai.structured_data")
@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "x", "confidence": 1.5},  # out of range
        {"answer": "   ", "confidence": 0.5},  # empty after stripping
        {"confidence": 0.5},  # missing field
        {"answer": "x", "confidence": 0.5, "source": "hallucinated"},  # extra field
        {"answer": "x", "confidence": 0.5, "citations": [{"source_id": "", "quote": "q"}]},
    ],
)
def test_parse_answer_schema_violations(payload):
    with pytest.raises(pydantic.ValidationError):
        ex.parse_answer(json.dumps(payload))


@pytest.mark.topic("py.genai.structured_output")
def test_parse_answer_bad_json_is_value_error():
    with pytest.raises(ValueError):
        ex.parse_answer('{"answer": "x", "confidence": 0.5,}')


@pytest.mark.topic("py.genai.structured_output")
def test_generate_first_try():
    llm = ScriptedLLM(json.dumps(VALID))
    result = ex.generate_structured(llm, "Capital of France?", ex.Answer)
    assert result == ex.Answer.model_validate(VALID)
    (conversation,) = llm.conversations
    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"] == ex.SYSTEM_TEMPLATE.format(
        schema=json.dumps(ex.Answer.model_json_schema())
    )
    assert conversation[1] == {"role": "user", "content": "Capital of France?"}
    assert len(conversation) == 2


@pytest.mark.topic("py.genai.structured_output")
def test_generate_repairs_with_error_feedback():
    bad1 = "I think the answer is Paris."  # no JSON at all
    bad2 = '{"answer": "Paris", "confidence": 7}'  # schema violation
    llm = ScriptedLLM(bad1, bad2, "```json\n" + json.dumps(VALID) + "\n```")
    result = ex.generate_structured(llm, "Capital of France?", ex.Answer, max_attempts=3)
    assert result.answer == "Paris"
    assert [len(c) for c in llm.conversations] == [2, 4, 6], "each call gets the whole conversation"
    third = llm.conversations[2]
    assert third[2] == {"role": "assistant", "content": bad1}
    assert third[4] == {"role": "assistant", "content": bad2}
    assert third[3]["role"] == third[5]["role"] == "user"
    assert "confidence" in third[5]["content"], "the pydantic error must be fed back"
    assert third[5]["content"].startswith("Your previous reply was invalid:")


@pytest.mark.topic("py.genai.structured_output")
def test_generate_sends_copies():
    llm = ScriptedLLM("nope", json.dumps(VALID))
    ex.generate_structured(llm, "q", ex.Answer)
    first, second = llm.conversations
    assert first is not second
    assert len(first) == 2, "the first conversation must not grow after the call"


@pytest.mark.topic("py.genai.structured_output")
def test_generate_gives_up():
    llm = ScriptedLLM("nope", '{"answer": ""}', "still nope", "never used")
    with pytest.raises(ex.StructuredOutputError) as info:
        ex.generate_structured(llm, "q", ex.Answer, max_attempts=3)
    assert len(llm.conversations) == 3
    assert isinstance(info.value.__cause__, ValueError), "chain the last error with `from`"


@pytest.mark.topic("py.genai.structured_output", "py.genai.structured_data")
def test_generate_works_with_any_schema():
    class Ticket(pydantic.BaseModel):
        category: str
        urgency: int = pydantic.Field(ge=1, le=5)

    llm = ScriptedLLM('{"category": "billing", "urgency": 9}', '{"category": "billing", "urgency": 4}')
    ticket = ex.generate_structured(llm, "classify", Ticket)
    assert ticket == Ticket(category="billing", urgency=4)
    assert "urgency" in llm.conversations[0][0]["content"], "system prompt carries the schema"
