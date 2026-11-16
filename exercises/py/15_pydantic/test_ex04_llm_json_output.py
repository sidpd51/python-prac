import json

import pytest
from pydantic import BaseModel, ValidationError

from study.loader import load

ex = load(__file__, "ex04_llm_json_output")

GOOD = '{"category": "bug", "priority": 2, "summary": "Login fails"}'


@pytest.mark.topic("py.pydantic.api_validation")
@pytest.mark.parametrize(
    "text",
    [
        GOOD,
        f"  {GOOD}\n",
        f"Sure! Here is the triage:\n```json\n{GOOD}\n```\nLet me know if you need anything else.",
        f"```JSON\n{GOOD}```",
        f"```\n{GOOD}\n```",
        f"Here you go: {GOOD} Hope that helps!",
        f"Fill the template {{name}} first. Result: {GOOD}",  # brace in prose
    ],
)
def test_extract_json_variants(text):
    assert ex.extract_json(text) == GOOD


@pytest.mark.topic("py.pydantic.api_validation")
def test_extract_json_braces_inside_strings():
    tricky = '{"summary": "use } and { freely", "n": [1, {"x": "]"}]}'
    assert ex.extract_json(f"Answer: {tricky} -- done }}}}") == tricky


@pytest.mark.topic("py.pydantic.api_validation")
def test_extract_json_first_fence_only_and_arrays():
    text = 'Example (do not use): ```json\n[1, 2]\n``` and the real one ```json\n{"a": 1}\n```'
    assert ex.extract_json(text) == "[1, 2]"
    assert ex.extract_json('tags: ["a", "b"]') == '["a", "b"]'


@pytest.mark.topic("py.pydantic.api_validation")
@pytest.mark.parametrize("text", ["", "I cannot help with that.", "{not json}", '{"a": 1'])
def test_extract_json_nothing_found(text):
    with pytest.raises(ex.LLMOutputError):
        ex.extract_json(text)


@pytest.mark.topic("py.pydantic.model_validate", "py.pydantic.serialization")
def test_parse_llm_output_into_model():
    triage = ex.parse_llm_output(f"```json\n{GOOD}\n```", ex.TicketTriage)
    assert isinstance(triage, ex.TicketTriage)
    assert (triage.category, triage.priority, triage.tags, triage.needs_human) == (
        "bug",
        2,
        [],
        False,
    )


@pytest.mark.topic("py.pydantic.model_validate")
def test_parse_llm_output_is_generic():
    class Sentiment(BaseModel):
        label: str
        score: float

    out = ex.parse_llm_output('Result -> {"label": "pos", "score": 0.93}', Sentiment)
    assert out == Sentiment(label="pos", score=0.93)


@pytest.mark.topic("py.pydantic.validation", "py.pydantic.api_validation")
def test_parse_llm_output_errors_propagate():
    with pytest.raises(ValidationError):
        ex.parse_llm_output('{"category": "spam", "priority": 2, "summary": "x"}', ex.TicketTriage)
    with pytest.raises(ex.LLMOutputError):
        ex.parse_llm_output("no json here", ex.TicketTriage)


def _err(payload):
    try:
        ex.TicketTriage.model_validate_json(json.dumps(payload))
    except ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


@pytest.mark.topic("py.pydantic.validation", "py.pydantic.api_validation")
def test_error_feedback_exact_format():
    err = _err({"category": "bug", "priority": 9, "summary": "x", "tags": ["ok", 3]})
    assert ex.error_feedback(err) == (
        "Your JSON failed validation with 2 error(s):\n"
        "- priority: Input should be less than or equal to 5 (got 9)\n"
        "- tags.1: Input should be a valid string (got 3)\n"
        "Return ONLY the corrected JSON object."
    )


@pytest.mark.topic("py.pydantic.validation")
def test_error_feedback_missing_and_non_scalar_inputs():
    err = _err({"priority": [1], "summary": "x"})
    lines = ex.error_feedback(err).splitlines()
    assert lines[0] == "Your JSON failed validation with 2 error(s):"
    assert "- category: Field required" in lines  # no "(got ...)" for missing
    assert "- priority: Input should be a valid integer" in lines  # list input not echoed


@pytest.mark.topic("py.pydantic.validation")
def test_error_feedback_root_loc():
    err = _err(["not", "an", "object"])
    lines = ex.error_feedback(err).splitlines()
    assert lines[1].startswith("- (root): Input should be an object")
