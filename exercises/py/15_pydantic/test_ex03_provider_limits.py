import pytest
from pydantic import BaseModel, ValidationError

from study.loader import load

ex = load(__file__, "ex03_provider_limits")

OPENAI = ex.ProviderLimits("openai", max_output_tokens=4096, max_temperature=2.0, max_stop_sequences=4)
ANTHROPIC = ex.ProviderLimits(
    "anthropic", max_output_tokens=8192, max_temperature=1.0, max_stop_sequences=2, supports_seed=False
)
OK = {"model": "gpt-4o-mini", "prompt": "Summarise this.", "max_tokens": 256}


def _errors(model, payload):
    with pytest.raises(ValidationError) as info:
        model.model_validate(payload)
    return {(e["loc"], e["type"]) for e in info.value.errors()}


@pytest.mark.topic("py.pydantic.field", "py.pydantic.required_optional")
def test_defaults_and_name():
    Req = ex.request_model_for(OPENAI)
    assert issubclass(Req, BaseModel)
    assert Req.__name__ == "OpenaiRequest"
    req = Req.model_validate(OK)
    assert (req.temperature, req.top_p, req.stop, req.seed) == (1.0, None, [], None)


@pytest.mark.topic("py.pydantic.required_optional")
def test_required_fields():
    errs = _errors(ex.request_model_for(OPENAI), {})
    assert errs == {(("model",), "missing"), (("prompt",), "missing"), (("max_tokens",), "missing")}


@pytest.mark.topic("py.pydantic.constraints", "py.pydantic.validation")
@pytest.mark.parametrize(
    ("override", "loc", "etype"),
    [
        ({"max_tokens": 0}, "max_tokens", "greater_than"),
        ({"max_tokens": 4097}, "max_tokens", "less_than_equal"),
        ({"temperature": -0.1}, "temperature", "greater_than_equal"),
        ({"temperature": 2.5}, "temperature", "less_than_equal"),
        ({"top_p": 0}, "top_p", "greater_than"),
        ({"top_p": 1.5}, "top_p", "less_than_equal"),
        ({"stop": ["a", "b", "c", "d", "e"]}, "stop", "too_long"),
        ({"prompt": ""}, "prompt", "string_too_short"),
        ({"model": "   "}, "model", "string_too_short"),
    ],
)
def test_constraints(override, loc, etype):
    errs = _errors(ex.request_model_for(OPENAI), {**OK, **override})
    assert ((loc,), etype) in errs


@pytest.mark.topic("py.pydantic.constraints")
def test_boundaries_are_inclusive_where_specified():
    Req = ex.request_model_for(OPENAI)
    req = Req.model_validate({**OK, "max_tokens": 4096, "temperature": 2.0, "top_p": 1})
    assert (req.max_tokens, req.temperature) == (4096, 2.0)
    assert Req.model_validate({**OK, "model": "  gpt-4o  "}).model == "gpt-4o"


@pytest.mark.topic("py.pydantic.constraints")
def test_empty_stop_sequence_item_rejected():
    errs = _errors(ex.request_model_for(OPENAI), {**OK, "stop": ["\n\n", ""]})
    assert (("stop", 1), "string_too_short") in errs


@pytest.mark.topic("py.pydantic.constraints", "py.pydantic.validation")
@pytest.mark.parametrize("bad", ["100", 100.0, True])
def test_max_tokens_is_strict(bad):
    errs = _errors(ex.request_model_for(OPENAI), {**OK, "max_tokens": bad})
    assert any(loc == ("max_tokens",) for loc, _ in errs)


@pytest.mark.topic("py.pydantic.validation", "py.pydantic.field")
def test_extra_fields_forbidden():
    errs = _errors(ex.request_model_for(OPENAI), {**OK, "temprature": 0.2})
    assert errs == {(("temprature",), "extra_forbidden")}


@pytest.mark.topic("py.pydantic.field", "py.pydantic.constraints")
def test_provider_specific_limits():
    Req = ex.request_model_for(ANTHROPIC)
    assert Req.__name__ == "AnthropicRequest"
    req = Req.model_validate({**OK, "max_tokens": 8000})
    assert req.temperature == 1.0
    assert not hasattr(req, "seed")
    assert (("seed",), "extra_forbidden") in _errors(Req, {**OK, "seed": 7})
    assert (("temperature",), "less_than_equal") in _errors(Req, {**OK, "temperature": 1.5})
    # low max_temperature -> default must be clamped so the default itself is valid
    cold = ex.ProviderLimits("cold", 100, max_temperature=0.5, max_stop_sequences=1)
    assert ex.request_model_for(cold).model_validate({**OK, "max_tokens": 10}).temperature == 0.5


@pytest.mark.topic("py.pydantic.field")
def test_no_shared_mutable_default():
    Req = ex.request_model_for(OPENAI)
    a, b = Req.model_validate(OK), Req.model_validate(OK)
    a.stop.append("###")
    assert b.stop == []


@pytest.mark.topic("py.pydantic.models")
def test_models_are_cached():
    same = ex.ProviderLimits("openai", 4096, 2.0, 4)
    assert ex.request_model_for(same) is ex.request_model_for(OPENAI)
    assert ex.request_model_for(OPENAI) is not ex.request_model_for(ANTHROPIC)


@pytest.mark.topic("py.pydantic.json_schema", "py.pydantic.constraints")
def test_constraints_show_in_json_schema():
    schema = ex.request_model_for(OPENAI).model_json_schema()
    props = schema["properties"]
    assert schema["title"] == "OpenaiRequest"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"model", "prompt", "max_tokens"}
    assert props["max_tokens"]["maximum"] == 4096
    assert props["max_tokens"]["exclusiveMinimum"] == 0
    assert props["temperature"]["maximum"] == 2.0
    assert props["stop"]["maxItems"] == 4
    assert all(p.get("description") for p in props.values())
