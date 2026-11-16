"""
Exercise 15.7 - Agent actions: discriminated unions + repair loop  level: ★★★
topics: pydantic.api_validation, pydantic.validation, pydantic.model_validate,
        pydantic.models, pydantic.nested

Problem
-------
An agent LLM replies with ONE JSON action per turn, e.g.
    {"type": "search", "query": "refund policy", "top_k": 3}
    {"type": "final_answer", "answer": "...", "citations": ["doc-1"]}
The action models and AgentOutputError are given.

1. build_action_adapter(*models) -> pydantic.TypeAdapter
     * a TypeAdapter for a union of the models DISCRIMINATED on the "type" field
       (Annotated[Union[...], Field(discriminator="type")])
     * no models -> ValueError
     * every model must declare `type` as a Literal -> otherwise TypeError
     * two models claiming the same tag value -> ValueError (check it yourself, before
       building the adapter)
     Effect to observe: an invalid {"type": "search", "top_k": 99} gives ONE error located
     at ("search", "top_k") instead of one error per union member.

2. parse_actions(items, adapter) -> (actions, failures)
     * the LLM emitted several actions (parallel tool calls) as already-parsed dicts
     * validate each independently; actions = the valid ones in input order;
       failures = [(index, ValidationError), ...] for the invalid ones
     * one bad item must not prevent the others from running

3. next_action(call_llm, messages, adapter, max_attempts=3) -> action
     * call_llm(history: list[dict]) -> str  (raw model output)
     * validate the raw text with adapter.validate_json
     * on ValidationError: extend the history with
           {"role": "assistant", "content": <raw text>}
           {"role": "user", "content": feedback}
       where feedback = "Invalid action. Fix these errors and reply with JSON only: "
                        + "; ".join(f"{loc}: {msg}") over err.errors(), loc parts joined
                        by "." ("(root)" if empty)
       and call the LLM again with the longer history
     * the caller's `messages` list must NOT be mutated
     * after max_attempts failed attempts raise AgentOutputError(attempts, last_error)
       chained (`raise ... from`) to the last ValidationError
     * max_attempts < 1 -> ValueError

Interviewers look for
---------------------
* Discriminated (tagged) unions: O(1) dispatch on the tag, precise error locations, and a
  clean `oneOf` + `discriminator` JSON schema; plain unions try members left-to-right
  and may silently coerce into the wrong one ("smart mode" mitigates but costs).
* TypeAdapter validates non-BaseModel types (unions, lists, TypedDicts).
* Self-correction / "reflexion" loop: feed structured validation errors back, bounded
  by a retry budget; keep the original conversation immutable.
"""

import typing
from collections.abc import Callable
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

__stub_keep__ = ["AgentOutputError"]

FEEDBACK_PREFIX = "Invalid action. Fix these errors and reply with JSON only: "


class SearchAction(BaseModel):
    type: Literal["search"]
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class CalculatorAction(BaseModel):
    type: Literal["calculator"]
    expression: str


class FinalAnswer(BaseModel):
    type: Literal["final_answer"]
    answer: str
    citations: list[str] = Field(default_factory=list)


class AgentOutputError(Exception):
    def __init__(self, attempts: int, last_error: ValidationError) -> None:
        super().__init__(f"no valid action after {attempts} attempt(s)")
        self.attempts = attempts
        self.last_error = last_error


def build_action_adapter(*models: type[BaseModel]) -> TypeAdapter[Any]:
    if not models:
        raise ValueError("need at least one action model")
    seen: dict[str, str] = {}
    for model in models:
        field = model.model_fields.get("type")
        if field is None or typing.get_origin(field.annotation) is not Literal:
            raise TypeError(f"{model.__name__}.type must be a Literal")
        for tag in typing.get_args(field.annotation):
            if tag in seen:
                raise ValueError(f"tag {tag!r} used by {seen[tag]} and {model.__name__}")
            seen[tag] = model.__name__
    return TypeAdapter(Annotated[Union[models], Field(discriminator="type")])  # noqa: UP007


def parse_actions(
    items: list[Any], adapter: TypeAdapter[Any]
) -> tuple[list[Any], list[tuple[int, ValidationError]]]:
    actions: list[Any] = []
    failures: list[tuple[int, ValidationError]] = []
    for i, item in enumerate(items):
        try:
            actions.append(adapter.validate_python(item))
        except ValidationError as err:
            failures.append((i, err))
    return actions, failures


def _feedback(err: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
        for e in err.errors(include_url=False)
    ]
    return FEEDBACK_PREFIX + "; ".join(parts)


def next_action(
    call_llm: Callable[[list[dict[str, str]]], str],
    messages: list[dict[str, str]],
    adapter: TypeAdapter[Any],
    max_attempts: int = 3,
) -> Any:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    history = list(messages)
    last_error: ValidationError | None = None
    for _ in range(max_attempts):
        raw = call_llm(list(history))
        try:
            return adapter.validate_json(raw)
        except ValidationError as err:
            last_error = err
            history += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _feedback(err)},
            ]
    assert last_error is not None
    raise AgentOutputError(max_attempts, last_error) from last_error
