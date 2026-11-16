"""
Exercise 15.1 - Chat-completion messages                         level: ★☆☆
topics: pydantic.basemodel, pydantic.models, pydantic.nested, pydantic.required_optional,
        pydantic.custom_validators, pydantic.model_validator, pydantic.validation

Problem
-------
The FIELDS of the models below are given (read them - they are part of the spec).
Your job is the validation logic and two small methods.

FunctionCall(name, arguments)      `arguments` is the raw JSON *string* the LLM produced
ToolCall(id, type="function", function: FunctionCall)
Message(role, content=None, name=None, tool_calls=None, tool_call_id=None)
Conversation(messages: list[Message])

1. Message.normalise_role  - field_validator("role", mode="before")
     * if the raw value is a str: strip whitespace and lowercase it (" User " -> "user")
     * anything else is returned unchanged (so the Literal type rejects it normally)

2. Message.clean_content   - field_validator("content")   (runs AFTER str validation)
     * strip surrounding whitespace; a whitespace-only / empty string becomes None

3. Message.check_shape     - model_validator(mode="after"), raise ValueError on:
     * role "tool" without tool_call_id, or without content
     * tool_calls on any role other than "assistant"
     * tool_call_id on any role other than "tool"
     * role "assistant" with neither content nor tool_calls
     * role "system"/"user" without content
     returns self when valid

4. ToolCall.parsed_arguments() -> dict
     * json.loads(function.arguments); must decode to a dict, else ValueError
       (json.JSONDecodeError is already a ValueError subclass - let it propagate)

5. Conversation.check_order  - model_validator(mode="after")
     * at least one message (ValueError otherwise)
     * a "system" message is only allowed at index 0
     * every "tool" message must answer a tool call id issued by an EARLIER
       assistant message in the conversation

6. Conversation.last_assistant_text() -> str | None
     * content of the last assistant message that has content, else None

All violations must surface as pydantic.ValidationError when constructing the model
(raising ValueError inside a validator is what makes pydantic wrap it).

Interviewers look for
---------------------
* `mode="before"` sees the RAW input (could be anything); the default "after" sees the
  already type-validated value - choose based on whether you need to pre-process.
* Required vs optional: `x: str | None` WITHOUT a default is still REQUIRED in pydantic v2;
  it only becomes optional with `= None`.
* Cross-field rules belong in a model_validator(mode="after"), where all fields are typed.
* Validators raise ValueError/AssertionError -> pydantic collects them into ValidationError;
  any other exception type escapes un-wrapped (a classic bug).
* Nested models validate recursively: errors carry a `loc` path like ('messages', 2, 'role').
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, field_validator, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

    def parsed_arguments(self) -> dict[str, Any]:
        raise NotImplementedError


class Message(BaseModel):
    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @field_validator("role", mode="before")
    @classmethod
    def normalise_role(cls, value: Any) -> Any:
        raise NotImplementedError

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str | None) -> str | None:
        raise NotImplementedError

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        raise NotImplementedError


class Conversation(BaseModel):
    messages: list[Message]

    @model_validator(mode="after")
    def check_order(self) -> Self:
        raise NotImplementedError

    def last_assistant_text(self) -> str | None:
        raise NotImplementedError
