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

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, field_validator, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string, exactly as the LLM emitted it


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

    def parsed_arguments(self) -> dict[str, Any]:
        args = json.loads(self.function.arguments)
        if not isinstance(args, dict):
            raise ValueError(f"tool arguments must be a JSON object, got {type(args).__name__}")
        return args


class Message(BaseModel):
    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @field_validator("role", mode="before")
    @classmethod
    def normalise_role(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool_calls")
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError("only tool messages may have a tool_call_id")
        match self.role:
            case "tool":
                if self.tool_call_id is None or self.content is None:
                    raise ValueError("tool messages need tool_call_id and content")
            case "assistant":
                if self.content is None and not self.tool_calls:
                    raise ValueError("assistant message needs content or tool_calls")
            case _:
                if self.content is None:
                    raise ValueError(f"{self.role} message needs content")
        return self


class Conversation(BaseModel):
    messages: list[Message]

    @model_validator(mode="after")
    def check_order(self) -> Self:
        if not self.messages:
            raise ValueError("conversation needs at least one message")
        issued: set[str] = set()
        for i, msg in enumerate(self.messages):
            if msg.role == "system" and i != 0:
                raise ValueError(f"system message only allowed first (found at {i})")
            if msg.role == "assistant" and msg.tool_calls:
                issued.update(tc.id for tc in msg.tool_calls)
            if msg.role == "tool" and msg.tool_call_id not in issued:
                raise ValueError(f"tool message {i} answers unknown call {msg.tool_call_id!r}")
        return self

    def last_assistant_text(self) -> str | None:
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.content is not None:
                return msg.content
        return None
