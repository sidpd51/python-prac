"""
Exercise 14.1 - Typed chat messages (TypedDict + Literal)        level: ★☆☆
topics: type_hints.typeddict, type_hints.literal, type_hints.basic,
        type_hints.functions, type_hints.returns, type_hints.optional,
        type_hints.collections

Problem
-------
The types are given: `Role` (a Literal) and `ChatMessage` (a TypedDict) - the exact
shape chat-completion APIs use. Type hints are NOT checked at runtime, so data coming
from JSON/HTTP still has to be validated by code - ideally derived from the same types,
so there is one source of truth.

1. make_message(role, content, name=None) -> ChatMessage
     * role must be one of the Literal's values - read them with typing.get_args(Role),
       don't repeat the strings -> ValueError otherwise
     * content must be a str -> TypeError otherwise
     * the "name" key is present only when name is not None

2. validate_messages(raw: list[dict[str, object]]) -> list[ChatMessage]
     * every item must be a dict with all required keys of ChatMessage
       (ChatMessage.__required_keys__), no keys outside required + optional
       (__optional_keys__), a valid role, str content, and str name if present
     * any problem -> ValueError whose message starts with f"message {index}: "
     * returns NEW dicts (mutating the result must not change the input)

3. tokens_by_role(messages, count=len) -> dict[Role, int]
     * sum count(content) per role; roles that don't occur are absent

4. last_message(messages, role="user") -> ChatMessage | None
     * the last message with that role, or None

Interviewers look for
---------------------
* Annotations are metadata: Python never enforces them; mypy/pyright do, statically.
* TypedDict = typed *dict* shape (zero runtime cost, JSON-friendly) vs dataclass /
  pydantic model (real classes, pydantic validates at runtime).
* `X | None` (Optional) means "may be None" - not "the argument is optional".
* Literal for closed sets of strings (roles, finish reasons) + get_args() to reuse them.
"""

from collections.abc import Callable, Sequence
from typing import Literal, NotRequired, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(TypedDict):
    role: Role
    content: str
    name: NotRequired[str]


def make_message(role: Role, content: str, name: str | None = None) -> ChatMessage:
    raise NotImplementedError


def validate_messages(raw: list[dict[str, object]]) -> list[ChatMessage]:
    raise NotImplementedError


def tokens_by_role(messages: Sequence[ChatMessage], count: Callable[[str], int] = len) -> dict[Role, int]:
    raise NotImplementedError


def last_message(messages: Sequence[ChatMessage], role: Role = "user") -> ChatMessage | None:
    raise NotImplementedError
