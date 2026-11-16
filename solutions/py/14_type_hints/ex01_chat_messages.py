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
from typing import Literal, NotRequired, TypedDict, get_args

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(TypedDict):
    role: Role
    content: str
    name: NotRequired[str]


def make_message(role: Role, content: str, name: str | None = None) -> ChatMessage:
    if role not in get_args(Role):
        raise ValueError(f"invalid role {role!r}; expected one of {get_args(Role)}")
    if not isinstance(content, str):
        raise TypeError(f"content must be str, got {type(content).__name__}")
    msg: ChatMessage = {"role": role, "content": content}
    if name is not None:
        msg["name"] = name
    return msg


def validate_messages(raw: list[dict[str, object]]) -> list[ChatMessage]:
    allowed = ChatMessage.__required_keys__ | ChatMessage.__optional_keys__
    out: list[ChatMessage] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"message {i}: expected an object, got {type(item).__name__}")
        missing = ChatMessage.__required_keys__ - item.keys()
        extra = item.keys() - allowed
        if missing or extra:
            raise ValueError(f"message {i}: missing {sorted(missing)}, unknown {sorted(extra)}")
        name = item.get("name")
        if "name" in item and not isinstance(name, str):
            raise ValueError(f"message {i}: name must be a string")
        try:
            out.append(make_message(item["role"], item["content"], name))  # type: ignore[arg-type]
        except (ValueError, TypeError) as e:
            raise ValueError(f"message {i}: {e}") from e
    return out


def tokens_by_role(messages: Sequence[ChatMessage], count: Callable[[str], int] = len) -> dict[Role, int]:
    totals: dict[Role, int] = {}
    for m in messages:
        totals[m["role"]] = totals.get(m["role"], 0) + count(m["content"])
    return totals


def last_message(messages: Sequence[ChatMessage], role: Role = "user") -> ChatMessage | None:
    for m in reversed(messages):
        if m["role"] == role:
            return m
    return None
