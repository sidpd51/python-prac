"""
Exercise 10.1 - Parsing LLM tool calls safely                    level: ★☆☆
topics: exceptions.try_except, exceptions.multiple, exceptions.raising,
        exceptions.chaining, exceptions.syntax_vs_runtime

Problem
-------
An LLM returns a tool call as JSON text:   {"name": "search", "arguments": {"q": "..."}}
("arguments" may also be a JSON-encoded *string*, which is how the OpenAI API sends it).

1. class ToolCallError(ValueError)
     __init__(self, message: str, raw: str)  -> stores .raw (the offending text);
     str(err) == message

2. parse_tool_call(raw: str) -> ToolCall          (ToolCall dataclass is given)
     * "arguments" missing -> {} ; if it is a str, json-decode it too
     * every *data* problem raises ToolCallError - callers must never see a bare
       json.JSONDecodeError / KeyError / TypeError coming out of this function:
         - invalid JSON (outer text, or the inner arguments string)
               -> ToolCallError chained with `raise ... from e`, so err.__cause__ is the
                  json.JSONDecodeError
         - top level is not an object, or "name" is missing   -> ToolCallError
           (hint: `data["name"]` raises KeyError for a dict without it and TypeError for a
            list/str/int - one `except (KeyError, TypeError)` handles both; chain it)
         - "name" not a non-empty str, arguments not an object  -> ToolCallError
     * `raw` that is not a str at all (None, bytes...) is a *programming* bug, not bad
       data: let the TypeError from json.loads propagate - don't wrap it.

3. parse_many(raws: Iterable[str]) -> tuple[list[ToolCall], list[tuple[int, ToolCallError]]]
     * parse every item; failures are collected as (index, error) and don't stop the loop
     * catch ToolCallError ONLY (a None item must still blow up with TypeError)

4. syntax_error_at(source: str) -> tuple[int, str] | None
     * LLM-generated Python: return (lineno, msg) of its SyntaxError WITHOUT executing it
       (compile() / ast.parse), else None
     * code that would only fail at runtime (1/0, undefined names) is syntactically valid
       -> None, and it must not run (no side effects)
     * source containing NUL bytes counts as a syntax error too (lineno 0 if unknown)

Interviewers look for
---------------------
* Small try blocks: only the line that can raise the expected error goes in `try`;
  the rest goes in `else` (so you don't accidentally catch your own bugs).
* Translating low-level errors into a domain error at an API boundary, with `from e`
  so the original traceback is kept ("The above exception was the direct cause...").
* Catching exactly what you expect - a broad `except Exception` here would also hide
  the TypeError for None input, a real bug.
* SyntaxError is raised when code is *compiled*, before any of it runs; runtime errors
  only when the offending line executes.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolCallError(ValueError):
    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


def parse_tool_call(raw: str) -> ToolCall:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ToolCallError(f"tool call is not valid JSON: {e.msg}", raw) from e

    try:
        name = data["name"]
    except (KeyError, TypeError) as e:
        raise ToolCallError("tool call must be an object with a 'name'", raw) from e
    if not isinstance(name, str) or not name:
        raise ToolCallError("'name' must be a non-empty string", raw)

    args = data.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            raise ToolCallError(f"arguments are not valid JSON: {e.msg}", raw) from e
    if not isinstance(args, dict):
        raise ToolCallError("'arguments' must be a JSON object", raw)
    return ToolCall(name, args)


def parse_many(
    raws: Iterable[str],
) -> tuple[list[ToolCall], list[tuple[int, ToolCallError]]]:
    ok: list[ToolCall] = []
    failed: list[tuple[int, ToolCallError]] = []
    for i, raw in enumerate(raws):
        try:
            call = parse_tool_call(raw)
        except ToolCallError as e:
            failed.append((i, e))
        else:
            ok.append(call)
    return ok, failed


def syntax_error_at(source: str) -> tuple[int, str] | None:
    try:
        compile(source, "<llm-generated>", "exec")  # compiles only; nothing is executed
    except SyntaxError as e:
        return (e.lineno or 0, e.msg)
    return None
