"""
Exercise 01.5 - Safe access into nested LLM API responses           level: ★★☆
topics: fundamentals.nested_structures, fundamentals.none, fundamentals.conditionals,
        fundamentals.type_conversion, fundamentals.loops

Problem
-------
LLM APIs return deeply nested JSON (choices -> message -> tool_calls -> function ...),
and fields go missing (content filters, truncation, provider differences).

1. get_path(obj, path, default=None)
     * path is either a dotted string "choices.0.message.content" or a list of segments
       (["choices", 0, "message", "content"])
     * at a dict: look the segment up as a key
     * at a list/tuple: the segment must be an int, or a str that int() accepts
       ("0", "-1"); negative indexes work like Python's
     * any miss (missing key, index out of range, non-numeric index into a list,
       descending into a scalar or None) -> `default`
     * a key that exists with value None returns None, NOT default
     * empty path ("" or []) -> obj itself
     * never raises for a missing/malformed path

2. flatten(obj: dict, sep: str = ".") -> dict[str, Any]
     * {"a": {"b": 1, "c": [10, {"d": 2}]}} -> {"a.b": 1, "a.c.0": 10, "a.c.1.d": 2}
     * list elements are addressed by index; empty dicts/lists are kept as leaf values
       ({"meta": {}} -> {"meta": {}})
     * flatten({}) -> {};  obj that is not a dict -> TypeError
     * the order of keys follows a depth-first walk of the input

3. extract_tool_calls(response: dict) -> list[tuple[str, dict | None]]
     * OpenAI-style: response["choices"][0]["message"]["tool_calls"] is a list of
       {"function": {"name": str, "arguments": "<JSON string>"}}
     * return [(name, parsed_arguments), ...]; arguments that are not valid JSON or don't
       decode to a dict -> None (the model produced broken JSON - caller can retry)
     * no choices / no message / tool_calls missing or None -> []
     * entries without a function name are skipped

Interviewers look for
---------------------
* "missing" vs "present but None" are different states - a sentinel object (not None)
  is how you tell them apart.
* d.get(k, {}) chains break the moment a value is a list or None; write one helper.
* isinstance checks on the *container* type decide how a segment is interpreted.
* Never trust model output: json.loads in try/except json.JSONDecodeError, then validate
  the type you got.
"""

import json
from collections.abc import Sequence
from typing import Any

_MISSING = object()


def get_path(obj: Any, path: str | Sequence[str | int], default: Any = None) -> Any:
    if isinstance(path, str):
        segments: Sequence[str | int] = path.split(".") if path else []
    else:
        segments = path
    current = obj
    for seg in segments:
        if isinstance(current, dict):
            current = current.get(seg, _MISSING)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(seg)]
            except (ValueError, TypeError, IndexError):
                return default
        else:
            return default
        if current is _MISSING:
            return default
    return current


def flatten(obj: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict, got {type(obj).__name__}")
    flat: dict[str, Any] = {}
    if obj:
        _walk(obj, "", sep, flat)
    return flat


def _walk(node: Any, prefix: str, sep: str, out: dict[str, Any]) -> None:
    if isinstance(node, dict) and node:
        items: Any = node.items()
    elif isinstance(node, list) and node:
        items = enumerate(node)
    else:
        out[prefix] = node
        return
    for key, value in items:
        _walk(value, f"{prefix}{sep}{key}" if prefix else str(key), sep, out)


def extract_tool_calls(response: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None]]:
    calls = get_path(response, "choices.0.message.tool_calls") or []
    out: list[tuple[str, dict[str, Any] | None]] = []
    for call in calls:
        name = get_path(call, "function.name")
        if not name:
            continue
        raw = get_path(call, "function.arguments")
        try:
            args = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            args = None
        out.append((name, args if isinstance(args, dict) else None))
    return out
