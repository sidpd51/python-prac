"""
Exercise 03.4 - Agent tool registry and dispatcher                  level: ★★★
topics: functions.first_class, functions.higher_order, functions.reference_vs_call,
        functions.kwargs, functions.unpacking, functions.definition, functions.pos_kw_only

Problem
-------
An agent loop gets tool calls from the model as JSON: {"name": "search", "arguments": {...}}.
Functions are objects - store them, inspect them, call them later.

1. tools_from(*fns) -> dict[str, Callable]
     * maps fn.__name__ -> fn
     * an argument that is not callable -> TypeError (the classic bug: registering
       `search()` - the RESULT of a call - instead of `search`)
     * two functions with the same __name__ -> ValueError

2. tool_schema(fn) -> dict
     * {"name": fn.__name__, "description": <first line of the docstring, stripped, or "">,
        "required": [names of params without defaults, in order],
        "optional": {name: default, ...}}
     * use inspect.signature
     * the model passes arguments BY NAME, so a tool may not have *args, **kwargs or
       positional-only parameters -> TypeError

3. run_tool(tools, call) -> dict
     * call["arguments"] may be a dict or a JSON string (models emit strings); missing -> {}
     * unknown tool              -> {"ok": False, "error": "unknown tool: <name>"}
     * invalid JSON / not a dict -> {"ok": False, "error": "bad arguments: ..."}
     * arguments that don't fit the signature (missing / unexpected) ->
       {"ok": False, "error": "bad arguments: ..."}  WITHOUT calling the tool
     * exception raised INSIDE the tool ->
       {"ok": False, "error": "tool error: <ExceptionType>: <message>"}
       (a TypeError from inside the tool is a tool error, not "bad arguments"!)
     * success -> {"ok": True, "result": <return value>}
     * the error dict is fed back to the model so it can correct itself - never crash the loop

Interviewers look for
---------------------
* `f` is a reference, `f()` is a call. `handlers = {"a": run_a()}` calls at build time and
  stores the result.
* Higher-order design: a registry is just a dict of callables; dispatch is `tools[name](**args)`.
* Validate the call with `inspect.signature(fn).bind(**args)` BEFORE invoking, instead of
  catching TypeError afterwards (which can't tell a bad call from a bug inside the tool).
* Treat model output as untrusted input: parse, validate, return structured errors.
"""

import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

_BY_NAME_OK = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def tools_from(*fns: Callable[..., Any]) -> dict[str, Callable[..., Any]]:
    tools: dict[str, Callable[..., Any]] = {}
    for fn in fns:
        if not callable(fn):
            raise TypeError(f"{fn!r} is not callable - did you call the function instead of passing it?")
        name = fn.__name__
        if name in tools:
            raise ValueError(f"duplicate tool name {name!r}")
        tools[name] = fn
    return tools


def tool_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    required: list[str] = []
    optional: dict[str, Any] = {}
    for p in inspect.signature(fn).parameters.values():
        if p.kind not in _BY_NAME_OK:
            raise TypeError(f"parameter {p.name!r} of {fn.__name__} can't be passed by name")
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
        else:
            optional[p.name] = p.default
    doc = inspect.getdoc(fn) or ""
    return {
        "name": fn.__name__,
        "description": doc.splitlines()[0].strip() if doc else "",
        "required": required,
        "optional": optional,
    }


def run_tool(tools: Mapping[str, Callable[..., Any]], call: Mapping[str, Any]) -> dict[str, Any]:
    name = call.get("name")
    fn = tools.get(name) if isinstance(name, str) else None
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"bad arguments: invalid JSON ({e})"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "bad arguments: expected a JSON object"}
    try:
        inspect.signature(fn).bind(**args)
    except TypeError as e:
        return {"ok": False, "error": f"bad arguments: {e}"}
    try:
        result = fn(**args)
    except Exception as e:  # the agent loop must survive any tool failure
        return {"ok": False, "error": f"tool error: {type(e).__name__}: {e}"}
    return {"ok": True, "result": result}
