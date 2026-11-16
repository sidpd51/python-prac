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

from collections.abc import Callable, Mapping
from typing import Any


def tools_from(*fns: Callable[..., Any]) -> dict[str, Callable[..., Any]]:
    raise NotImplementedError


def tool_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    raise NotImplementedError


def run_tool(tools: Mapping[str, Callable[..., Any]], call: Mapping[str, Any]) -> dict[str, Any]:
    raise NotImplementedError
