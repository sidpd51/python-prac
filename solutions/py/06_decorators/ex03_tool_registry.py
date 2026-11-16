"""
Exercise 06.3 - @tool: a registration decorator for LLM function calling   level: ★★☆
topics: decorators.syntax, decorators.first_class, decorators.metadata, decorators.higher_order

Problem
-------
LLM "tool use" needs (a) a registry of callable tools and (b) a JSON-schema description of
each tool built from its signature. Write a REGISTRATION decorator - it does not wrap.

1. @tool
   * Registers the function in the module-level dict TOOLS under func.__name__.
     A name that is already registered -> ValueError.
   * Attaches `func.schema` (dict, format below) and returns the ORIGINAL function object
     (no wrapper - calling it directly must have zero overhead).
   * Schema, built with inspect.signature:
         {"name": "get_weather",
          "description": <first non-empty line of the docstring, stripped; "" if none>,
          "parameters": {"type": "object",
                         "properties": {"city": {"type": "string"}, ...},   # in signature order
                         "required": ["city"]}}                             # params w/o default
     Annotation -> JSON type: str "string", int "integer", float "number", bool "boolean",
     list "array", dict "object".
   * Raise TypeError AT DECORATION TIME (and register nothing) for: a parameter without an
     annotation, an unsupported annotation, or *args / **kwargs.

2. call_tool(name, arguments: dict) -> result
   What the agent loop runs when the model returns a tool call.
   * unknown name -> KeyError
   * bind with inspect.signature(...).bind(**arguments): missing/unexpected args -> TypeError
   * type-check every bound argument (defaults applied) with isinstance against its
     annotation -> TypeError naming the parameter. Two JSON realities:
       - an int is acceptable for a float parameter (JSON has one number type)
       - a bool is NOT acceptable for an int/float parameter (bool subclasses int!)
   * then call the function and return its result.

Interviewers look for
---------------------
* Decorators don't have to wrap: registration decorators (Flask routes, pytest fixtures,
  FastAPI endpoints, LangChain @tool) return the function unchanged.
* Functions are objects: you can attach attributes and introspect __name__, __doc__,
  signature, annotations.
* Validate at decoration time (import time) so a bad tool fails on startup, not mid-chat.
* Never trust model-produced arguments: validate before executing.
"""

import inspect
from collections.abc import Callable
from typing import Any

TOOLS: dict[str, Callable[..., Any]] = {}

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    name = func.__name__
    if name in TOOLS:
        raise ValueError(f"tool {name!r} already registered")
    sig = inspect.signature(func)
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"{name}: *args/**kwargs are not allowed in tools")
        if param.annotation is param.empty:
            raise TypeError(f"{name}: parameter {param.name!r} needs a type annotation")
        json_type = _JSON_TYPES.get(param.annotation)
        if json_type is None:
            raise TypeError(f"{name}: unsupported annotation for {param.name!r}")
        properties[param.name] = {"type": json_type}
        if param.default is param.empty:
            required.append(param.name)

    doc = inspect.getdoc(func) or ""
    description = next((ln.strip() for ln in doc.splitlines() if ln.strip()), "")
    func.schema = {  # type: ignore[attr-defined]
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }
    TOOLS[name] = func
    return func


def _matches(value: Any, annotation: type) -> bool:
    if isinstance(value, bool) and annotation is not bool:
        return False
    if annotation is float:
        return isinstance(value, (int, float))
    return isinstance(value, annotation)


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    func = TOOLS[name]
    sig = inspect.signature(func)
    bound = sig.bind(**arguments)
    bound.apply_defaults()
    for pname, value in bound.arguments.items():
        annotation = sig.parameters[pname].annotation
        if not _matches(value, annotation):
            raise TypeError(f"{name}: argument {pname!r} must be {annotation.__name__}")
    return func(*bound.args, **bound.kwargs)
