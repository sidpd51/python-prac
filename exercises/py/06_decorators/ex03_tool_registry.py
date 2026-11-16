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

from collections.abc import Callable
from typing import Any

TOOLS: dict[str, Callable[..., Any]] = {}


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    raise NotImplementedError


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    raise NotImplementedError
