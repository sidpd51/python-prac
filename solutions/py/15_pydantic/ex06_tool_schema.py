"""
Exercise 15.6 - LLM tool definitions from models and functions     level: ★★★
topics: pydantic.json_schema, pydantic.field, pydantic.api_validation, pydantic.models,
        pydantic.model_validate

Problem
-------
Function-calling APIs want each tool as
    {"type": "function",
     "function": {"name": ..., "description": ..., "parameters": <JSON schema>}}
and many providers reject schemas that contain "$ref"/"$defs" and want every object
to forbid unknown keys.

1. tool_definition(model, name=None, description=None) -> dict
     * name defaults to the class name in snake_case:
         "GetWeather" -> "get_weather", "SearchKBArticles" -> "search_kb_articles"
     * description defaults to inspect.cleandoc(model.__doc__), or "" if no docstring
     * parameters = model.model_json_schema(), post-processed:
         - every {"$ref": "#/$defs/X", ...} is replaced by the (processed) definition of X,
           merged with any sibling keys of the $ref (e.g. a "description");
           the top-level "$defs" is removed
         - every "title" KEYWORD is removed - but a *property* that is literally named
           "title" (a key inside a "properties" mapping) must survive!
         - every schema with "type": "object" gets "additionalProperties": False unless it
           already has an "additionalProperties" entry (dict[str, X] fields do)
         - values under "default", "examples", "enum", "const" are data: copy untouched
         - a self-referencing (recursive) model cannot be inlined -> ValueError
     * the input schema must not be mutated in any way that changes later calls

2. tool_from_function(fn) -> Tool        (Tool is a given dataclass)
     * build an args model with pydantic.create_model, one field per parameter:
       annotation from the signature (Annotated[..., Field(description=...)] metadata kept),
       parameters with defaults are optional, without defaults required
     * the args model forbids extra keys
     * a parameter without annotation, or *args / **kwargs -> TypeError
     * Tool.name = fn.__name__, Tool.description = cleaned docstring or ""

3. invoke_tool(tool, arguments) -> Any
     * arguments: the JSON string from the LLM, or an already-parsed dict
     * validate with the args model (ValidationError propagates), then call
       tool.fn(**kwargs) with the VALIDATED values - nested BaseModel parameters must
       arrive as model instances, not dicts (so don't use model_dump() here!)

Interviewers look for
---------------------
* Pydantic is the schema source of truth: one model gives validation, JSON schema for
  the LLM, and typed parsing of the LLM's arguments (what Instructor / OpenAI Agents SDK /
  LangChain @tool do under the hood).
* Descriptions on fields are prompt engineering - the model reads them.
* $ref inlining with cycle detection; distinguishing schema KEYWORDS from property NAMES.
* LLM-produced arguments are untrusted input: validate before executing.
"""

import inspect
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

_DATA_KEYS = frozenset({"default", "examples", "enum", "const"})


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]


def _snake(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _inline(node: Any, defs: dict[str, Any], stack: frozenset[str]) -> Any:
    if isinstance(node, list):
        return [_inline(item, defs, stack) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        ref_name = node["$ref"].rsplit("/", 1)[-1]
        if ref_name in stack:
            raise ValueError(f"recursive schema via {ref_name!r} cannot be inlined")
        resolved = _inline(defs[ref_name], defs, stack | {ref_name})
        siblings = _inline({k: v for k, v in node.items() if k != "$ref"}, defs, stack)
        return {**resolved, **siblings}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$defs", "title"):
            continue
        if key == "properties":
            out[key] = {prop: _inline(sub, defs, stack) for prop, sub in value.items()}
        elif key in _DATA_KEYS:
            out[key] = value
        else:
            out[key] = _inline(value, defs, stack)
    if out.get("type") == "object" and "additionalProperties" not in out:
        out["additionalProperties"] = False
    return out


def tool_definition(
    model: type[BaseModel], name: str | None = None, description: str | None = None
) -> dict[str, Any]:
    schema = model.model_json_schema()
    parameters = _inline(schema, schema.get("$defs", {}), frozenset())
    if description is None:
        description = inspect.cleandoc(model.__doc__) if model.__doc__ else ""
    return {
        "type": "function",
        "function": {
            "name": name or _snake(model.__name__),
            "description": description,
            "parameters": parameters,
        },
    }


def tool_from_function(fn: Callable[..., Any]) -> Tool:
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"{fn.__name__}: *args/**kwargs can't be described to an LLM")
        if param.name not in hints:
            raise TypeError(f"{fn.__name__}: parameter {param.name!r} needs a type annotation")
        default = ... if param.default is param.empty else param.default
        fields[param.name] = (hints[param.name], default)
    args_model = create_model(
        f"{fn.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields
    )
    doc = inspect.getdoc(fn) or ""
    return Tool(name=fn.__name__, description=doc, args_model=args_model, fn=fn)


def invoke_tool(tool: Tool, arguments: str | dict[str, Any]) -> Any:
    if isinstance(arguments, str):
        validated = tool.args_model.model_validate_json(arguments)
    else:
        validated = tool.args_model.model_validate(arguments)
    kwargs = {name: getattr(validated, name) for name in type(validated).model_fields}
    return tool.fn(**kwargs)
