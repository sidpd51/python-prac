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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]


def tool_definition(
    model: type[BaseModel], name: str | None = None, description: str | None = None
) -> dict[str, Any]:
    raise NotImplementedError


def tool_from_function(fn: Callable[..., Any]) -> Tool:
    raise NotImplementedError


def invoke_tool(tool: Tool, arguments: str | dict[str, Any]) -> Any:
    raise NotImplementedError
