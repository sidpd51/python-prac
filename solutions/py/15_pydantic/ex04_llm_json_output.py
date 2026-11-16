"""
Exercise 15.4 - Parsing messy LLM JSON into a model               level: ★★☆
topics: pydantic.api_validation, pydantic.model_validate, pydantic.validation,
        pydantic.serialization

Problem
-------
LLMs asked for "JSON only" still reply with things like:

    Sure! Here is the triage:
    ```json
    {"category": "bug", "priority": 2, "summary": "Login fails"}
    ```
    Let me know if you need anything else.

`TicketTriage` and `LLMOutputError` are given.

1. extract_json(text) -> str
     * if the text contains a ``` fenced block (``` or ```json, any case), search only
       inside the FIRST fenced block; otherwise search the whole text
     * return the FIRST complete JSON object or array, as the exact substring of the input
       (prose before it and any trailing text after it are dropped)
     * braces inside prose ("use {name} here") or inside JSON strings must not confuse you
       -> hint: json.JSONDecoder().raw_decode(s, idx) parses ONE value starting at idx
          and tells you where it ended
     * nothing parseable -> raise LLMOutputError

2. parse_llm_output(text, model) -> instance of `model`
     * extract_json, then validate with model.model_validate_json (pydantic.ValidationError
       propagates unchanged; LLMOutputError propagates unchanged)
     * generic: must work for any BaseModel subclass

3. error_feedback(err: ValidationError) -> str
     A message we send BACK to the LLM so it can fix its output. Exact format:

         Your JSON failed validation with 2 error(s):
         - priority: Input should be less than or equal to 5 (got 9)
         - tags.1: Input should be a valid string (got 3)
         Return ONLY the corrected JSON object.

     * one line per error, in pydantic's order; loc parts joined with "." ; an empty loc
       is written "(root)"
     * "msg" is pydantic's error message
     * append " (got <repr(input)>)" ONLY when the input is a scalar (str/int/float/bool/
       None) AND the error type is not "missing" (missing fields have no useful input)

Interviewers look for
---------------------
* Never `json.loads` LLM output blindly: extraction + schema validation + a structured
  error you can feed back (that's the core of "structured output" libraries).
* model_validate_json validates directly from the JSON string (one pass, JSON-mode
  coercion rules) vs json.loads -> model_validate (two passes, python-mode rules).
* ValidationError.errors() gives machine-readable dicts: type, loc, msg, input.
* Don't echo huge inputs back into the prompt (token cost) - only short scalars.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

_FENCE = re.compile(r"```[a-zA-Z]*[ \t]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


class LLMOutputError(ValueError):
    """The LLM response contains no parseable JSON."""


class TicketTriage(BaseModel):
    category: Literal["bug", "billing", "feature_request", "other"]
    priority: int = Field(ge=1, le=5)
    summary: str = Field(min_length=1, max_length=200)
    tags: list[str] = Field(default_factory=list)
    needs_human: bool = False


def extract_json(text: str) -> str:
    fence = _FENCE.search(text)
    haystack = fence.group(1) if fence else text
    for idx, char in enumerate(haystack):
        if char not in "{[":
            continue
        try:
            _, end = _DECODER.raw_decode(haystack, idx)
        except json.JSONDecodeError:
            continue
        return haystack[idx:end]
    raise LLMOutputError("no JSON object or array found in LLM output")


def parse_llm_output[M: BaseModel](text: str, model: type[M]) -> M:
    return model.model_validate_json(extract_json(text))


def error_feedback(err: ValidationError) -> str:
    errors = err.errors(include_url=False)
    lines = [f"Your JSON failed validation with {len(errors)} error(s):"]
    for e in errors:
        loc = ".".join(str(part) for part in e["loc"]) or "(root)"
        line = f"- {loc}: {e['msg']}"
        value = e.get("input")
        if e["type"] != "missing" and (value is None or isinstance(value, (str, int, float))):
            line += f" (got {value!r})"
        lines.append(line)
    lines.append("Return ONLY the corrected JSON object.")
    return "\n".join(lines)
