"""
Exercise 23.5 - Structured output: extract, validate, repair-and-retry   level: ★★☆
topics: genai.structured_output, genai.structured_data, genai.regex

Problem
-------
You ask an LLM for JSON. You get JSON wrapped in ```json fences, JSON with chatty prose
around it, or JSON that's syntactically fine but violates your schema. Make it robust.
The pydantic schema `Answer` (+ `Citation`) is given below.

1. extract_json(text) -> str
   * if the text contains a fenced code block (``` optionally followed by a language tag
     such as json, then a newline), return the content of the FIRST such block, stripped
   * otherwise return the substring from the first "{" to the last "}" (inclusive)
   * neither -> ValueError
       extract_json('Sure!\\n```json\\n{"a": 1}\\n```\\nHope it helps') == '{"a": 1}'
       extract_json('Result: {"a": {"b": 2}} - done') == '{"a": {"b": 2}}'

2. parse_answer(text) -> Answer
   extract_json + Answer.model_validate_json. Let ValueError / pydantic.ValidationError
   propagate (note: ValidationError is a subclass of ValueError).

3. generate_structured(llm, prompt, schema, *, max_attempts=3) -> schema instance
   llm: a callable  llm(messages: list[dict]) -> str  (chat-style messages).
   * attempt 1 sends exactly:
       [{"role": "system", "content": SYSTEM_TEMPLATE.format(schema=json.dumps(schema.model_json_schema()))},
        {"role": "user", "content": prompt}]
   * on failure (no JSON found / invalid JSON / schema violation) append the bad reply as
     {"role": "assistant", "content": reply} and then
     {"role": "user", "content": REPAIR_TEMPLATE.format(error=str(exc))}
     and call llm again with the WHOLE conversation so far
   * pass a fresh list each call (a copy) - the caller may keep references
   * success -> return the validated model instance
   * after max_attempts failures -> raise StructuredOutputError (given) chained from the
     last error (`raise ... from exc`)

Interviewers look for
---------------------
* Validate at the boundary: LLM output is untrusted input, exactly like a user's HTTP body.
* Feeding the validation error back lets the model self-correct (most failures fix in 1 retry);
  cap the attempts - each retry costs latency and tokens.
* Prefer provider-native structured output / tool calling / JSON mode when available,
  but still validate - "JSON mode" guarantees syntax, not your schema.
* extra="forbid" catches hallucinated fields; Field constraints catch out-of-range values.
"""

import json
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__stub_keep__ = ["Citation", "Answer", "StructuredOutputError"]

SYSTEM_TEMPLATE = "Reply ONLY with a JSON object matching this JSON schema:\n{schema}"
REPAIR_TEMPLATE = (
    "Your previous reply was invalid:\n{error}\nReply again with ONLY the corrected JSON object."
)

M = TypeVar("M", bound=BaseModel)
LLM = Callable[[list[dict[str, str]]], str]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1)
    quote: str


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = []


class StructuredOutputError(Exception):
    """The LLM never produced output matching the schema."""


def extract_json(text: str) -> str:
    if match := re.search(r"```[\w+-]*[ \t]*\n(.*?)```", text, re.DOTALL):
        return match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object found in LLM output")
    return text[start : end + 1]


def parse_answer(text: str) -> Answer:
    return Answer.model_validate_json(extract_json(text))


def generate_structured(llm: LLM, prompt: str, schema: type[M], *, max_attempts: int = 3) -> M:
    schema_json = json.dumps(schema.model_json_schema())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(schema=schema_json)},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    for _ in range(max_attempts):
        reply = llm(list(messages))
        try:
            return schema.model_validate_json(extract_json(reply))
        except (ValidationError, ValueError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": REPAIR_TEMPLATE.format(error=str(exc))})
    raise StructuredOutputError(f"no valid output after {max_attempts} attempts") from last_error
