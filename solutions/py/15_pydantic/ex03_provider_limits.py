"""
Exercise 15.3 - Per-provider request schemas (Field constraints)    level: ★★☆
topics: pydantic.field, pydantic.constraints, pydantic.required_optional,
        pydantic.validation, pydantic.json_schema

Problem
-------
An LLM gateway accepts one request shape but every provider has different limits
(max output tokens, temperature range, number of stop sequences, seed support).
Build the request model per provider.

request_model_for(limits: ProviderLimits) -> type[BaseModel]
  Model class name: f"{limits.provider.capitalize()}Request"  (e.g. "OpenaiRequest").
  Config: unknown fields are an ERROR (a typo like "temprature" must not be silently
  ignored - that's pydantic's default behaviour!).

  field        type / default               rules
  -----------  ---------------------------  ---------------------------------------------
  model        str, REQUIRED                whitespace stripped, then min length 1
  prompt       str, REQUIRED                min length 1
  max_tokens   int, REQUIRED                > 0 and <= limits.max_output_tokens;
                                            STRICT: "100", 100.0 and True are rejected
  temperature  float = min(1.0, max_temp)   0 <= t <= limits.max_temperature
  top_p        float | None = None          0 < p <= 1 when given
  stop         list[str] = []               at most limits.max_stop_sequences items,
                                            each item non-empty; no shared mutable default
  seed         int | None = None            ONLY present if limits.supports_seed;
                                            otherwise sending seed is an "extra" error

  Every field has a short `description` (it shows up in the JSON schema / API docs).
  Calling request_model_for twice with equal limits returns the SAME class object
  (building models is not free - cache them).

Use Field(...) / Annotated constraints - NOT hand-written validators.

Interviewers look for
---------------------
* Required vs optional vs nullable: `x: int` (required), `x: int = 0` (optional),
  `x: int | None` (required but may be null!), `x: int | None = None` (optional, nullable).
* Lax vs strict: by default pydantic coerces "100" -> 100 and True -> 1. For token limits
  and money, strict=True is often what you want.
* extra="forbid" for request bodies catches client typos early.
* Constraints are declarative -> they appear in the JSON schema (maximum, minLength...),
  hand-written validators don't.
* create_model() / classes built in a function for config-driven schemas.
"""

import functools
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, create_model


@dataclass(frozen=True)
class ProviderLimits:
    provider: str
    max_output_tokens: int
    max_temperature: float
    max_stop_sequences: int
    supports_seed: bool = True


@functools.cache
def request_model_for(limits: ProviderLimits) -> type[BaseModel]:
    fields: dict[str, Any] = {
        "model": (
            Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)],
            Field(description="Model identifier, e.g. 'gpt-4o-mini'"),
        ),
        "prompt": (str, Field(min_length=1, description="User prompt")),
        "max_tokens": (
            int,
            Field(
                gt=0,
                le=limits.max_output_tokens,
                strict=True,
                description="Maximum number of tokens to generate",
            ),
        ),
        "temperature": (
            float,
            Field(
                default=min(1.0, limits.max_temperature),
                ge=0,
                le=limits.max_temperature,
                description="Sampling temperature",
            ),
        ),
        "top_p": (float | None, Field(default=None, gt=0, le=1, description="Nucleus sampling")),
        "stop": (
            list[Annotated[str, Field(min_length=1)]],
            Field(
                default_factory=list,
                max_length=limits.max_stop_sequences,
                description="Stop sequences",
            ),
        ),
    }
    if limits.supports_seed:
        fields["seed"] = (int | None, Field(default=None, description="Sampling seed"))
    return create_model(
        f"{limits.provider.capitalize()}Request",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
