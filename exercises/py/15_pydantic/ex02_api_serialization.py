"""
Exercise 15.2 - camelCase API payloads: aliases, model_dump, round-trips   level: ★☆☆
topics: pydantic.serialization, pydantic.model_dump, pydantic.model_validate,
        pydantic.api_validation

Problem
-------
An embedding service speaks camelCase JSON (`jobId`, `modelName`, `createdAt`...),
our Python code uses snake_case. The fields + model_config of `EmbeddingJob` and
`JobUpdate` are given; read the config (alias_generator=to_camel, validate_by_name).

Implement on EmbeddingJob:

1. num_inputs   - a @computed_field property: len(input_texts). It must appear in dumps.

2. serialize_created_at - @field_serializer("created_at", when_used="json")
     * output ISO-8601 in UTC with a trailing "Z", seconds precision:
       "2024-05-01T12:00:00Z"
     * a naive datetime is assumed to already be UTC; an aware one is converted to UTC
     * (when_used="json": python-mode dumps keep the real datetime object)

3. to_api(self) -> dict
     * JSON-compatible dict for the HTTP body: camelCase keys, None values omitted,
       UUID -> str, datetime -> the "Z" string above, includes "numInputs"
     * `api_key` must NEVER be in the output (it is declared with exclude=True - check
       that you didn't defeat that)

4. to_db_row(self) -> dict
     * python-mode dump with snake_case keys, None values KEPT, real UUID/datetime objects

5. from_api(cls, data) -> EmbeddingJob   (classmethod)
     * data may be a dict (already-parsed JSON) -> use model_validate
     * or str/bytes (raw HTTP body)         -> use model_validate_json (one fast pass in
       Rust, no intermediate dict)
     * accepts camelCase or snake_case keys; invalid input -> pydantic.ValidationError

And a function:

6. patch_body(update: JobUpdate) -> dict
     * body for a PATCH request: camelCase, JSON mode, containing ONLY the fields the
       caller explicitly set - including fields explicitly set to None (that means
       "clear it" in our API). Fields left at their default must not be sent.

Interviewers look for
---------------------
* exclude_none vs exclude_unset: "not provided" and "explicitly null" are different
  things in a PATCH. exclude_none would make it impossible to clear a field.
* mode="python" vs mode="json": python mode keeps UUID/datetime objects; json mode gives
  only JSON-native types (what you'd put on the wire).
* by_alias=True on dump; validate_by_name/validate_by_alias on load.
* model_validate_json(raw) is faster and stricter-per-JSON than json.loads + model_validate.
* Secrets: Field(exclude=True) / SecretStr so they never leak into logs or payloads.
"""

from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, computed_field, field_serializer
from pydantic.alias_generators import to_camel

_API_CONFIG = ConfigDict(alias_generator=to_camel, validate_by_name=True, validate_by_alias=True)


class EmbeddingJob(BaseModel):
    model_config = _API_CONFIG
    job_id: UUID
    model_name: str
    input_texts: list[str]
    created_at: datetime
    dimensions: int | None = None
    user_tag: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)

    @computed_field
    @property
    def num_inputs(self) -> int:
        raise NotImplementedError

    @field_serializer("created_at", when_used="json")
    def serialize_created_at(self, value: datetime) -> str:
        raise NotImplementedError

    def to_api(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_db_row(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_api(cls, data: dict[str, Any] | str | bytes) -> Self:
        raise NotImplementedError


class JobUpdate(BaseModel):
    model_config = _API_CONFIG
    user_tag: str | None = None
    dimensions: int | None = None
    model_name: str | None = None


def patch_body(update: JobUpdate) -> dict[str, Any]:
    raise NotImplementedError
