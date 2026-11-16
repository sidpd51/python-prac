"""
Exercise 15.5 - App configuration with pydantic-settings          level: ★★☆
topics: pydantic.settings, pydantic.nested, pydantic.custom_validators,
        pydantic.model_validator, pydantic.model_dump

Problem
-------
Configuration for a RAG service, read from environment variables (and optionally a
.env file). The fields and model_config of `LLMConfig` / `AppSettings` are given -
read them: env_prefix="RAG_", env_nested_delimiter="__", so e.g.

    RAG_ENV=prod  RAG_API_KEY=sk-...  RAG_LLM__MODEL=gpt-4o  RAG_LLM__TEMPERATURE=0.0
    RAG_ALLOWED_MODELS="gpt-4o, gpt-4o-mini"

Implement:

1. AppSettings.split_csv - field_validator("allowed_models", mode="before")
     * a str value is a comma-separated list: split, strip each item, drop empties
       ("gpt-4o, gpt-4o-mini ," -> ["gpt-4o", "gpt-4o-mini"])
     * a str that starts with "[" is a JSON array -> json.loads it
     * any other value (already a list) passes through unchanged
     (the field is annotated NoDecode so pydantic-settings does NOT try to json-decode it)

2. AppSettings.check_consistency - model_validator(mode="after"), ValueError if:
     * llm.model is not in allowed_models (any env)
     * env == "prod" and api_key is missing
     * env == "prod" and debug is True

3. AppSettings.public_dict() -> dict
     * JSON-mode dump safe to log / show on a /config endpoint:
       secrets come out as "**********" (that's what SecretStr gives in JSON mode),
       and the password inside database_url (scheme://user:PASSWORD@host/...) is
       replaced by "***". URLs without a password are unchanged.

4. get_settings() -> AppSettings
     * build the settings ONCE per process and return the same object afterwards
       (tests call get_settings.cache_clear() - so use functools.cache / lru_cache)

Interviewers look for
---------------------
* Precedence: init kwargs > environment variables > .env file > field defaults.
* Env values are strings; pydantic coerces "false"/"0"/"no" -> False, "30" -> 30.0.
  Complex types (list/dict/models) are JSON-decoded unless you opt out (NoDecode).
* Nested config via env_nested_delimiter, and validating ALL config at startup
  ("fail fast") instead of crashing on first use deep inside a request.
* SecretStr: repr/str/JSON dump are masked; .get_secret_value() is explicit.
* Settings as a cached singleton (dependency injection in FastAPI: Depends(get_settings)),
  and why tests must clear that cache.
"""

import functools
import json
import re
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_URL_PASSWORD = re.compile(r"(?P<head>://[^:/@]+:)[^@]*(?=@)")


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout_s: float = Field(default=30.0, gt=0)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_nested_delimiter="__", extra="ignore"
    )

    env: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    api_key: SecretStr | None = None
    database_url: str = "sqlite:///./dev.db"
    allowed_models: Annotated[list[str], NoDecode] = ["gpt-4o-mini"]
    llm: LLMConfig = LLMConfig()

    @field_validator("allowed_models", mode="before")
    @classmethod
    def split_csv(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.llm.model not in self.allowed_models:
            raise ValueError(f"llm.model {self.llm.model!r} not in allowed_models")
        if self.env == "prod":
            if self.api_key is None:
                raise ValueError("api_key is required in prod")
            if self.debug:
                raise ValueError("debug must be off in prod")
        return self

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["database_url"] = _URL_PASSWORD.sub(r"\g<head>***", self.database_url)
        return data


@functools.cache
def get_settings() -> AppSettings:
    return AppSettings()
