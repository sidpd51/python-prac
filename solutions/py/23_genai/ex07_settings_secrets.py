"""
Exercise 23.7 - Config from the environment, secrets that never leak   level: ★★☆
topics: genai.config, genai.env_vars, genai.secrets, genai.logging_tracking

Problem
-------
12-factor config: settings come from environment variables (optionally a .env file in
dev), are validated at startup, and secrets never end up in logs, reprs or tracebacks.

1. class Settings(BaseSettings) - the fields are given; you write the validators.
   env prefix "APP_" (APP_API_KEY, APP_MODEL, ...), case-insensitive, no .env file by
   default (pass `_env_file=path` to read one), unknown APP_* variables ignored.
   * api_key: SecretStr (required)      * model: str = "small-chat"
   * temperature: float in [0, 2]       * max_tokens: int > 0      * context_window: int > 0
   * base_url: str                      * log_level: str = "INFO"
   Validators to write:
   * log_level (field validator): case-insensitive, stored upper-case, must be one of
     DEBUG/INFO/WARNING/ERROR/CRITICAL
   * check (model validator, mode="after"), runs on every instantiation:
       - max_tokens must be < context_window (leave room for the prompt)
       - base_url must start with "https://" - except http://localhost / http://127.0.0.1
         for local dev
   Invalid config -> pydantic.ValidationError at startup (fail fast, not at first request).

2. get_settings() -> Settings - cached with functools.lru_cache (one Settings object per
   process; tests call get_settings.cache_clear()).

3. class RedactingFilter(logging.Filter)
   RedactingFilter(secrets: Iterable[str]) - attach it to a HANDLER. filter(record) renders
   the message (record.getMessage()), replaces every given secret (ignore empty strings)
   and anything that looks like an API key (regex: sk-[A-Za-z0-9_-]{8,}) with "***",
   stores the result in record.msg and sets record.args = None. Always returns True
   (redact, never drop records).

4. class JsonFormatter(logging.Formatter) - structured logs, one JSON object per line:
   format(record) -> json.dumps({"level": record.levelname, "logger": record.name,
   "message": record.getMessage(), **extras}) where extras are the attributes passed via
   `extra={...}` (= record.__dict__ keys that a plain LogRecord doesn't have).
   If record.exc_info is set, add "exc": self.formatException(record.exc_info).
   Values that aren't JSON-serialisable are converted with str().

Interviewers look for
---------------------
* SecretStr: repr/str/model_dump show "**********"; .get_secret_value() only at the call site
  that needs it (the HTTP header).
* Fail fast: validate config at startup; one cached Settings, injected where needed.
* Precedence: init kwargs > env vars > .env file > defaults. Never commit .env; secrets come
  from a secret manager / CI secret store in prod.
* Filters on a LOGGER don't see records propagated from child loggers; filters on the
  HANDLER see everything that handler emits - that's where redaction belongs.
* Structured (JSON) logs with request_id/model/latency/tokens fields -> searchable,
  aggregatable, alertable; same fields go to error tracking (Sentry) as tags.
"""

import json
import logging
import re
from collections.abc import Iterable
from functools import cache, lru_cache
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOCAL_HTTP = ("http://localhost", "http://127.0.0.1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=None, extra="ignore", case_sensitive=False)

    api_key: SecretStr
    model: str = "small-chat"
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, gt=0)
    context_window: int = Field(8192, gt=0)
    base_url: str = "https://api.example.com/v1"
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in LEVELS:
            raise ValueError(f"log_level must be one of {LEVELS}")
        return level

    @model_validator(mode="after")
    def check(self) -> Self:
        if self.max_tokens >= self.context_window:
            raise ValueError("max_tokens must be smaller than context_window")
        if not (self.base_url.startswith("https://") or self.base_url.startswith(LOCAL_HTTP)):
            raise ValueError("base_url must use https (http only for localhost)")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]
        self._key_pattern = re.compile(r"sk-[A-Za-z0-9_-]{8,}")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self._secrets:
            message = message.replace(secret, "***")
        record.msg = self._key_pattern.sub("***", message)
        record.args = None
        return True


@cache
def _standard_attrs() -> frozenset[str]:
    # everything a bare LogRecord has; anything else on a record came from `extra=`
    return frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _standard_attrs():
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
