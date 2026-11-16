"""
Milestone M0.1 - configuration & logging
topics: py.pydantic.settings, py.genai.config, py.genai.secrets, py.genai.env_vars,
        py.testing.logging_module, py.genai.logging_tracking

Build
-----
1. `Settings` (pydantic-settings) - env prefix DOCMIND_, optional `.env` file, extra env ignored.
   Fields + defaults are given. Add validation (model_validator, mode="after"):
     * chunk_overlap must be < chunk_size                         -> ValueError
     * embedder == "http" requires embedding_url to be set         -> ValueError
   The API key is a SecretStr: it must never appear in repr()/str()/logs.

2. `get_settings()` - cached (one Settings per process); tests call `get_settings.cache_clear()`.

3. `configure_logging(level="INFO", json=False)` - configures the "docmind" logger (NOT root):
     * idempotent: calling it twice must not add a second handler (duplicate log lines bug)
     * json=True -> each record is one JSON object per line with keys
       "ts", "level", "logger", "msg" (+ "exc" when exc_info is present)
     * returns the "docmind" logger
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCMIND_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    index_dir: Path = Path(".docmind/index")
    chunk_size: int = Field(800, gt=0)
    chunk_overlap: int = Field(100, ge=0)
    embedder: Literal["hashing", "http"] = "hashing"
    embedding_dim: int = Field(256, gt=0)
    embedding_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    embed_batch_size: int = Field(64, gt=0)
    embed_concurrency: int = Field(4, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.embedder == "http" and not self.embedding_url:
            raise ValueError("embedder='http' requires DOCMIND_EMBEDDING_URL")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_HANDLER_NAME = "docmind-handler"


def configure_logging(level: str = "INFO", json: bool = False) -> logging.Logger:
    logger = logging.getLogger("docmind")
    logger.setLevel(level)
    handler = next((h for h in logger.handlers if h.get_name() == _HANDLER_NAME), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        logger.addHandler(handler)
    handler.setFormatter(
        JsonFormatter() if json else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    return logger
