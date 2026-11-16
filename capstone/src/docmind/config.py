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

import logging

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings (env prefix DOCMIND_, optional .env file, ignore unknown env vars).

    Fields to define (name: type = default):
        data_dir: Path = Path("data")
        index_dir: Path = Path(".docmind/index")
        chunk_size: int = 800            (> 0)
        chunk_overlap: int = 100         (>= 0)
        embedder: "hashing" | "http" = "hashing"
        embedding_dim: int = 256         (> 0)
        embedding_url: str | None = None
        embedding_model: str = "text-embedding-3-small"
        embedding_api_key: SecretStr | None = None
        embed_batch_size: int = 64       (> 0)
        embed_concurrency: int = 4       (> 0)
        log_level: "DEBUG" | "INFO" | "WARNING" | "ERROR" = "INFO"
    """

    def __init__(self, **data):  # TODO: delete this __init__ once fields + validator exist
        raise NotImplementedError


def get_settings() -> Settings:
    raise NotImplementedError


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        raise NotImplementedError


def configure_logging(level: str = "INFO", json: bool = False) -> logging.Logger:
    raise NotImplementedError
