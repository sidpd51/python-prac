import json
import logging

import pytest
from docmind import config
from pydantic import ValidationError


def _clear_settings_cache():
    # get_settings should be cached (functools.lru_cache/cache); tolerate it not being yet
    getattr(config.get_settings, "cache_clear", lambda: None)()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for key in list(__import__("os").environ):
        if key.startswith("DOCMIND_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)  # no stray .env file
    _clear_settings_cache()
    yield
    _clear_settings_cache()
    logger = logging.getLogger("docmind")
    for h in list(logger.handlers):
        logger.removeHandler(h)


@pytest.mark.topic("py.pydantic.settings", "py.genai.env_vars")
def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DOCMIND_CHUNK_SIZE", "300")
    monkeypatch.setenv("DOCMIND_CHUNK_OVERLAP", "30")
    s = config.Settings()
    assert (s.chunk_size, s.chunk_overlap) == (300, 30)


@pytest.mark.topic("py.pydantic.model_validator", "py.genai.config")
def test_cross_field_validation(monkeypatch):
    with pytest.raises(ValidationError):
        config.Settings(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValidationError):
        config.Settings(embedder="http")
    assert config.Settings(embedder="http", embedding_url="http://x/embed").embedder == "http"


@pytest.mark.topic("py.genai.secrets")
def test_secret_never_leaks(monkeypatch):
    monkeypatch.setenv("DOCMIND_EMBEDDING_API_KEY", "sk-very-secret")
    s = config.Settings()
    assert "sk-very-secret" not in repr(s) and "sk-very-secret" not in str(s)
    assert s.embedding_api_key.get_secret_value() == "sk-very-secret"


@pytest.mark.topic("py.genai.config", "py.functions.higher_order")
def test_get_settings_cached():
    assert config.get_settings() is config.get_settings()


@pytest.mark.topic("py.testing.logging_module")
def test_configure_logging_idempotent():
    logger = config.configure_logging("DEBUG")
    config.configure_logging("DEBUG")
    assert logger.name == "docmind"
    assert len(logger.handlers) == 1, "calling configure_logging twice duplicated the handler"
    assert logger.level == logging.DEBUG
    assert logging.getLogger().handlers == [] or all(
        h not in logging.getLogger().handlers for h in logger.handlers
    ), "configure the 'docmind' logger, not the root logger"


@pytest.mark.topic("py.genai.logging_tracking", "py.testing.logging_module")
def test_json_logging(capsys):
    logger = config.configure_logging("INFO", json=True)
    logger.info("hello %s", "world")
    try:
        raise KeyError("boom")
    except KeyError:
        logger.exception("failed")
    lines = [json.loads(ln) for ln in capsys.readouterr().err.strip().splitlines()]
    assert lines[0]["msg"] == "hello world" and lines[0]["level"] == "INFO"
    assert lines[0]["logger"] == "docmind" and "ts" in lines[0]
    assert "KeyError" in lines[1]["exc"]
