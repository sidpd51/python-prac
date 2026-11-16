import json
import os

import pytest
from pydantic import SecretStr, ValidationError

from study.loader import load

ex = load(__file__, "ex05_app_settings")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.upper().startswith("RAG_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    ex.get_settings.cache_clear()
    yield monkeypatch
    ex.get_settings.cache_clear()


def _env(monkeypatch, **values):
    for k, v in values.items():
        monkeypatch.setenv(k, v)


@pytest.mark.topic("py.pydantic.settings")
def test_defaults():
    s = ex.AppSettings()
    assert (s.env, s.debug, s.api_key, s.allowed_models) == ("dev", False, None, ["gpt-4o-mini"])
    assert s.llm.temperature == 0.2


@pytest.mark.topic("py.pydantic.settings", "py.pydantic.nested")
def test_reads_prefixed_and_nested_env(clean_env):
    _env(
        clean_env,
        RAG_ENV="staging",
        RAG_DEBUG="true",
        RAG_ALLOWED_MODELS="gpt-4o, gpt-4o-mini ,",
        RAG_LLM__MODEL="gpt-4o",
        RAG_LLM__TEMPERATURE="0",
        RAG_LLM__TIMEOUT_S="12.5",
    )
    s = ex.AppSettings()
    assert s.env == "staging" and s.debug is True
    assert s.allowed_models == ["gpt-4o", "gpt-4o-mini"]
    assert (s.llm.model, s.llm.temperature, s.llm.timeout_s) == ("gpt-4o", 0.0, 12.5)


@pytest.mark.topic("py.pydantic.custom_validators", "py.pydantic.settings")
def test_allowed_models_json_array_and_list(clean_env):
    _env(clean_env, RAG_ALLOWED_MODELS='["gpt-4o-mini", "o3"]')
    assert ex.AppSettings().allowed_models == ["gpt-4o-mini", "o3"]
    assert ex.AppSettings(allowed_models=["gpt-4o-mini"]).allowed_models == ["gpt-4o-mini"]


@pytest.mark.topic("py.pydantic.settings")
def test_init_kwargs_beat_env(clean_env):
    _env(clean_env, RAG_DEBUG="true", RAG_DATABASE_URL="postgresql://db/rag")
    s = ex.AppSettings(debug=False)
    assert s.debug is False
    assert s.database_url == "postgresql://db/rag"


@pytest.mark.topic("py.pydantic.settings")
def test_env_beats_dotenv_file(clean_env, tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text("RAG_ENV=staging\nRAG_DATABASE_URL=sqlite:///file.db\n")
    _env(clean_env, RAG_DATABASE_URL="sqlite:///from-env.db")
    s = ex.AppSettings(_env_file=env_file)
    assert s.env == "staging"
    assert s.database_url == "sqlite:///from-env.db"


@pytest.mark.topic("py.pydantic.settings", "py.pydantic.validation")
def test_invalid_nested_value_fails_fast(clean_env):
    _env(clean_env, RAG_LLM__TEMPERATURE="5", RAG_ALLOWED_MODELS="gpt-4o-mini")
    with pytest.raises(ValidationError) as info:
        ex.AppSettings()
    assert info.value.errors()[0]["loc"] == ("llm", "temperature")


@pytest.mark.topic("py.pydantic.model_validator", "py.pydantic.settings")
@pytest.mark.parametrize(
    "env",
    [
        {"RAG_LLM__MODEL": "gpt-5-preview"},  # not allowed
        {"RAG_ENV": "prod"},  # no api key
        {"RAG_ENV": "prod", "RAG_API_KEY": "sk-1", "RAG_DEBUG": "1"},  # debug in prod
    ],
)
def test_consistency_rules(clean_env, env):
    _env(clean_env, **env)
    with pytest.raises(ValidationError):
        ex.AppSettings()


@pytest.mark.topic("py.pydantic.settings", "py.pydantic.model_validator")
def test_prod_ok_and_secret_masked(clean_env):
    _env(clean_env, RAG_ENV="prod", RAG_API_KEY="sk-live-123")
    s = ex.AppSettings()
    assert isinstance(s.api_key, SecretStr)
    assert s.api_key.get_secret_value() == "sk-live-123"
    assert "sk-live-123" not in repr(s)


@pytest.mark.topic("py.pydantic.model_dump", "py.pydantic.settings")
def test_public_dict_masks_secrets(clean_env):
    _env(
        clean_env,
        RAG_ENV="prod",
        RAG_API_KEY="sk-live-123",
        RAG_DATABASE_URL="postgresql://rag:hunter2@db.internal:5432/rag",
    )
    data = ex.AppSettings().public_dict()
    assert data["api_key"] == "**********"
    assert data["database_url"] == "postgresql://rag:***@db.internal:5432/rag"
    assert data["llm"] == {"model": "gpt-4o-mini", "temperature": 0.2, "timeout_s": 30.0}
    dumped = json.dumps(data)  # must be JSON-serialisable
    assert "sk-live-123" not in dumped and "hunter2" not in dumped


@pytest.mark.topic("py.pydantic.model_dump")
def test_public_dict_url_without_password_unchanged(clean_env):
    _env(clean_env, RAG_DATABASE_URL="postgresql://rag@db/rag")
    assert ex.AppSettings().public_dict()["database_url"] == "postgresql://rag@db/rag"


@pytest.mark.topic("py.pydantic.settings")
def test_get_settings_is_cached(clean_env):
    _env(clean_env, RAG_ENV="staging")
    first = ex.get_settings()
    clean_env.setenv("RAG_ENV", "dev")
    assert ex.get_settings() is first
    assert ex.get_settings().env == "staging"
    ex.get_settings.cache_clear()
    assert ex.get_settings().env == "dev"
