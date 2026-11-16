import io
import json
import logging
from datetime import timedelta

import pydantic
import pytest

from study.loader import load

ex = load(__file__, "ex07_settings_secrets")

KEY = "sk-live-abcdef123456"
FIELDS = ["API_KEY", "MODEL", "TEMPERATURE", "MAX_TOKENS", "CONTEXT_WINDOW", "BASE_URL", "LOG_LEVEL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in FIELDS:
        monkeypatch.delenv(f"APP_{name}", raising=False)
        monkeypatch.delenv(f"app_{name.lower()}", raising=False)
    monkeypatch.setenv("APP_API_KEY", KEY)
    ex.get_settings.cache_clear()
    yield
    ex.get_settings.cache_clear()


@pytest.mark.topic("py.genai.config", "py.genai.env_vars")
def test_defaults_and_env_overrides(monkeypatch):
    s = ex.get_settings()
    assert (s.model, s.temperature, s.max_tokens, s.log_level) == ("small-chat", 0.2, 1024, "INFO")
    ex.get_settings.cache_clear()
    monkeypatch.setenv("APP_MODEL", "big-chat")
    monkeypatch.setenv("APP_TEMPERATURE", "0.7")
    monkeypatch.setenv("app_max_tokens", "2048")  # case-insensitive names
    monkeypatch.setenv("APP_UNRELATED", "ignored")
    s = ex.get_settings()
    assert (s.model, s.temperature, s.max_tokens) == ("big-chat", 0.7, 2048)
    assert isinstance(s.max_tokens, int), "env strings are parsed into the field types"


@pytest.mark.topic("py.genai.config")
def test_get_settings_is_cached(monkeypatch):
    first = ex.get_settings()
    monkeypatch.setenv("APP_MODEL", "changed")
    assert ex.get_settings() is first
    ex.get_settings.cache_clear()
    assert ex.get_settings().model == "changed"


@pytest.mark.topic("py.genai.secrets", "py.genai.env_vars")
def test_api_key_required(monkeypatch):
    monkeypatch.delenv("APP_API_KEY")
    with pytest.raises(pydantic.ValidationError):
        ex.get_settings()


@pytest.mark.topic("py.genai.secrets")
def test_secret_is_masked_everywhere():
    s = ex.get_settings()
    assert isinstance(s.api_key, pydantic.SecretStr)
    assert s.api_key.get_secret_value() == KEY
    for rendering in (repr(s), str(s), s.model_dump_json(), str(s.model_dump())):
        assert KEY not in rendering


@pytest.mark.topic("py.genai.config")
@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("APP_TEMPERATURE", "2.5"),
        ("APP_MAX_TOKENS", "0"),
        ("APP_MAX_TOKENS", "lots"),
        ("APP_LOG_LEVEL", "verbose"),
        ("APP_MAX_TOKENS", "8192"),  # == context_window
        ("APP_BASE_URL", "http://api.example.com/v1"),  # plain http in prod
    ],
)
def test_invalid_config_fails_fast(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(pydantic.ValidationError):
        ex.get_settings()


@pytest.mark.topic("py.genai.config")
def test_validators_accept_valid_edge_values(monkeypatch):
    monkeypatch.setenv("APP_LOG_LEVEL", " debug ")
    monkeypatch.setenv("APP_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("APP_MAX_TOKENS", "8191")
    s = ex.get_settings()
    assert s.log_level == "DEBUG"
    assert s.base_url == "http://localhost:8000/v1"


@pytest.mark.topic("py.genai.config", "py.genai.env_vars")
def test_env_file_and_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("APP_MODEL=from-file\nAPP_TEMPERATURE=1.0\n", encoding="utf-8")
    monkeypatch.setenv("APP_TEMPERATURE", "0.3")  # real env beats .env
    s = ex.Settings(_env_file=env_file)
    assert s.model == "from-file"
    assert s.temperature == 0.3
    assert ex.Settings(_env_file=env_file, model="kwarg").model == "kwarg"  # init beats all


def _logger_with(handler, name):
    log = logging.getLogger(name)
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log


@pytest.mark.topic("py.genai.secrets", "py.genai.logging_tracking")
def test_redacting_filter_on_handler():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(ex.RedactingFilter([KEY, ""]))
    log = _logger_with(handler, "t23.redact")
    child = logging.getLogger("t23.redact.http")

    log.info("calling with key=%s", KEY)
    child.warning("retrying, header Authorization: Bearer %s", KEY)  # propagated from a child
    log.error("leaked other key sk-proj-ZZZZ9999yyyy in a message")
    log.info("short sk-abc is not a key")

    out = stream.getvalue()
    assert KEY not in out and "sk-proj-ZZZZ9999yyyy" not in out
    assert out.splitlines() == [
        "INFO calling with key=***",
        "WARNING retrying, header Authorization: Bearer ***",
        "ERROR leaked other key *** in a message",
        "INFO short sk-abc is not a key",
    ]


@pytest.mark.topic("py.genai.secrets")
def test_redacting_filter_never_drops_records():
    record = logging.makeLogRecord({"msg": "token %s and %d", "args": (KEY, 3)})
    assert ex.RedactingFilter([KEY]).filter(record) is True
    assert record.getMessage() == "token *** and 3"
    assert record.args is None


class ListHandler(logging.Handler):
    # NB: Handler.emit swallows exceptions raised by formatters (it prints them via
    # handleError), so the tests capture records and call YOUR formatter directly.
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.mark.topic("py.genai.logging_tracking")
def test_json_formatter_structured_fields():
    handler = ListHandler()
    log = _logger_with(handler, "t23.json")
    log.info("llm call done in %d ms", 412, extra={"request_id": "r-1", "model": "small-chat", "tokens": 97})
    line = ex.JsonFormatter().format(handler.records[0])
    assert "\n" not in line, "one JSON object per line"
    assert json.loads(line) == {
        "level": "INFO",
        "logger": "t23.json",
        "message": "llm call done in 412 ms",
        "request_id": "r-1",
        "model": "small-chat",
        "tokens": 97,
    }


@pytest.mark.topic("py.genai.logging_tracking")
def test_json_formatter_exceptions_and_unserialisable_values():
    handler = ListHandler()
    log = _logger_with(handler, "t23.json_exc")
    try:
        raise TimeoutError("upstream timed out")
    except TimeoutError:
        log.exception("embedding failed", extra={"elapsed": timedelta(seconds=2)})
    entry = json.loads(ex.JsonFormatter().format(handler.records[0]))
    assert entry["level"] == "ERROR"
    assert "TimeoutError: upstream timed out" in entry["exc"]
    assert entry["elapsed"] == "0:00:02", "non-JSON values are str()-ed"


@pytest.mark.topic("py.genai.secrets", "py.genai.logging_tracking")
def test_filter_and_json_formatter_together():
    handler = ListHandler()
    handler.addFilter(ex.RedactingFilter([KEY]))
    log = _logger_with(handler, "t23.both")
    log.warning("auth failed for %s", KEY, extra={"status": 401})
    entry = json.loads(ex.JsonFormatter().format(handler.records[0]))
    assert entry["message"] == "auth failed for ***"
    assert entry["status"] == 401
