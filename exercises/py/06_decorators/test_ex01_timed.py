import logging
import time

import pytest

from study.loader import load

ex = load(__file__, "ex01_timed")


@pytest.mark.topic("py.decorators.timing", "py.decorators.syntax")
def test_logs_duration(caplog):
    @ex.timed
    def call_llm(prompt):
        time.sleep(0.02)
        return f"answer to {prompt}"

    with caplog.at_level(logging.INFO, logger=ex.LOGGER_NAME):
        assert call_llm("q") == "answer to q"
    records = [r for r in caplog.records if r.name == ex.LOGGER_NAME]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.INFO
    assert rec.msg == "%s took %.1f ms", "use lazy %-formatting"
    name, ms = rec.args
    assert name.endswith("call_llm")
    assert 15 <= ms < 1000
    assert 0.015 <= call_llm.last_duration < 1.0


@pytest.mark.topic("py.decorators.args_kwargs", "py.decorators.wrappers")
def test_forwards_args_and_kwargs():
    @ex.timed
    def build(system, *messages, temperature=0.0, **extra):
        return system, messages, temperature, extra

    assert build("s", "m1", "m2", temperature=0.5, top_p=0.9) == (
        "s",
        ("m1", "m2"),
        0.5,
        {"top_p": 0.9},
    )


@pytest.mark.topic("py.decorators.timing", "py.decorators.wrapping")
def test_failure_is_logged_and_reraised(caplog):
    @ex.timed
    def flaky():
        raise TimeoutError("upstream timeout")

    with (
        caplog.at_level(logging.INFO, logger=ex.LOGGER_NAME),
        pytest.raises(TimeoutError, match="upstream timeout"),
    ):
        flaky()
    rec = [r for r in caplog.records if r.name == ex.LOGGER_NAME][-1]
    assert rec.levelno == logging.WARNING
    assert isinstance(flaky.last_duration, float)


@pytest.mark.topic("py.decorators.wraps", "py.decorators.metadata")
def test_metadata_preserved():
    def embed(text: str) -> list[float]:
        """Embed one text."""
        return [0.0]

    wrapped = ex.timed(embed)
    assert wrapped is not embed
    assert wrapped.__name__ == "embed"
    assert wrapped.__qualname__ == embed.__qualname__
    assert wrapped.__doc__ == "Embed one text."
    assert wrapped.__wrapped__ is embed
    assert wrapped("x") == [0.0]


@pytest.mark.topic("py.decorators.syntax")
def test_decorator_runs_once_wrapper_runs_per_call(caplog):
    @ex.timed
    def ping():
        return "pong"

    with caplog.at_level(logging.INFO, logger=ex.LOGGER_NAME):
        for _ in range(3):
            ping()
    assert len([r for r in caplog.records if r.name == ex.LOGGER_NAME]) == 3
