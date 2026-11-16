import logging

import pytest

from study.loader import load

ex = load(__file__, "ex03_retry_with_backoff")

LOGGER = logging.getLogger("study.tests.retry")


class Flaky:
    """Callable that raises the given exceptions in order, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def run(fn, **kw):
    sleeps = []
    kw.setdefault("logger", LOGGER)
    result = ex.call_with_retry(fn, sleep=sleeps.append, **kw)
    return result, sleeps


@pytest.mark.topic("py.exceptions.try_except")
def test_success_first_try():
    fn = Flaky()
    assert run(fn) == ("ok", [])
    assert fn.calls == 1


@pytest.mark.topic("py.exceptions.best_practices")
def test_retries_transient_with_exponential_backoff():
    fn = Flaky(ex.TransientError("503"), ex.TransientError("503"), ex.TransientError("503"))
    result, sleeps = run(fn, attempts=4, base_delay=0.5)
    assert result == "ok"
    assert fn.calls == 4
    assert sleeps == [0.5, 1.0, 2.0]


@pytest.mark.topic("py.exceptions.best_practices")
def test_retry_after_hint_and_cap():
    fn = Flaky(ex.TransientError("429", retry_after=3.0), ex.TransientError("429", retry_after=60))
    result, sleeps = run(fn, attempts=3, max_delay=10.0)
    assert result == "ok"
    assert sleeps == [3.0, 10.0]
    fn = Flaky(*[ex.TransientError("x") for _ in range(5)])
    _, sleeps = run(fn, attempts=6, base_delay=1.0, max_delay=4.0)
    assert sleeps == [1.0, 2.0, 4.0, 4.0, 4.0]


@pytest.mark.topic("py.exceptions.chaining", "py.exceptions.raising")
def test_exhaustion_raises_chained_retry_error():
    last = ex.TransientError("still 503")
    fn = Flaky(ex.TransientError("503"), ex.TransientError("503"), last)
    sleeps = []
    with pytest.raises(ex.RetryError) as info:
        ex.call_with_retry(fn, attempts=3, sleep=sleeps.append, logger=LOGGER)
    err = info.value
    assert err.attempts == 3
    assert err.last is last
    assert err.__cause__ is last
    assert str(err) == f"gave up after 3 attempts: {last!r}"
    assert len(sleeps) == 2, "no sleep after the final attempt"


@pytest.mark.topic("py.exceptions.best_practices")
@pytest.mark.parametrize("error", [ex.PermanentError("401"), ValueError("bug"), KeyboardInterrupt()])
def test_non_retryable_propagate_unchanged(error):
    fn = Flaky(error)
    sleeps = []
    with pytest.raises(type(error)) as info:
        ex.call_with_retry(fn, attempts=5, sleep=sleeps.append, logger=LOGGER)
    assert info.value is error
    assert fn.calls == 1
    assert sleeps == []


@pytest.mark.topic("py.exceptions.raising")
def test_invalid_attempts():
    fn = Flaky()
    with pytest.raises(ValueError):
        ex.call_with_retry(fn, attempts=0, sleep=lambda s: None)
    assert fn.calls == 0


@pytest.mark.topic("py.exceptions.logging")
def test_logs_warnings_then_error(caplog):
    fn = Flaky(*[ex.TransientError("503") for _ in range(3)])
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    with pytest.raises(ex.RetryError):
        ex.call_with_retry(fn, attempts=3, sleep=lambda s: None, logger=LOGGER)
    levels = [r.levelno for r in caplog.records if r.name == LOGGER.name]
    assert levels == [logging.WARNING, logging.WARNING, logging.ERROR]
    assert "1" in caplog.records[0].getMessage()


@pytest.mark.topic("py.exceptions.logging")
def test_default_logger_is_module_logger(caplog):
    fn = Flaky(ex.TransientError("503"))
    caplog.set_level(logging.WARNING)
    assert ex.call_with_retry(fn, attempts=2, sleep=lambda s: None) == "ok"
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert caplog.records[0].name == ex.__name__
