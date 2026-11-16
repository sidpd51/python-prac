import inspect

import pytest

from study.loader import load

ex = load(__file__, "ex06_retry")


def flaky_factory(failures, exc=TimeoutError):
    calls = []

    def call(x, *, scale=1):
        calls.append(x)
        if len(calls) <= failures:
            raise exc(f"attempt {len(calls)} failed")
        return x * scale

    return call, calls


@pytest.mark.topic("py.decorators.with_arguments", "py.decorators.args_kwargs")
def test_retries_until_success_with_backoff():
    delays = []
    fn, calls = flaky_factory(2)
    wrapped = ex.retry(times=4, exceptions=TimeoutError, backoff=0.5, sleep=delays.append)(fn)
    assert wrapped(3, scale=2) == 6
    assert calls == [3, 3, 3]
    assert delays == [0.5, 1.0]


@pytest.mark.topic("py.decorators.with_arguments")
def test_reraises_last_exception_and_no_sleep_after_last_attempt():
    delays = []
    fn, calls = flaky_factory(10)
    wrapped = ex.retry(times=3, backoff=1.0, sleep=delays.append)(fn)
    with pytest.raises(TimeoutError, match="attempt 3 failed"):
        wrapped(1)
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


@pytest.mark.topic("py.decorators.with_arguments")
def test_non_matching_exception_not_retried():
    delays = []
    fn, calls = flaky_factory(1, exc=KeyError)
    wrapped = ex.retry(times=5, exceptions=(TimeoutError, ConnectionError), sleep=delays.append)(fn)
    with pytest.raises(KeyError):
        wrapped(1)
    assert len(calls) == 1
    assert delays == []


@pytest.mark.topic("py.decorators.syntax", "py.decorators.with_arguments")
def test_bare_decorator_uses_defaults():
    fn, calls = flaky_factory(2)

    @ex.retry
    def call(x):
        """Docs."""
        return fn(x)

    assert call(5) == 5
    assert len(calls) == 3
    assert call.__name__ == "call"
    assert call.__doc__ == "Docs."

    fn2, calls2 = flaky_factory(3)
    with pytest.raises(TimeoutError):
        ex.retry(fn2)(1)
    assert len(calls2) == 3


@pytest.mark.topic("py.decorators.with_arguments")
def test_zero_backoff_never_sleeps_and_times_validated():
    delays = []
    fn, _ = flaky_factory(2)
    assert ex.retry(times=3, sleep=delays.append)(fn)(1) == 1
    assert delays == []
    with pytest.raises(ValueError):
        ex.retry(times=0)


@pytest.mark.topic("py.decorators.wraps", "py.decorators.higher_order")
async def test_async_function_retried():
    delays = []
    attempts = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    @ex.retry(times=3, exceptions=ConnectionError, backoff=0.1, async_sleep=fake_sleep)
    async def acomplete(prompt):
        """Async completion."""
        attempts.append(prompt)
        if len(attempts) < 3:
            raise ConnectionError("reset by peer")
        return f"answer:{prompt}"

    assert inspect.iscoroutinefunction(acomplete), "async in, async out"
    assert acomplete.__name__ == "acomplete"
    assert acomplete.__doc__ == "Async completion."
    assert await acomplete("hi") == "answer:hi"
    assert attempts == ["hi", "hi", "hi"]
    assert delays == [pytest.approx(0.1), pytest.approx(0.2)]


@pytest.mark.topic("py.decorators.with_arguments")
async def test_async_gives_up():
    attempts = []

    async def no_sleep(seconds):
        return None

    @ex.retry(times=2, backoff=0.01, async_sleep=no_sleep)
    async def always_fails():
        attempts.append(1)
        raise TimeoutError("still down")

    with pytest.raises(TimeoutError, match="still down"):
        await always_fails()
    assert len(attempts) == 2
