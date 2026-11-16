import threading

import pytest

from study.loader import load

ex = load(__file__, "ex04_rate_limited")


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.mark.topic("py.decorators.with_arguments", "py.decorators.wrappers")
def test_allows_up_to_limit_then_raises():
    clock = FakeClock()
    calls = []

    @ex.rate_limited(3, 10.0, clock=clock)
    def complete(prompt):
        calls.append(prompt)
        return prompt.upper()

    assert [complete(p) for p in "abc"] == ["A", "B", "C"]
    with pytest.raises(ex.RateLimitExceeded) as info:
        complete("d")
    assert calls == ["a", "b", "c"], "a rejected call must not reach the function"
    assert info.value.retry_after == pytest.approx(10.0)


@pytest.mark.topic("py.decorators.with_arguments")
def test_sliding_window_and_retry_after():
    clock = FakeClock(0.0)

    @ex.rate_limited(2, 10.0, clock=clock)
    def f():
        return "ok"

    f()  # t=0
    clock.t = 4.0
    f()  # t=4
    clock.t = 6.0
    with pytest.raises(ex.RateLimitExceeded) as info:
        f()
    assert info.value.retry_after == pytest.approx(4.0)
    clock.t = 10.0  # the t=0 call leaves the window exactly now
    assert f() == "ok"
    with pytest.raises(ex.RateLimitExceeded) as info:
        f()
    assert info.value.retry_after == pytest.approx(4.0)  # oldest is t=4 -> expires at 14
    clock.t = 14.0
    assert f() == "ok"


@pytest.mark.topic("py.decorators.wrapping")
def test_rejected_calls_do_not_count_but_failures_do():
    clock = FakeClock(0.0)

    @ex.rate_limited(2, 5.0, clock=clock)
    def flaky(fail):
        if fail:
            raise ConnectionError("boom")
        return "ok"

    with pytest.raises(ConnectionError):
        flaky(True)
    assert flaky(False) == "ok"
    for _ in range(3):
        with pytest.raises(ex.RateLimitExceeded):
            flaky(False)
    clock.t = 5.0
    assert flaky(False) == "ok"


@pytest.mark.topic("py.decorators.with_arguments")
def test_each_function_has_own_window():
    clock = FakeClock()
    limiter = ex.rate_limited(1, 60.0, clock=clock)

    @limiter
    def a():
        return "a"

    @limiter
    def b():
        return "b"

    assert a() == "a"
    assert b() == "b"
    with pytest.raises(ex.RateLimitExceeded):
        a()


@pytest.mark.topic("py.decorators.with_arguments")
def test_validates_config_eagerly():
    with pytest.raises(ValueError):
        ex.rate_limited(0, 1.0)
    with pytest.raises(ValueError):
        ex.rate_limited(1, 0)


@pytest.mark.topic("py.decorators.wraps", "py.decorators.args_kwargs")
def test_metadata_and_args():
    @ex.rate_limited(5, 1.0)
    def chat(model, *, temperature=0.0):
        """Chat completion."""
        return model, temperature

    assert chat("m", temperature=0.3) == ("m", 0.3)
    assert chat.__name__ == "chat"
    assert chat.__doc__ == "Chat completion."


@pytest.mark.topic("py.decorators.wrappers")
def test_thread_safe():
    clock = FakeClock()
    allowed = []
    barrier = threading.Barrier(20)

    @ex.rate_limited(5, 60.0, clock=clock)
    def call(i):
        allowed.append(i)

    errors = []

    def worker(i):
        barrier.wait()
        try:
            call(i)
        except ex.RateLimitExceeded:
            pass
        except NotImplementedError as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    assert len(allowed) == 5
