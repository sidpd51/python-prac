import itertools

import pytest

from study.loader import load

ex = load(__file__, "ex04_with_desugared")


class Recorder:
    """Configurable CM that logs every protocol call into a shared list."""

    def __init__(self, name, log, *, swallow=False, fail_enter=False, fail_exit=False):
        self.name, self.log = name, log
        self.swallow, self.fail_enter, self.fail_exit = swallow, fail_enter, fail_exit

    def __enter__(self):
        self.log.append(f"enter {self.name}")
        if self.fail_enter:
            raise OSError(f"enter {self.name} failed")
        return self.name.upper()

    def __exit__(self, et, e, tb):
        self.log.append(f"exit {self.name} {et.__name__ if et else None}")
        if self.fail_exit:
            raise RuntimeError(f"exit {self.name} failed")
        return self.swallow


def outcome(fn):
    """Run fn, return ('ok', value) or ('raised', type name, context type name)."""
    try:
        return ("ok", fn())
    except NotImplementedError:
        raise  # unsolved stub -> reported as TODO
    except BaseException as e:  # noqa: BLE001 - we compare outcomes, incl. KeyboardInterrupt
        ctx = type(e.__context__).__name__ if e.__context__ else None
        return ("raised", type(e).__name__, ctx)


def body_ok(v):
    return f"result:{v}"


def body_fail(v):
    raise ValueError(f"body failed with {v}")


def body_interrupt(v):
    raise KeyboardInterrupt


def real_with(cm, body):
    with cm as v:
        return body(v)


def real_nested(cms, body):
    """Reference: genuinely nested `with` statements (ExitStack differs in corner cases)."""
    if not cms:
        return body([])
    with cms[0] as v:
        return real_nested(cms[1:], lambda vs: body([v, *vs]))


@pytest.mark.topic("py.context_managers.with_internals", "py.context_managers.protocol")
@pytest.mark.parametrize("body", [body_ok, body_fail, body_interrupt])
@pytest.mark.parametrize(
    "flags",
    [{}, {"swallow": True}, {"fail_enter": True}, {"fail_exit": True}],
    ids=["plain", "swallow", "fail_enter", "fail_exit"],
)
def test_run_with_matches_real_with(body, flags):
    real_log, our_log = [], []
    expected = outcome(lambda: real_with(Recorder("a", real_log, **flags), body))
    got = outcome(lambda: ex.run_with(Recorder("a", our_log, **flags), body))
    assert got == expected
    assert our_log == real_log


@pytest.mark.topic("py.context_managers.exit")
def test_run_with_passes_exception_details_and_reraises_original():
    seen = {}
    err = ValueError("boom")

    class Spy:
        def __enter__(self):
            return self

        def __exit__(self, et, e, tb):
            seen.update(et=et, e=e, tb=tb)
            return None

    def body(v):
        raise err

    with pytest.raises(ValueError) as info:
        ex.run_with(Spy(), body)
    assert info.value is err
    assert seen["et"] is ValueError and seen["e"] is err
    assert seen["tb"] is not None


@pytest.mark.topic("py.context_managers.with_internals", "py.context_managers.enter")
def test_run_with_looks_up_methods_on_the_type():
    log = []

    class NotACM:
        pass

    obj = NotACM()
    obj.__enter__ = lambda: log.append("instance enter")
    obj.__exit__ = lambda *a: log.append("instance exit")
    with pytest.raises(TypeError):
        ex.run_with(obj, body_ok)
    assert log == []

    class OnlyEnter:
        def __enter__(self):
            log.append("enter")

    with pytest.raises(TypeError):
        ex.run_with(OnlyEnter(), body_ok)
    assert log == [], "check both methods exist before calling __enter__"


SCENARIOS = list(
    itertools.product([{}, {"swallow": True}, {"fail_enter": True}, {"fail_exit": True}], repeat=3)
)


@pytest.mark.topic("py.context_managers.with_internals")
@pytest.mark.parametrize("body", [body_ok, body_fail])
@pytest.mark.parametrize("flags", SCENARIOS, ids=lambda f: "-".join(next(iter(d), "plain") for d in f))
def test_run_nested_matches_real_nested_with(flags, body):
    real_log, our_log = [], []
    real_cms = [Recorder(n, real_log, **f) for n, f in zip("abc", flags, strict=True)]
    our_cms = [Recorder(n, our_log, **f) for n, f in zip("abc", flags, strict=True)]
    expected = outcome(lambda: real_nested(real_cms, lambda vs: body("+".join(vs))))
    got = outcome(lambda: ex.run_nested(our_cms, lambda vs: body("+".join(vs))))
    assert got == expected
    assert our_log == real_log


@pytest.mark.topic("py.context_managers.with_internals")
def test_run_nested_values_and_empty():
    log = []
    cms = [Recorder("x", log), Recorder("y", log)]
    assert ex.run_nested(cms, lambda vs: vs) == ["X", "Y"]
    assert log == ["enter x", "enter y", "exit y None", "exit x None"]
    assert ex.run_nested([], lambda vs: vs) == []
