import pytest

from study.loader import load

ex = load(__file__, "ex04_memoize")


def make_embed():
    calls = []

    def embed(text, model="small"):
        """Return a fake embedding."""
        calls.append((text, model))
        return (len(text), model)

    return ex.memoize(embed), calls


@pytest.mark.topic("py.closures.closures")
def test_caches_results():
    embed, calls = make_embed()
    assert embed("hello") == (5, "small")
    assert embed("hello") == (5, "small")
    assert calls == [("hello", "small")]
    assert embed.cache_info() == (1, 1)


@pytest.mark.topic("py.closures.closures")
def test_kwargs_order_does_not_matter():
    calls = []

    @ex.memoize
    def f(x, a=0, b=0):
        calls.append(x)
        return x + a + b

    assert f(1, a=2, b=3) == 6
    assert f(1, b=3, a=2) == 6
    assert len(calls) == 1
    assert f(1, a=3, b=2) == 6
    assert len(calls) == 2


@pytest.mark.topic("py.closures.closures")
def test_unhashable_args_bypass_cache():
    calls = []

    @ex.memoize
    def total(xs):
        calls.append(1)
        return sum(xs)

    assert total([1, 2]) == 3
    assert total([1, 2]) == 3
    assert len(calls) == 2
    assert total.cache_info() == (0, 0)


@pytest.mark.topic("py.closures.closures")
def test_exceptions_are_not_cached():
    attempts = []

    @ex.memoize
    def flaky(x):
        attempts.append(x)
        if len(attempts) == 1:
            raise ConnectionError("transient")
        return x * 2

    with pytest.raises(ConnectionError):
        flaky(2)
    assert flaky(2) == 4
    assert flaky(2) == 4
    assert len(attempts) == 2
    assert flaky.cache_info() == (1, 2)


@pytest.mark.topic("py.closures.nonlocal", "py.closures.free_variables")
def test_cache_clear_resets_shared_state():
    embed, calls = make_embed()
    embed("a")
    embed("a")
    embed.cache_clear()
    assert embed.cache_info() == (0, 0)
    embed("a")
    assert len(calls) == 2
    assert embed.cache_info() == (0, 1)


@pytest.mark.topic("py.closures.free_variables", "py.closures.inspection")
def test_each_decorated_function_has_its_own_cache_in_a_cell():
    e1, c1 = make_embed()
    e2, c2 = make_embed()
    e1("x")
    e2("x")
    assert len(c1) == len(c2) == 1
    assert e1.__closure__, "wrapper must be a closure"
    cell_values = [c.cell_contents for c in e1.__closure__]
    assert any(isinstance(v, dict) for v in cell_values), "cache dict should be a free variable"
    assert not any(isinstance(v, dict) for v in vars(e1).values()), "no dict attributes"


@pytest.mark.topic("py.decorators.wraps", "py.decorators.metadata")
def test_metadata_preserved():
    embed, _ = make_embed()
    assert embed.__name__ == "embed"
    assert embed.__doc__ == "Return a fake embedding."
    assert embed.__wrapped__("abc") == (3, "small")
