import warnings

import pytest

from study.loader import load

ex = load(__file__, "ex05_class_based")


def embed(text, model="small"):
    """Embed a text."""
    return f"{model}:{text}"


@pytest.mark.topic("py.decorators.class_based")
def test_counts_calls():
    counted = ex.CountCalls(embed)
    assert isinstance(counted, ex.CountCalls)
    assert counted.calls == 0
    assert counted("a") == "small:a"
    assert counted("b", model="large") == "large:b"
    assert counted.calls == 2
    counted.reset()
    assert counted.calls == 0


@pytest.mark.topic("py.decorators.class_based")
def test_counts_failing_calls():
    @ex.CountCalls
    def boom():
        raise ValueError("bad")

    for _ in range(2):
        with pytest.raises(ValueError):
            boom()
    assert boom.calls == 2


@pytest.mark.topic("py.decorators.metadata", "py.decorators.wraps")
def test_update_wrapper_metadata():
    counted = ex.CountCalls(embed)
    assert counted.__name__ == "embed"
    assert counted.__qualname__ == "embed"
    assert counted.__doc__ == "Embed a text."
    assert counted.__module__ == embed.__module__
    assert counted.__wrapped__ is embed


@pytest.mark.topic("py.decorators.class_based")
def test_works_on_methods():
    class Retriever:
        def __init__(self, k):
            self.k = k

        @ex.CountCalls
        def search(self, query):
            return [f"{query}#{i}" for i in range(self.k)]

    r1, r2 = Retriever(2), Retriever(1)
    assert r1.search("q") == ["q#0", "q#1"]
    assert r2.search(query="z") == ["z#0"]
    assert isinstance(Retriever.__dict__["search"], ex.CountCalls)
    assert Retriever.search is Retriever.__dict__["search"]
    assert Retriever.search.calls == 2
    assert Retriever.search(r1, "x") == ["x#0", "x#1"]  # unbound access still works
    assert Retriever.search.calls == 3


@pytest.mark.topic("py.decorators.class_based", "py.decorators.with_arguments")
def test_deprecated_warns_once():
    @ex.deprecated("use embed_v2")
    def embed_v1(text):
        """Old embedder."""
        return text[::-1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert embed_v1("abc") == "cba"
        assert embed_v1("xy") == "yx"
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert str(caught[0].message) == "use embed_v2"
    assert caught[0].filename == __file__, "stacklevel=2 should point at the caller"
    assert embed_v1.__name__ == "embed_v1"
    assert embed_v1.__doc__ == "Old embedder."


@pytest.mark.topic("py.decorators.with_arguments", "py.decorators.class_based")
def test_deprecated_custom_category_and_independent_state():
    dep = ex.deprecated("legacy", category=FutureWarning)
    assert isinstance(dep, ex.deprecated)

    def a():
        return "a"

    def b():
        return "b"

    wa, wb = dep(a), dep(b)
    with pytest.warns(FutureWarning, match="legacy"):
        wa()
    with pytest.warns(FutureWarning, match="legacy"):
        wb()


@pytest.mark.topic("py.decorators.stacking", "py.decorators.class_based")
def test_stacks_with_count_calls():
    @ex.CountCalls
    @ex.deprecated("old")
    def f(x):
        return x * 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert f(2) == 4
        assert f(3) == 6
    assert f.calls == 2
    assert f.__name__ == "f"
