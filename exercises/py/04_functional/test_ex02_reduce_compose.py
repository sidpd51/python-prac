import copy
import re

import pytest

from study.loader import load

ex = load(__file__, "ex02_reduce_compose")


def remove_markup(s):
    return re.sub(r"<[^>]+>", "", s)


@pytest.mark.topic("py.functional.reduce", "py.functional.concepts")
def test_compose_right_to_left():
    clean = ex.compose(str.strip, str.lower, remove_markup)
    assert clean("  <b>Hello</b> RAG  ") == "hello rag"
    trace = []
    f = ex.compose(lambda x: trace.append("f") or x + 1, lambda x: trace.append("g") or x * 10)
    assert f(2) == 21
    assert trace == ["g", "f"]


@pytest.mark.topic("py.functional.reduce", "py.functional.lambda")
def test_compose_identity_and_single():
    ident = ex.compose()
    sentinel = object()
    assert ident(sentinel) is sentinel
    assert ex.compose(len)("abc") == 3


@pytest.mark.topic("py.functional.reduce", "py.functional.concepts")
def test_pipe_left_to_right():
    assert ex.pipe("  <i>X</i> ", remove_markup, str.strip, str.lower) == "x"
    assert ex.pipe(3, lambda x: x + 1, lambda x: x * 10) == 40
    assert ex.pipe(7) == 7


@pytest.mark.topic("py.functional.reduce", "py.functional.pure_functions")
def test_merge_usage():
    usages = [
        {"prompt_tokens": 120, "completion_tokens": 30},
        {"prompt_tokens": 80, "completion_tokens": 20, "cached_tokens": 64},
        {"completion_tokens": 5},
    ]
    snapshot = copy.deepcopy(usages)
    assert ex.merge_usage(usages) == {"prompt_tokens": 200, "completion_tokens": 55, "cached_tokens": 64}
    assert usages == snapshot, "an input dict was mutated"


@pytest.mark.topic("py.functional.reduce")
def test_merge_usage_empty_and_single():
    assert ex.merge_usage([]) == {}
    assert ex.merge_usage(iter([])) == {}
    single = {"prompt_tokens": 1}
    out = ex.merge_usage([single])
    assert out == {"prompt_tokens": 1}
    out["prompt_tokens"] = 999
    assert single == {"prompt_tokens": 1}, "result must not alias an input dict"
    assert ex.merge_usage([]) == {}, "the initializer leaked state between calls"


@pytest.mark.topic("py.functional.reduce", "py.functional.lambda")
def test_max_by():
    samples = [
        {"answer": "A", "judge": 0.7},
        {"answer": "B", "judge": 0.9},
        {"answer": "C", "judge": 0.9},
        {"answer": "D", "judge": 0.1},
    ]
    assert ex.max_by(samples, key=lambda s: s["judge"])["answer"] == "B"  # first wins on ties
    assert ex.max_by(["bb", "a", "ccc"], key=len) == "ccc"
    assert ex.max_by(iter([5]), key=lambda x: x) == 5


@pytest.mark.topic("py.functional.reduce")
def test_max_by_empty_and_key_calls():
    calls = []

    def key(x):
        calls.append(x)
        return -x

    assert ex.max_by([3, 1, 2], key=key) == 1
    assert sorted(calls) == [1, 2, 3], "key must be called exactly once per item"
    with pytest.raises(ValueError):
        ex.max_by([], key=len)
