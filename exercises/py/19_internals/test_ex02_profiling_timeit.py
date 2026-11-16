import pstats
import time

import pytest

from study.loader import load

ex = load(__file__, "ex02_profiling_timeit")


def slow_tokenize(text):
    # deliberately wasteful pure-Python work -> dominates tottime
    tokens = []
    for _ in range(200):
        tokens = []
        for word in text.split():
            tokens.append(word.lower())
    return tokens


def fast_normalize(text):
    return text.strip()


def pipeline(docs):
    out = []
    for doc in docs:
        out.append(len(slow_tokenize(fast_normalize(doc))))
    return out


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


DOCS = ["  The quick brown fox jumps over the lazy dog  " * 5] * 10


@pytest.mark.topic("py.internals.cprofile")
def test_profile_call_returns_result_and_stats():
    result, stats = ex.profile_call(pipeline, DOCS)
    assert result == [45] * 10
    assert isinstance(stats, pstats.Stats)


@pytest.mark.topic("py.internals.cprofile")
def test_profile_call_propagates_errors():
    def broken():
        raise KeyError("boom")

    with pytest.raises(KeyError):
        ex.profile_call(broken)
    # profiler must be off again: a fresh profile still works
    assert ex.profile_call(lambda: 7)[0] == 7


@pytest.mark.topic("py.internals.profiling", "py.internals.cprofile")
def test_hottest_functions_tottime_vs_cumtime():
    _, stats = ex.profile_call(pipeline, DOCS)
    assert ex.hottest_functions(stats, n=1) == ["slow_tokenize"]
    assert ex.hottest_functions(stats, n=1, key="cumtime") == ["pipeline"]
    top = ex.hottest_functions(stats, n=10)
    assert "fast_normalize" in top
    assert not any(name.startswith("<") for name in top), "built-ins must be skipped"
    with pytest.raises(ValueError):
        ex.hottest_functions(stats, key="ncalls")


@pytest.mark.topic("py.internals.cprofile")
def test_call_counts_include_recursion():
    result, stats = ex.profile_call(fib, 10)
    assert result == 55
    counts = ex.call_counts(stats)
    assert counts["fib"] == 177
    _, stats = ex.profile_call(pipeline, DOCS)
    counts = ex.call_counts(stats)
    assert counts["slow_tokenize"] == 10 and counts["pipeline"] == 1


@pytest.mark.topic("py.internals.timeit")
def test_best_time_per_call():
    t = ex.best_time(lambda: time.sleep(0.002), number=3, repeat=3)
    assert t >= 0.002  # sleep never returns early
    assert t < 0.5
    calls = []
    ex.best_time(lambda: calls.append(1), number=7, repeat=2)
    assert len(calls) == 14


@pytest.mark.topic("py.internals.timeit", "py.internals.profiling")
def test_rank_fastest_first():
    ranking = ex.rank(
        {"sleepy": lambda: time.sleep(0.003), "noop": lambda: None},
        number=3,
        repeat=3,
    )
    assert [label for label, _ in ranking] == ["noop", "sleepy"]
    assert all(isinstance(sec, float) for _, sec in ranking)
