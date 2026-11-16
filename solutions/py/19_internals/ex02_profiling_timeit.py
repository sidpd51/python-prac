"""
Exercise 19.2 - Find the hot spot: cProfile, pstats and timeit        level: ★★☆
topics: internals.profiling, internals.cprofile, internals.timeit

Problem
-------
"Measure, don't guess." Build a tiny profiling toolkit.

1. profile_call(fn, *args, **kwargs) -> tuple[result, pstats.Stats]
   Run fn under cProfile.Profile and return fn's result plus a pstats.Stats object.
   The profiler must be disabled afterwards even if fn raises (exception propagates).

2. hottest_functions(stats, n=3, key="tottime") -> list[str]
   Names (just the function name, e.g. "tokenize") of the n most expensive PYTHON
   functions, most expensive first.
     key="tottime" - time spent in the function's own body (excluding callees)
     key="cumtime" - time including everything it called
     any other key -> ValueError
   Skip built-in/C functions (in Stats.stats their filename is "~").
   Hint: Stats.stats maps (filename, lineno, funcname) ->
         (primitive_calls, total_calls, tottime, cumtime, callers)

3. call_counts(stats) -> dict[str, int]
   funcname -> TOTAL number of calls (recursive calls included) for Python functions.

4. best_time(fn, number=1000, repeat=5) -> float
   Seconds per call of the zero-argument callable fn: run timeit `repeat` times with
   `number` calls each and take the MINIMUM run, divided by number.

5. rank(candidates, number=1000, repeat=5) -> list[tuple[str, float]]
   candidates: {"label": zero-arg callable}. Return (label, best_time) pairs sorted
   fastest first.

Interviewers look for
---------------------
* Profile first (cProfile/py-spy) to find WHERE time goes; micro-benchmark (timeit) to
  compare alternatives for that one spot.
* tottime vs cumtime: a function with huge cumtime but tiny tottime is just a caller -
  optimise where tottime is.
* Why min() of repeats: noise only ever adds time. timeit disables GC during timing and
  uses time.perf_counter.
* Deterministic profilers add per-call overhead (distorting tiny functions); sampling
  profilers (py-spy, scalene, PEP 799's `profiling.sampling` in 3.15) suit production.
"""

import cProfile
import pstats
import timeit
from collections.abc import Callable
from typing import Any


def profile_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, pstats.Stats]:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = fn(*args, **kwargs)
    finally:
        profiler.disable()
    return result, pstats.Stats(profiler)


def hottest_functions(stats: pstats.Stats, n: int = 3, key: str = "tottime") -> list[str]:
    column = {"tottime": 2, "cumtime": 3}.get(key)
    if column is None:
        raise ValueError(f"key must be 'tottime' or 'cumtime', got {key!r}")
    rows = [
        (values[column], funcname)
        for (filename, _, funcname), values in stats.stats.items()  # type: ignore[attr-defined]
        if filename != "~"
    ]
    rows.sort(reverse=True)
    return [name for _, name in rows[:n]]


def call_counts(stats: pstats.Stats) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (filename, _, funcname), values in stats.stats.items():  # type: ignore[attr-defined]
        if filename != "~":
            counts[funcname] = counts.get(funcname, 0) + values[1]
    return counts


def best_time(fn: Callable[[], Any], number: int = 1000, repeat: int = 5) -> float:
    return min(timeit.Timer(fn).repeat(repeat=repeat, number=number)) / number


def rank(
    candidates: dict[str, Callable[[], Any]], number: int = 1000, repeat: int = 5
) -> list[tuple[str, float]]:
    timings = [(label, best_time(fn, number, repeat)) for label, fn in candidates.items()]
    return sorted(timings, key=lambda pair: pair[1])
