"""
Exercise 08.3 - Generator coroutines: send(), throw(), close()   level: ★★★
topics: generators.send, generators.throw, generators.close, generators.resumption

Problem
-------
1. `primed` - a decorator that creates the generator and advances it to the first
   `yield` so callers can `.send()` immediately. Preserve the wrapped function's
   __name__/__doc__.

2. `averager()` - decorated with @primed. A running-average coroutine:
       avg = averager()
       avg.send(10)  -> 10.0
       avg.send(20)  -> 15.0
   * `avg.throw(Reset)` resets the state and returns 0.0 (the value of the next yield)
   * `avg.send(None)` is ignored: it returns the current average without counting
   * `avg.close()` must make the generator record a final summary into the module-level
     list CLOSED_SUMMARIES as a tuple (count, average) - use try/finally or catch
     GeneratorExit. Do NOT yield after GeneratorExit (that's a RuntimeError).
   (`Reset` is defined below; average of zero values is 0.0)

Interviewers look for
---------------------
* Why a fresh generator must be primed (send(non-None) to a just-started generator
  raises TypeError).
* send(x) == "resume, and make the paused `yield` expression evaluate to x".
* throw() raises AT the paused yield; close() throws GeneratorExit there.
* This is the ancestor of async/await: coroutines are generators that get resumed
  by an event loop.
"""

CLOSED_SUMMARIES: list[tuple[int, float]] = []


class Reset(Exception):
    """Throw this into averager() to reset it."""


def primed(func):
    raise NotImplementedError


@primed
def averager():
    """Running average coroutine."""
    raise NotImplementedError
