"""
Exercise 12.1 - Class-based context managers: Timer & suppress    level: ★☆☆
topics: context_managers.protocol, context_managers.enter, context_managers.exit,
        context_managers.custom

Problem
-------
1. class Timer - time a block (an LLM call, a retrieval step)
     Timer(name="block", *, clock=time.perf_counter, on_exit=None)
     * `with Timer("embed") as t:` binds t to the Timer itself (__enter__ returns self)
     * t.elapsed is None until the block finishes, then the float seconds
       (clock() at exit - clock() at enter)
     * on_exit(name, elapsed) is called after every block, also when it raised
     * a Timer NEVER swallows exceptions
     * reusable: entering it again starts a new measurement

2. class suppress_errors - like contextlib.suppress, but remembers what it swallowed
     suppress_errors(*exc_types)
     * exceptions that are instances of exc_types are swallowed and stored in .error
     * anything else propagates (and .error stays None)
     * .error is None when the block succeeded; reset on each entry
     * `as` target is the suppress_errors object itself

Interviewers look for
---------------------
* The protocol: `__enter__(self)` -> value bound by `as`; `__exit__(self, exc_type,
  exc, tb)` -> truthy return value SWALLOWS the exception. Returning None (falsy) lets
  it propagate - which is what you want 99% of the time.
* `as` binds __enter__'s return value, not the CM object (open() returns self, but
  e.g. a lock's __enter__ returns True).
* Cleanup in __exit__ runs on success AND failure.
"""

import time
from collections.abc import Callable
from types import TracebackType


class Timer:
    def __init__(
        self,
        name: str = "block",
        *,
        clock: Callable[[], float] = time.perf_counter,
        on_exit: Callable[[str, float], object] | None = None,
    ) -> None:
        raise NotImplementedError

    def __enter__(self) -> "Timer":
        raise NotImplementedError

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        raise NotImplementedError


class suppress_errors:
    def __init__(self, *exc_types: type[BaseException]) -> None:
        raise NotImplementedError

    def __enter__(self) -> "suppress_errors":
        raise NotImplementedError

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> bool:
        raise NotImplementedError
