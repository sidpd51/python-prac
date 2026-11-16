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
        self.name = name
        self._clock = clock
        self._on_exit = on_exit
        self._start = 0.0
        self.elapsed: float | None = None

    def __enter__(self) -> "Timer":
        self.elapsed = None
        self._start = self._clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed = self._clock() - self._start
        if self._on_exit is not None:
            self._on_exit(self.name, self.elapsed)
        # returning None == falsy -> any exception propagates


class suppress_errors:  # noqa: N801 - mirrors contextlib.suppress
    def __init__(self, *exc_types: type[BaseException]) -> None:
        self.exc_types = exc_types
        self.error: BaseException | None = None

    def __enter__(self) -> "suppress_errors":
        self.error = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is not None and isinstance(exc, self.exc_types):
            self.error = exc
            return True
        return False
