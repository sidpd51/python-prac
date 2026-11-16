"""
Exercise 06.1 - @timed: log how long each LLM call takes          level: ★☆☆
topics: decorators.timing, decorators.wraps, decorators.args_kwargs, decorators.wrappers, decorators.syntax

Problem
-------
Write `timed`, a decorator (no arguments: used as `@timed`) that measures wall-clock
duration of each call and logs it.

* Use time.perf_counter (monotonic, high resolution) - not time.time.
* Log on the logger named "llm.timing" (module constant LOGGER_NAME), level INFO, with
  exactly this message format (use lazy %-formatting, the logging way):
      "%s took %.1f ms"   with args (func.__qualname__, duration_ms)
* Forward any positional/keyword arguments unchanged and return the function's result.
* If the function raises, STILL log the duration (message format the same, but level
  WARNING) and re-raise the original exception unchanged.
* Store the last duration in seconds on the wrapper as `wrapper.last_duration` (float),
  set after every call (success or failure).
* Preserve the wrapped function's metadata (__name__, __qualname__, __doc__, __wrapped__).

Interviewers look for
---------------------
* `@timed` above `def f` is just `f = timed(f)` - the decorator runs ONCE, at definition time;
  the wrapper runs on every call.
* *args/**kwargs so the decorator works for any signature.
* try/finally (or except + re-raise) so failures are measured too - slow failures (timeouts)
  are exactly what you want to see in latency dashboards.
* perf_counter vs time.time (wall clock can jump with NTP).
"""

import logging
from collections.abc import Callable
from typing import Any

LOGGER_NAME = "llm.timing"
logger = logging.getLogger(LOGGER_NAME)


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    raise NotImplementedError
