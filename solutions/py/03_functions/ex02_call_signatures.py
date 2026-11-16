"""
Exercise 03.2 - Designing call signatures for an LLM client         level: ★★☆
topics: functions.pos_kw_only, functions.args, functions.kwargs, functions.unpacking,
        functions.positional_args, functions.keyword_args

Problem
-------
1. build_request(model, messages, /, *, temperature=0.0, max_tokens=None, **extra) -> dict
     * model and messages are positional-only; everything else keyword-only
     * returns {"model": model, "messages": list(messages), "temperature": temperature}
       plus "max_tokens" only if it is not None, plus every `extra` item whose value is
       not None (provider-specific options: top_p, stop, seed, response_format...)
     * because model/messages are positional-only, a caller CAN pass model=... or
       messages=... and it lands in **extra - that would silently override the real value:
       raise TypeError for those keys

2. call_with_retry(fn, /, *args, retries=2, retry_on=(TimeoutError, ConnectionError), **kwargs)
     * calls fn(*args, **kwargs) and returns its result
     * if it raises an exception that is an instance of a `retry_on` type, call again - at
       most `retries` extra attempts; if every attempt fails, re-raise the LAST exception
     * any other exception propagates immediately (no retry)
     * retries < 0 -> ValueError (before calling fn)
     * `fn` is positional-only so that the wrapped function may itself have a parameter
       called `fn` (call_with_retry(apply, fn=str.upper) must forward fn=... to apply)

3. bind(fn, /, *args, **kwargs) -> callable                      (a mini functools.partial)
     * bind(f, 1, x=2)(3, y=4) == f(1, 3, x=2, y=4)
     * keyword arguments given at call time override bound ones
     * the bound args are captured at bind time (later mutation of the caller's kwargs dict
       must not leak in)

Interviewers look for
---------------------
* `/` (3.8+) makes parameter names private to the implementation - you can rename them
  later, and **kwargs can accept those names. `*` forces keyword use for readability
  (`temperature=0.2`, not a mystery positional 0.2) and safe future reordering.
* *args collects extra positionals into a tuple, **kwargs extra keywords into a dict;
  `f(*xs, **d)` unpacks at the call site.
* Only retry what is retryable (timeouts, 429, 5xx) - never validation errors.
"""

from collections.abc import Callable, Iterable
from typing import Any


def build_request(
    model: str,
    messages: Iterable[dict[str, Any]],
    /,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    clash = {"model", "messages"} & extra.keys()
    if clash:
        raise TypeError(f"{sorted(clash)} must be passed positionally")
    payload: dict[str, Any] = {"model": model, "messages": list(messages), "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def call_with_retry(
    fn: Callable[..., Any],
    /,
    *args: Any,
    retries: int = 2,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError),
    **kwargs: Any,
) -> Any:
    if retries < 0:
        raise ValueError("retries must be >= 0")
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except retry_on:
            if attempt == retries:
                raise
    raise AssertionError("unreachable")


def bind(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Callable[..., Any]:
    bound_args, bound_kwargs = args, dict(kwargs)

    def call(*more_args: Any, **more_kwargs: Any) -> Any:
        return fn(*bound_args, *more_args, **{**bound_kwargs, **more_kwargs})

    return call
