"""
Exercise 14.5 - Typing decorators & overloads (mypy --strict)    level: ★★★
topics: type_hints.callable, type_hints.typevar, type_hints.generics,
        type_hints.literal, type_hints.static_checking, type_hints.mypy_pyright,
        type_hints.aliases

Problem
-------
This module must pass `mypy --strict` (the tests run it) - including your code.

1. retry(*, attempts=3, exceptions=(Exception,), base_delay=0.1, sleep=time.sleep)
   -> a decorator that keeps the decorated function's EXACT signature for type checkers
     * uses ParamSpec (PEP 695 syntax: `def retry[**P, R](...)`) so that after
           @retry(attempts=2)
           def ask(prompt: str, temperature: float = 0.0) -> str: ...
       mypy still knows ask(prompt: str, temperature: float = 0.0) -> str and rejects
       ask(123). (Typing the decorator as Callable[..., Any] would silently accept it.)
     * attempts < 1 -> ValueError when retry(...) is called
     * calls the function; on one of `exceptions` sleeps base_delay * 2**k and retries;
       after the last attempt re-raises the last exception; other exceptions propagate
     * functools.wraps

2. get_setting(key) with @overload:
     get_setting("temperature") -> float, get_setting("max_tokens") -> int,
     get_setting("model") -> str ; values come from the SETTINGS dict (given);
     any other key -> KeyError.  mypy must infer the precise return type per literal.
     (The three @overload signatures are given; in real code their bodies are just
      `...`. Only the final, undecorated get_setting is implemented.)

3. compose[A, B, C](f: Callable[[A], B], g: Callable[[B], C]) -> Callable[[A], C]
     compose(f, g)(x) == g(f(x))

4. pipeline[T](*steps: Callable[[T], T]) -> Callable[[T], T]
     applies the steps left to right; no steps -> identity

Interviewers look for
---------------------
* Callable[[Args], R] vs Callable[..., R]; ParamSpec / Concatenate for decorators
  (the most common place where types get lost in real codebases).
* @overload + Literal for APIs whose return type depends on an argument value
  (`open(mode="rb")` returns BinaryIO - same idea).
* mypy vs pyright: both implement the typing spec; pyright is faster and stricter about
  inference, mypy has plugins (pydantic, django). Run one in CI with --strict on new code.
"""

import functools
import time
from collections.abc import Callable
from typing import Final, Literal, overload

SETTINGS: Final[dict[str, float | int | str]] = {
    "temperature": 0.2,
    "max_tokens": 1024,
    "model": "gpt-4o-mini",
}

type SettingKey = Literal["temperature", "max_tokens", "model"]


def retry[**P, R](
    *,
    attempts: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    base_delay: float = 0.1,
    sleep: Callable[[float], object] = time.sleep,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == attempts - 1:
                        raise
                    sleep(base_delay * 2**attempt)
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator


@overload
def get_setting(key: Literal["temperature"]) -> float: ...
@overload
def get_setting(key: Literal["max_tokens"]) -> int: ...
@overload
def get_setting(key: Literal["model"]) -> str: ...
def get_setting(key: SettingKey) -> float | int | str:
    if key not in SETTINGS:
        raise KeyError(key)
    value: float | int | str = SETTINGS[key]
    return value


def compose[A, B, C](f: Callable[[A], B], g: Callable[[B], C]) -> Callable[[A], C]:
    def composed(x: A) -> C:
        return g(f(x))

    return composed


def pipeline[T](*steps: Callable[[T], T]) -> Callable[[T], T]:
    def run(x: T) -> T:
        for step in steps:
            x = step(x)
        return x

    return run
