"""
Exercise 06.2 - Stacking order: decoration time vs call time      level: ★★☆
topics: decorators.stacking, decorators.first_class, decorators.higher_order, decorators.syntax

Problem
-------
1. trace(log, name) -> decorator
   A decorator factory for observing ORDER. It appends to the list `log`:
     * f"{name}:decorate"  when the decorator is APPLIED to a function
     * f"{name}:before"    on each call, before calling the wrapped function
     * f"{name}:after"     on each call, after it returned (not on exceptions)
   and returns the wrapped function's result. Use functools.wraps.

       @trace(log, "auth")
       @trace(log, "cache")
       def answer(q): ...

   must produce, at definition:  ["cache:decorate", "auth:decorate"]
   and per call:                 ["auth:before", "cache:before", "cache:after", "auth:after"]

2. apply_decorators(func, *decorators) -> callable
   Apply decorators so that apply_decorators(f, d1, d2, d3) is EXACTLY equivalent to
       @d1
       @d2
       @d3
       def f(): ...
   (no decorators -> return func itself).

3. compose(*funcs) -> callable
   Mathematical composition, right-to-left: compose(f, g, h)(x) == f(g(h(x))).
   The rightmost function may take any args/kwargs; the others take one argument.
   compose() with no functions -> TypeError (raised by compose itself).

Interviewers look for
---------------------
* Decorators are applied bottom-up (closest to the def first) but the outermost wrapper
  runs first on a call - an "onion".
* Why order matters in practice: @cache above @auth caches responses for unauthorized users;
  @retry above @timed times each attempt, below it times the total.
* Functions are first-class objects: decorators, apply_decorators and compose just take and
  return functions.
"""

import functools
from collections.abc import Callable
from typing import Any


def trace(log: list[str], name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log.append(f"{name}:decorate")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log.append(f"{name}:before")
            result = func(*args, **kwargs)
            log.append(f"{name}:after")
            return result

        return wrapper

    return decorator


def apply_decorators(
    func: Callable[..., Any], *decorators: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> Callable[..., Any]:
    # the decorator written closest to `def` is applied first
    for deco in reversed(decorators):
        func = deco(func)
    return func


def compose(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    if not funcs:
        raise TypeError("compose() needs at least one function")
    *outer, innermost = funcs

    def composed(*args: Any, **kwargs: Any) -> Any:
        result = innermost(*args, **kwargs)
        for f in reversed(outer):
            result = f(result)
        return result

    return composed
