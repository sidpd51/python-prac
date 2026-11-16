"""
Exercise 12.4 - What `with` really does (desugaring)             level: ★★★
topics: context_managers.with_internals, context_managers.protocol,
        context_managers.exit, context_managers.enter

Problem
-------
Implement the `with` statement by hand - no `with`, no contextlib, no ExitStack.

1. run_with(cm, body) -> result | None
     behaves exactly like:
         def f():
             with cm as value:
                 return body(value)
     * __enter__ / __exit__ are looked up on type(cm), not on the instance (like the
       interpreter). If either is missing -> TypeError, and nothing is called.
     * __exit__ is called exactly once:
         - body succeeded -> __exit__(None, None, None); return body's result
         - body raised e  -> __exit__(type(e), e, e.__traceback__)
               truthy return -> exception swallowed, run_with returns None
               falsy         -> re-raise the ORIGINAL exception (bare `raise`)
       This holds for BaseExceptions too (KeyboardInterrupt reaches __exit__).
     * if __enter__ raises, __exit__ is NOT called
     * if __exit__ itself raises, that new exception propagates (Python chains the
       original one as its __context__ automatically)

2. run_nested(cms, body) -> result | None
     behaves like `with cms[0] as v0, cms[1] as v1, ...: return body([v0, v1, ...])`
     * enter left to right, exit right to left
     * each __exit__ sees the exception that is current at that point: if an inner
       one swallowed it, the outer ones get (None, None, None); if an inner one raised
       a new exception, the outer ones see the new one
     * if entering cms[k] fails, cms[k-1] ... cms[0] are exited with that exception
     * empty cms -> body([])
     (hint: it's a few lines on top of run_with)

Interviewers look for
---------------------
* Precise protocol knowledge: special-method lookup on the type, the 3 exit arguments,
  truthiness of __exit__'s return value, __enter__ failure semantics, reverse exit order.
* How an exception inside __exit__ replaces the original (and how __context__ keeps it).
* This is exactly what ExitStack implements generically.
"""

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

R = TypeVar("R")


def run_with(cm: Any, body: Callable[[Any], R]) -> R | None:
    raise NotImplementedError


def run_nested(cms: Sequence[Any], body: Callable[[list[Any]], R]) -> R | None:
    raise NotImplementedError
