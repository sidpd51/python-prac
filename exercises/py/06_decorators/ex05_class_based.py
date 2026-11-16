"""
Exercise 06.5 - Class-based decorators (and the method-binding trap)   level: ★★★
topics: decorators.class_based, decorators.metadata, decorators.wraps, decorators.with_arguments

Problem
-------
1. CountCalls - a class used directly as a decorator:
       @CountCalls
       def embed(text): ...
       embed("a"); embed("b"); embed.calls  -> 2
   * `embed` is now a CountCalls INSTANCE; calling it forwards *args/**kwargs and returns
     the result. `calls` counts every call (including ones that raise).
   * `reset()` sets calls back to 0.
   * Copy the metadata with functools.update_wrapper (you can't use @wraps on a class body):
     __name__, __qualname__, __doc__, __module__, __wrapped__ must match/point to the function.
   * It must ALSO work on methods:
         class Retriever:
             @CountCalls
             def search(self, q): ...
         Retriever().search("x")   # `self` must be passed correctly
     A plain instance with __call__ is not a descriptor, so `self` would be missing.
     Implement __get__: accessed on the class -> return the CountCalls object itself;
     accessed on an instance -> return a bound method (types.MethodType) that still updates
     the same counter. (The count is per decorated function, shared by all instances.)

2. deprecated(message, *, category=DeprecationWarning) - a class used as a decorator
   FACTORY (a decorator with arguments implemented as a class):
       @deprecated("use embed_v2")
       def embed_v1(text): ...
   * `deprecated(...)` creates an instance holding the config; calling that instance with a
     function returns a WRAPPER FUNCTION (made with functools.wraps).
   * The wrapper emits warnings.warn(message, category, stacklevel=2) on the FIRST call
     only (per decorated function), then just forwards the call.
   * Stack it with CountCalls in either order; nothing special required.

Interviewers look for
---------------------
* @Cls -> Cls(func): the instance replaces the function; state lives on the instance
  (cleaner than function attributes when there's real state).
* Functions are descriptors (they have __get__) - that's how they become bound methods.
  Callable instances aren't, which is the classic "missing self" bug of class decorators.
* With arguments, the roles shift: __init__ gets the config, __call__ gets the function.
* functools.update_wrapper is what @functools.wraps calls under the hood.
"""

from collections.abc import Callable
from typing import Any


class CountCalls:
    def __init__(self, func: Callable[..., Any]) -> None:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class deprecated:
    def __init__(self, message: str, *, category: type[Warning] = DeprecationWarning) -> None:
        raise NotImplementedError

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        raise NotImplementedError
