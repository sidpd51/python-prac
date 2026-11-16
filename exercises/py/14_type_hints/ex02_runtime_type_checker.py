"""
Exercise 14.2 - A mini runtime type checker (pydantic-lite)      level: ★★★
topics: type_hints.union, type_hints.optional, type_hints.collections,
        type_hints.any_vs_object, type_hints.literal, type_hints.aliases,
        type_hints.typeddict, type_hints.callable, type_hints.functions,
        type_hints.generics

Problem
-------
LLM "structured output" must be validated against the type you expect. Build the core
of what pydantic/typeguard/beartype do, using typing.get_origin / get_args /
get_type_hints.

1. check_type(value, tp) -> bool
     tp                                   value is accepted when
     ----------------------------------   -------------------------------------------
     Any, object                          always
     None (or type(None))                 value is None
     float                                int or float (PEP 484 numeric promotion)
     a plain class (int, str, MyClass)    isinstance(value, tp)
     X | Y, Union[X, Y], Optional[X]      it matches any member
     Literal[a, b]                        == one of them AND same type (True != 1 here)
     list[X], set[X], frozenset[X]        right container type, every item matches X
     tuple[X, Y] / tuple[X, ...] / tuple[()]   fixed length & per-position / any length
     dict[K, V]                           a dict whose keys match K and values match V
     Sequence[X] / Mapping[K, V]          isinstance of the abc + items checked
                                          (a str IS a Sequence[str] - that's Python)
     Iterable[X], Iterator[X]             isinstance of the abc only (don't consume!)
     Callable[..., R]                     callable(value) (signatures aren't checked)
     a TypedDict class                    a dict with all required keys, no unknown
                                          keys, every present value matches its hint
     Annotated[X, ...]                    check against X
     a `type Alias = ...` alias (3.12)    check against Alias.__value__
     bare list / dict / tuple             isinstance only
     anything else (e.g. a TypeVar)       raise TypeError("unsupported type ...")

2. @validate_call - decorator
     * on every call, bind the arguments (inspect.signature), and check each annotated
       parameter with check_type (*args: each element, **kwargs: each value) ->
       TypeError naming the parameter if it doesn't match
     * check the return value against the return annotation -> TypeError
     * resolve annotations with typing.get_type_hints (so string annotations work);
       resolve them once, lazily on the first call is fine
     * keep the function's metadata (functools.wraps)

Interviewers look for
---------------------
* get_origin/get_args - how `list[int]`, `int | None` and `Literal[...]` look at runtime.
* Any vs object: both accept everything here, but statically Any switches checking OFF,
  object allows almost no operations without narrowing.
* What can't be checked at runtime: generators/iterators (consuming them), Callable
  signatures, TypeVars - and why static checking is still needed.
* bool is a subclass of int - a classic validation bug (Literal handles it).
"""

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def check_type(value: object, tp: Any) -> bool:
    raise NotImplementedError


def validate_call(fn: F) -> F:
    raise NotImplementedError
