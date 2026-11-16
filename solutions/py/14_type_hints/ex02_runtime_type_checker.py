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

import collections.abc as cabc
import functools
import inspect
import types
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

F = TypeVar("F", bound=Callable[..., Any])


def _check_typeddict(value: object, td: Any) -> bool:
    if not isinstance(value, dict):
        return False
    hints = get_type_hints(td)
    required: frozenset[str] = td.__required_keys__
    if not required <= value.keys() or not value.keys() <= hints.keys():
        return False
    return all(check_type(v, hints[k]) for k, v in value.items())


def check_type(value: object, tp: Any) -> bool:
    if tp is Any or tp is object:
        return True
    if tp is None or tp is type(None):
        return value is None
    if isinstance(tp, TypeAliasType):
        return check_type(value, tp.__value__)
    if is_typeddict(tp):
        return _check_typeddict(value, tp)
    if tp is float:
        return isinstance(value, (int, float))

    origin, args = get_origin(tp), get_args(tp)
    if origin is None:
        if isinstance(tp, type):
            return isinstance(value, tp)
        raise TypeError(f"unsupported type {tp!r}")

    if origin is Annotated:
        return check_type(value, args[0])
    if origin is Union or origin is types.UnionType:
        return any(check_type(value, a) for a in args)
    if origin is Literal:
        return any(value == a and type(value) is type(a) for a in args)
    if origin is cabc.Callable:
        return callable(value)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(check_type(v, args[0]) for v in value)
        if args == ((),):  # tuple[()] on older reprs
            return value == ()
        return len(value) == len(args) and all(map(check_type, value, args))
    if origin in (dict, cabc.Mapping, cabc.MutableMapping):
        if not isinstance(value, origin):
            return False
        k_tp, v_tp = args or (Any, Any)
        return all(check_type(k, k_tp) and check_type(v, v_tp) for k, v in value.items())
    if origin in (list, set, frozenset, cabc.Sequence, cabc.MutableSequence, cabc.Set):
        if not isinstance(value, origin):
            return False
        item_tp = args[0] if args else Any
        return all(check_type(v, item_tp) for v in value)
    if isinstance(origin, type):  # Iterable, Iterator, type[...], user generics...
        return isinstance(value, origin)
    raise TypeError(f"unsupported type {tp!r}")


def validate_call(fn: F) -> F:
    sig = inspect.signature(fn)
    hints: dict[str, Any] | None = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal hints
        if hints is None:
            hints = get_type_hints(fn)
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            if name not in hints:
                continue
            kind = sig.parameters[name].kind
            if kind is inspect.Parameter.VAR_POSITIONAL:
                values = list(value)
            elif kind is inspect.Parameter.VAR_KEYWORD:
                values = list(value.values())
            else:
                values = [value]
            if not all(check_type(v, hints[name]) for v in values):
                raise TypeError(f"argument {name!r}={value!r} is not {hints[name]!r}")
        result = fn(*args, **kwargs)
        if "return" in hints and not check_type(result, hints["return"]):
            raise TypeError(f"return value {result!r} is not {hints['return']!r}")
        return result

    return wrapper  # type: ignore[return-value]
