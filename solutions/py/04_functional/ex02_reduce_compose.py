"""
Exercise 04.2 - compose, pipe and reduce-based aggregation          level: ★★☆
topics: functional.reduce, functional.concepts, functional.lambda, functional.pure_functions

Problem
-------
Implement each of these with functools.reduce (no explicit loops).

1. compose(*fns) -> callable
     * RIGHT-to-left: compose(f, g, h)(x) == f(g(h(x)))
     * compose() is the identity function
     * e.g. clean = compose(str.strip, str.lower, remove_markup)  (remove_markup runs first)

2. pipe(value, *fns) -> Any
     * LEFT-to-right application: pipe(x, f, g) == g(f(x)); no fns -> value

3. merge_usage(usages) -> dict[str, int]
     * usages: iterable of token-usage dicts from many LLM calls, e.g.
       {"prompt_tokens": 120, "completion_tokens": 30}; keys may differ between dicts
       (missing key counts as 0)
     * returns a NEW dict with the per-key sums; no input dict is modified
     * empty input -> {}   (think about reduce's initializer!)

4. max_by(items, key) -> item
     * the item with the largest key(item); on ties the FIRST one wins
     * empty -> ValueError; key() is called exactly once per item (it may be expensive)
     * e.g. pick the best answer from n samples by a judge score

Interviewers look for
---------------------
* reduce(f, xs) without an initializer raises TypeError on empty input and uses xs[0] as the
  start value - give an initializer whenever the input can be empty.
* reduce folds left: f(f(f(init, x1), x2), x3). Function composition IS a fold over
  functions.
* The reducer should be pure: return a new accumulator instead of mutating the one you got
  (mutating `acc` in place works but makes the initializer a shared mutable object).
* Readability: sum/min/max/any/all/"".join beat reduce for the common cases - Guido moved
  reduce to functools for a reason. Use it when no builtin fits.
"""

from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from typing import Any


def compose(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return reduce(lambda f, g: lambda x: f(g(x)), fns, lambda x: x)


def pipe(value: Any, *fns: Callable[[Any], Any]) -> Any:
    return reduce(lambda acc, fn: fn(acc), fns, value)


def merge_usage(usages: Iterable[Mapping[str, int]]) -> dict[str, int]:
    def add(acc: dict[str, int], usage: Mapping[str, int]) -> dict[str, int]:
        return {k: acc.get(k, 0) + usage.get(k, 0) for k in acc.keys() | usage.keys()}

    return reduce(add, usages, {})


def max_by[T](items: Iterable[T], key: Callable[[T], Any]) -> T:
    empty: Any = object()
    # decorate once so key() runs exactly once per item
    pairs = ((key(x), x) for x in items)
    best = reduce(lambda a, b: b if a is empty or b[0] > a[0] else a, pairs, empty)
    if best is empty:
        raise ValueError("max_by() of an empty iterable")
    return best[1]
