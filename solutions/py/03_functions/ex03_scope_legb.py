'''
Exercise 03.3 - Scope bugs: LEGB, UnboundLocalError, late binding   level: ★★☆
topics: functions.legb, functions.first_class, functions.return_values

Problem
-------
Each snippet below is broken. Explain (to yourself) WHY, then write the fixed function.

```python
total_tokens = 0
def record(n):
    total_tokens += n              # (A) UnboundLocalError
    return total_tokens

def make_counter():
    count = 0
    def inc():
        count += 1                 # (B) UnboundLocalError
        return count
    return inc

def make_formatters(templates):
    return [lambda x: t.format(x) for t in templates]   # (C) every lambda uses the LAST t
```

1. make_meter() -> tuple[record, total]
     * replaces the global counter in (A) with state that is private to each meter
     * record(n) adds n tokens and returns the new running total; total() returns it
     * two meters are fully independent; no module-level state
2. make_counter(start=0) -> inc          fixed version of (B); inc() returns start+1, start+2, ...
3. make_formatters(templates) -> list    fixed version of (C); the i-th function formats with
                                         templates[i]
4. resolve(name, local_ns, enclosing, global_ns) -> tuple[str, object]
     * simulate how Python looks up a name: Local -> Enclosing -> Global -> Builtins
     * local_ns and global_ns are dicts; enclosing is a list of dicts, INNERMOST first
     * return (scope, value) where scope is "local", "enclosing", "global" or "builtin"
     * not found anywhere -> NameError (mention the name in the message)
     * e.g. resolve("len", {}, [], {}) -> ("builtin", len)

Interviewers look for
---------------------
* Assignment ANYWHERE in a function body makes the name local for the WHOLE body - decided
  at compile time. So `x += 1` reads a local that doesn't exist yet -> UnboundLocalError.
* `global` (module scope) / `nonlocal` (nearest enclosing function scope) change that; prefer
  closures or objects over `global` for state.
* Closures capture VARIABLES, not values: the lambdas share one `t` cell, looked up when
  called. Fix with a default arg (`lambda x, t=t: ...`) or a factory function.
* Shadowing builtins (`list = [...]`, `max = 3`) breaks later code in that scope.
'''

import builtins
from collections.abc import Callable, Mapping, Sequence


def make_meter() -> tuple[Callable[[int], int], Callable[[], int]]:
    total_tokens = 0

    def record(n: int) -> int:
        nonlocal total_tokens
        total_tokens += n
        return total_tokens

    def total() -> int:
        return total_tokens

    return record, total


def make_counter(start: int = 0) -> Callable[[], int]:
    count = start

    def inc() -> int:
        nonlocal count
        count += 1
        return count

    return inc


def make_formatters(templates: Sequence[str]) -> list[Callable[[object], str]]:
    # bind the CURRENT t as a default value, evaluated once per lambda creation
    return [lambda x, t=t: t.format(x) for t in templates]


def resolve(
    name: str,
    local_ns: Mapping[str, object],
    enclosing: Sequence[Mapping[str, object]],
    global_ns: Mapping[str, object],
) -> tuple[str, object]:
    if name in local_ns:
        return "local", local_ns[name]
    for ns in enclosing:
        if name in ns:
            return "enclosing", ns[name]
    if name in global_ns:
        return "global", global_ns[name]
    if hasattr(builtins, name):
        return "builtin", getattr(builtins, name)
    raise NameError(f"name {name!r} is not defined")
