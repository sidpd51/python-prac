"""
Exercise 05.5 - Inspecting and patching closure cells             level: ★★★
topics: closures.inspection, closures.co_freevars, closures.free_variables, closures.lexical_scoping

Problem
-------
You're debugging an agent framework that builds callbacks as closures and you need to see
(and in a test, override) what they captured. Use `fn.__code__.co_freevars` and
`fn.__closure__` - do NOT use inspect.getclosurevars.

1. closure_vars(fn) -> dict[str, object]
   Map each free-variable name to the current value in its cell, in co_freevars order.
   * A plain function (no closure) -> {}
   * Empty cells (the enclosing variable was never assigned yet, or was `del`eted) are
     omitted - reading `cell_contents` on an empty cell raises ValueError.

2. set_closure_var(fn, name, value) -> None
   Rebind the captured variable `name` (cells are writable: `cell.cell_contents = v`).
   Every closure sharing that cell sees the new value.
   `name` not a free variable of fn -> KeyError (message must contain the name).

3. shares_state(f, g) -> bool
   True if the two closures share at least one cell OBJECT (identity, not equal values) -
   i.e. they were created by the same call of the same factory and both use that variable.
   Plain functions share nothing -> False.

Interviewers look for
---------------------
* co_freevars (names, on the code object - shared by all closures from one def) vs
  __closure__ (a tuple of cell objects, one per free var, per function object).
* co_cellvars of the OUTER function lists the variables that inner functions capture.
* Sibling closures share cells; two factory calls create different cells.
* This is how a debugger/`inspect` shows captured state, and why late binding happens:
  the cell is read at call time.
"""

from collections.abc import Callable
from typing import Any


def closure_vars(fn: Callable[..., Any]) -> dict[str, object]:
    cells = fn.__closure__ or ()
    out: dict[str, object] = {}
    for name, cell in zip(fn.__code__.co_freevars, cells, strict=True):
        try:
            out[name] = cell.cell_contents
        except ValueError:  # empty cell
            continue
    return out


def set_closure_var(fn: Callable[..., Any], name: str, value: object) -> None:
    names = fn.__code__.co_freevars
    if name not in names:
        raise KeyError(f"{name!r} is not a free variable of {fn.__qualname__}")
    fn.__closure__[names.index(name)].cell_contents = value  # type: ignore[index]


def shares_state(f: Callable[..., Any], g: Callable[..., Any]) -> bool:
    f_cells = {id(c) for c in f.__closure__ or ()}
    return any(id(c) in f_cells for c in g.__closure__ or ())
