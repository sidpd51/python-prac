"""
Exercise 19.1 - Reading bytecode with dis                            level: ★☆☆
topics: internals.dis, internals.bytecode, internals.cpython

Problem
-------
CPython compiles source to bytecode for a stack-based virtual machine; each function
carries a code object (func.__code__). Write small introspection helpers - the kind of
tool you'd use to answer "why is version A faster than version B?".

1. opnames(func) -> set[str]
   The set of opcode names in func's bytecode (use dis.get_instructions or dis.Bytecode).

2. global_names(func) -> set[str]
   Names func loads with LOAD_GLOBAL - globals AND builtins (len, print, ...).
   Locals and attributes are not included. Only func's own code object.

3. code_info(func) -> dict with keys:
     "argcount"     - number of positional parameters (co_argcount)
     "locals"       - tuple of local variable names, parameters first (co_varnames)
     "is_generator" - True if func is a generator function   (check co_flags)
     "is_coroutine" - True if func is an `async def` function (check co_flags)

4. folded_constants(source) -> tuple
   Compile `source` (module-level code) and return the code object's co_consts.
   Lets you see compile-time constant folding: "X = 24 * 60 * 60" stores 86400 directly.

5. disassembly(func) -> str
   The human-readable dis listing of func as a string (don't print it: dis.dis accepts a
   `file=` argument, or use dis.Bytecode(func).dis()).

Interviewers look for
---------------------
* "Python is interpreted" really means: compiled to bytecode once (cached in __pycache__/
  *.pyc), then executed by the eval loop. 3.11+ adds a specializing adaptive interpreter
  (LOAD_ATTR_INSTANCE_VALUE, BINARY_OP_ADD_INT...), visible with dis(adaptive=True).
* LOAD_FAST (array index into the frame) vs LOAD_GLOBAL (dict lookups in globals then
  builtins) - the basis of the "bind to a local" micro-optimisation.
* Opcode names change between versions (3.13 LOAD_FAST_LOAD_FAST, 3.14 LOAD_FAST_BORROW...),
  so never hard-code exact bytecode in tests or tools.
"""

from collections.abc import Callable
from typing import Any


def opnames(func: Callable[..., Any]) -> set[str]:
    raise NotImplementedError


def global_names(func: Callable[..., Any]) -> set[str]:
    raise NotImplementedError


def code_info(func: Callable[..., Any]) -> dict[str, Any]:
    raise NotImplementedError


def folded_constants(source: str) -> tuple[Any, ...]:
    raise NotImplementedError


def disassembly(func: Callable[..., Any]) -> str:
    raise NotImplementedError
