"""
Exercise 12.2 - @contextmanager: temporary env vars, cwd, attrs   level: ★★☆
topics: context_managers.contextmanager, context_managers.cleanup,
        context_managers.resources

Problem
-------
Tests and scripts often need "change this global thing just for this block" - an API key
env var, the working directory, a module attribute. Write them with
contextlib.contextmanager (generator-based context managers).

1. temp_env(**overrides: str | None)
     * on enter: for each key set os.environ[key] = value; value None means UNSET the var
     * on exit - also when the block raises - restore the exact previous state: vars that
       existed get their old value back, vars that didn't exist are removed again
     * any value that is not str/None -> TypeError on enter, BEFORE anything is changed
     * yields nothing (`with temp_env(OPENAI_API_KEY="test"):`)

2. working_directory(path)
     * chdir into path, yield the new cwd as a resolved Path, always chdir back on exit

3. patched(obj, name, value)
     * setattr(obj, name, value) for the block, then restore the old value; if the
       attribute didn't exist before, delete it on exit

All three must restore state when the block raises and must not swallow the exception.
They must also nest correctly (`with temp_env(A="1"), temp_env(A="2"):`).

Interviewers look for
---------------------
* The classic @contextmanager bug: `yield` not wrapped in try/finally -> restore code is
  skipped when the block raises (the exception is thrown INTO the generator at the yield).
* Code before yield == __enter__, code after == __exit__; the yielded value is the `as`
  target; swallowing would require catching the exception around the yield.
* Distinguishing "was unset" from "was empty string" when restoring.
* Validate everything before mutating global state (no half-applied changes).
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_MISSING = object()


@contextmanager
def temp_env(**overrides: str | None) -> Iterator[None]:
    for key, value in overrides.items():
        if value is not None and not isinstance(value, str):
            raise TypeError(f"env var {key} must be str or None, got {type(value).__name__}")
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def working_directory(path: str | Path) -> Iterator[Path]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield Path.cwd().resolve()
    finally:
        os.chdir(old)


@contextmanager
def patched(obj: Any, name: str, value: Any) -> Iterator[None]:
    old = getattr(obj, name, _MISSING)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, old)
