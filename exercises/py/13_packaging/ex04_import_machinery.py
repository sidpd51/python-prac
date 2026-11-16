"""
Exercise 13.4 - Import machinery: load, locate, shadow, discover  level: ★★★
topics: packaging.imports, packaging.search_path, packaging.modules, packaging.packages

Problem
-------
1. import_from_path(path, module_name) -> ModuleType
     * load a .py file that is not on sys.path (a user-supplied tool, a config script)
       via importlib.util.spec_from_file_location + module_from_spec
     * register it in sys.modules[module_name] BEFORE executing it (dataclasses,
       pickling and recursive imports look the module up there), then execute it
     * if executing raises: remove it from sys.modules again and re-raise
     * missing file -> FileNotFoundError

2. find_origin(name, path=None) -> str | None
     * which entry of `path` (default sys.path) the TOP-LEVEL module `name` would be
       imported from - WITHOUT importing/executing it (importlib.machinery.PathFinder)
     * the first entry that has it wins (that's shadowing!); not found -> None
     * built-in modules (sys.builtin_module_names, e.g. "sys") -> "built-in"

3. shadowing_modules(directory, extra=()) -> list[str]
     * names of modules in `directory` that would shadow a standard-library module
       (sys.stdlib_module_names) or a name in `extra` (e.g. {"openai"}) if the directory
       came first on sys.path: "random.py", or a package dir "json/" WITH __init__.py
     * a directory WITHOUT __init__.py does not count: namespace packages lose to any
       regular module/package found later on the path
     * sorted, without duplicates

4. discover_tools(package_name) -> dict[str, Callable[..., Any]]
     * plugin registry for an agent: import the package, iterate its submodules with
       pkgutil.iter_modules(pkg.__path__), skip names starting with "_", import each
       (importlib.import_module with the full dotted name)
     * a submodule that defines a callable TOOL contributes {getattr(mod, "NAME",
       submodule_name): TOOL}; modules without TOOL are ignored
     * two modules registering the same NAME -> ValueError

Interviewers look for
---------------------
* sys.path order: script dir / cwd first, then PYTHONPATH, stdlib, site-packages ->
  a local `json.py`/`openai.py` silently shadows the real one.
* sys.modules is the import cache: a module executes once per process; registering
  before exec is what makes circular imports *partially* work.
* find_spec vs import: locating doesn't run code.
* Plugin discovery with pkgutil / entry points (importlib.metadata.entry_points).
"""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


def import_from_path(path: str | Path, module_name: str) -> ModuleType:
    raise NotImplementedError


def find_origin(name: str, path: Sequence[str] | None = None) -> str | None:
    raise NotImplementedError


def shadowing_modules(directory: str | Path, extra: Iterable[str] = ()) -> list[str]:
    raise NotImplementedError


def discover_tools(package_name: str) -> dict[str, Callable[..., Any]]:
    raise NotImplementedError
