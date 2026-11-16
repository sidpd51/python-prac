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

import importlib
import importlib.machinery
import importlib.util
import pkgutil
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


def import_from_path(path: str | Path, module_name: str) -> ModuleType:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def find_origin(name: str, path: Sequence[str] | None = None) -> str | None:
    if name in sys.builtin_module_names:
        return "built-in"
    for entry in sys.path if path is None else path:
        if importlib.machinery.PathFinder.find_spec(name, [entry]) is not None:
            return entry
    return None


def shadowing_modules(directory: str | Path, extra: Iterable[str] = ()) -> list[str]:
    protected = set(sys.stdlib_module_names) | set(extra)
    found: set[str] = set()
    for p in Path(directory).iterdir():
        if p.is_file() and p.suffix == ".py":
            name = p.stem
        elif p.is_dir() and (p / "__init__.py").is_file():
            name = p.name
        else:
            continue
        if name in protected:
            found.add(name)
    return sorted(found)


def discover_tools(package_name: str) -> dict[str, Callable[..., Any]]:
    pkg = importlib.import_module(package_name)
    tools: dict[str, Callable[..., Any]] = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{package_name}.{info.name}")
        tool = getattr(mod, "TOOL", None)
        if not callable(tool):
            continue
        name = getattr(mod, "NAME", info.name)
        if name in tools:
            raise ValueError(f"duplicate tool name {name!r} in {mod.__name__}")
        tools[name] = tool
    return tools
