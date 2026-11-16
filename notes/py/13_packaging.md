# Modules and packages

A module is a `.py` file; a package is a directory of modules. Python's import system is the machinery that finds
them, runs them once, and caches the result. Understanding it explains the classic failures: `ModuleNotFoundError`
when it works on your laptop, circular imports, "why did my script run twice", and "which Python and which
`pip` am I using". For AI engineering it also covers how you ship a project: `pyproject.toml`, virtual
environments and pinned dependencies so that a model service builds the same way on CI and in the container.

The runnable examples build tiny packages in a temp directory and run them in a subprocess, so they are
self-contained.

## Modules
<!-- topics: modules -->

A **module** is a single file of Python. `import foo` finds `foo.py`, executes it top to bottom **once**, and binds
the resulting module object to the name `foo`. Its top-level names (functions, classes, variables) become the
module's attributes.

```python
import math, types

print(type(math).__name__)                    # prints: module
print(math.pi)                                # prints: 3.141592653589793
m = types.ModuleType("scratch")               # a module is just an object with a namespace
m.answer = 42
print(m.answer, m.__name__)                   # prints: 42 scratch
```

Key facts:
* The module body runs at import time. Side effects at the top level (network calls, model loading) run on
  *import*, which slows startup and surprises users.
* `import x` binds the module; `from x import y` copies a reference to `y` into your namespace (rebinding `x.y`
  later is not seen by your copy).
* `import a.b.c as d` gives an alias. Avoid `from x import *`: it pollutes the namespace and hides where names
  come from (unless `__all__` limits it).

## Packages
<!-- topics: packages -->

A **package** is a module that contains other modules: a directory, importable by its name, with submodules as
attributes.

```text
myapp/
  __init__.py
  config.py
  llm/
    __init__.py
    client.py
```

`import myapp.llm.client` imports `myapp`, then `myapp.llm`, then `myapp.llm.client`, running each `__init__.py`
on the way. Directories without `__init__.py` still work as **namespace packages** (PEP 420), which can span
several directories, but regular packages with `__init__.py` are clearer for your own code and are what most
tools expect.

```python
import tempfile, pathlib, subprocess, sys, textwrap

root = pathlib.Path(tempfile.mkdtemp())
(root / "myapp" / "llm").mkdir(parents=True)
(root / "myapp" / "__init__.py").write_text("print('init myapp')\n")
(root / "myapp" / "llm" / "__init__.py").write_text("print('init llm')\n")
(root / "myapp" / "llm" / "client.py").write_text("NAME = 'client'\nprint('init client')\n")

code = "import myapp.llm.client as c; print(c.NAME)"
out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True).stdout
print(out)
# prints: init myapp
# prints: init llm
# prints: init client
# prints: client
```

## The import system
<!-- topics: imports -->

What `import foo` does:

```text
1. look in sys.modules["foo"]        -> hit: return it (no re-execution!)
2. miss: ask each finder on sys.meta_path to locate "foo" (searches sys.path)
3. loader creates the module object, **puts it in sys.modules first**, then executes the file
4. bind the name `foo` in the importing namespace
```

`sys.modules` is the cache dict. That is why a module's top-level code runs **once per process**, however many files
import it, and why importing is cheap the second time.

```python
import sys

print("colorsys" in sys.modules)       # prints: False  (nobody imported it yet)
import colorsys
print("colorsys" in sys.modules)       # prints: True
print(sys.modules["colorsys"] is colorsys)    # prints: True
```

You can force a re-execution with `importlib.reload(mod)`, mostly for notebooks. Dynamic import by string, used for
plugin systems and "model: provider.Class" configs:

```python
import importlib

mod = importlib.import_module("collections")
print(mod.OrderedDict.__name__)       # prints: OrderedDict
```

**Circular imports.** `a` imports `b` and `b` imports `a`. Because step 3 registers the half-built module in
`sys.modules` *before* executing it, the second import returns a partially initialised module, and the failure shows
up as `ImportError: cannot import name ...` (the exact wording of the hint after it varies by version):

```python
import tempfile, pathlib, subprocess, sys

root = pathlib.Path(tempfile.mkdtemp())
(root / "a.py").write_text("from b import B\nclass A: pass\n")
(root / "b.py").write_text("from a import A\nclass B: pass\n")
r = subprocess.run([sys.executable, "-c", "import a"], cwd=root, capture_output=True, text=True)
print(r.stderr.strip().splitlines()[-1][:40])
# prints: ImportError: cannot import name 'A' from
```

Fixes, best first: restructure (move shared code into a third module), import the *module* (`import a`) and use
`a.A` later at call time, or do the import inside the function that needs it. For type-only imports use
`from __future__ import annotations` plus `if TYPE_CHECKING:`.

> **Gotcha** Naming your file `json.py`, `random.py` or `typing.py` shadows the stdlib module for everything
> run from that directory. A top cause of "weird" import errors.

> **Interview** "Why does a module run once?" `sys.modules` caches it. "Why do circular imports fail?" The
> second importer receives a module that is still executing, so names defined after the import line do not exist yet.

## Absolute versus relative imports
<!-- topics: abs_vs_rel -->

* **Absolute**: full path from a top-level package on `sys.path`: `from myapp.llm.client import Client`.
* **Relative**: relative to the *current package*: `from .client import Client` (same package),
  `from ..config import settings` (parent). Leading dots only work inside a package.

```python
import tempfile, pathlib, subprocess, sys

root = pathlib.Path(tempfile.mkdtemp())
pkg = root / "shop"
pkg.mkdir()
(pkg / "__init__.py").write_text("")
(pkg / "prices.py").write_text("TAX = 0.2\n")
(pkg / "cart.py").write_text("from .prices import TAX\nfrom shop import prices\nprint(TAX, prices.TAX)\n")

r = subprocess.run([sys.executable, "-m", "shop.cart"], cwd=root, capture_output=True, text=True)
print(r.stdout.strip())                    # prints: 0.2 0.2
r = subprocess.run([sys.executable, str(pkg / "cart.py")], cwd=root, capture_output=True, text=True)
print(r.stderr.strip().splitlines()[-1])
# prints: ImportError: attempted relative import with no known parent package
```

The second run fails because running the file directly makes it `__main__` with no package, so `.prices` has no
anchor. Run modules with `python -m package.module` from the project root.

PEP 8 prefers absolute imports (clear, work when run as scripts if the package is installed); relative imports are
fine inside a package you may rename or vendor. **Implicit relative imports** (Python 2) no longer exist.

## __init__.py
<!-- topics: init_py -->

`__init__.py` runs when the package is first imported and its namespace *is* the package's namespace. Use it for:

* marking a regular package (even an empty file),
* re-exporting a public API so users write `from myapp import Client` instead of `from myapp.llm.client import Client`,
* `__all__` (what `import *` exports), `__version__`, light setup.

```python
import tempfile, pathlib, subprocess, sys

root = pathlib.Path(tempfile.mkdtemp())
(root / "kit").mkdir()
(root / "kit" / "__init__.py").write_text(
    "from .core import run\n__all__ = ['run']\n__version__ = '1.2'\n")
(root / "kit" / "core.py").write_text("def run():\n    return 'ran'\n")

code = "import kit; print(kit.run(), kit.__version__)"
print(subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True).stdout.strip())
# prints: ran 1.2
```

Keep it light: heavy imports there (torch, big models) make `import myapp` slow and increase circular-import
risk. Lazy imports inside functions are a legitimate fix for optional heavy dependencies.

## The main guard
<!-- topics: main_guard -->

When Python runs a file directly, that module's `__name__` is `"__main__"`; when it is imported, `__name__` is the
module's import path. The guard lets one file be both a library and a script.

```python
import tempfile, pathlib, subprocess, sys

root = pathlib.Path(tempfile.mkdtemp())
(root / "tool.py").write_text(
    "def main():\n    print('main ran')\n\n"
    "print('name is', __name__)\n"
    "if __name__ == '__main__':\n    main()\n")

run = lambda *a: subprocess.run([sys.executable, *a], cwd=root, capture_output=True, text=True).stdout
print(run("tool.py"))
# prints: name is __main__
# prints: main ran
print(run("-c", "import tool"))
# prints: name is tool
```

Without the guard, importing the module (for tests, or by `multiprocessing` on spawn platforms) runs the script's
work as a side effect. A package can be runnable via `__main__.py` (`python -m mypkg`).

> **Gotcha** `multiprocessing` with the `spawn` start method (Windows, macOS default) re-imports your main
> module in each worker. Without the guard you get an infinite process fork-bomb.

## Module search path
<!-- topics: search_path -->

When a module is not in `sys.modules`, Python searches the entries of **`sys.path`** in order, first match wins:

1. the script's directory (or `""`/cwd for `-c` and the REPL),
2. directories in the `PYTHONPATH` environment variable,
3. the standard library,
4. `site-packages` of the active environment (where `pip` installs),
5. entries added by `.pth` files and editable installs.

```python
import sys

print(isinstance(sys.path, list))                       # prints: True
print(any(p.endswith("site-packages") for p in sys.path))  # prints: True  (inside a venv or system python)
```

Because earlier entries win, a local file can **shadow** an installed or stdlib module. Debug with
`python -c "import x; print(x.__file__)"` and `python -m site`. Changing `sys.path` at runtime
(`sys.path.insert(0, ...)`) is a hack; prefer installing your project (`pip install -e .`) so it is importable
everywhere.

> **Gotcha** "It works in my IDE but `ModuleNotFoundError` in the terminal" is nearly always a different
> `sys.path` or a different interpreter. `python -c "import sys; print(sys.executable)"` tells you which one runs.

## Virtual environments
<!-- topics: venv -->

A **virtual environment** is a directory with its own Python executable link and its own `site-packages`, so each
project's dependencies are isolated from the system and from each other.

```bash
python -m venv .venv                  # create
source .venv/bin/activate             # activate (Windows: .venv\Scripts\activate)
which python                          # -> .../project/.venv/bin/python
deactivate
# without activating, call the env's interpreter directly:
.venv/bin/python -m pip install httpx
```

Activation merely puts `.venv/bin` first on `PATH`. Inside, `sys.prefix` differs from `sys.base_prefix`:

```python
import sys

in_venv = sys.prefix != sys.base_prefix
print(isinstance(in_venv, bool))      # prints: True
```

Tools: `venv` (stdlib), `virtualenv`, **`uv`** (fast, handles Python versions, lockfiles), `poetry`, `conda` (also
non-Python binaries such as CUDA libs). Never `pip install` into the system Python, never commit `.venv/`, and
always `python -m pip` so the pip matches the interpreter.

## pip and installing
<!-- topics: pip -->

`pip` installs distributions from PyPI (or git, or a local path) into the active environment.

```bash
python -m pip install httpx                # latest
python -m pip install "httpx>=0.27,<1"      # version range
python -m pip install -r requirements.txt   # from a file
python -m pip install -e .                  # editable: source changes apply without reinstall
python -m pip list                          # what is installed
python -m pip freeze                        # exact pins, requirements format
python -m pip show httpx                    # version, location, who requires it
python -m pip uninstall httpx
```

Under the hood pip resolves versions, downloads a **wheel** (prebuilt, fast) or builds an **sdist**, and unpacks
into `site-packages`. Version specifiers: `==1.2.3` exact, `>=1.2`, `~=1.2` (compatible release: `>=1.2,<2.0`),
`!=`. Extras install optional dependency groups: `pip install "mylib[openai]"`.

> **Gotcha** `pip install foo` into the wrong environment is the #1 packaging confusion. Check with
> `python -m pip --version`, which prints the path of the environment it will modify.

## Dependency management
<!-- topics: dependencies -->

Two kinds of dependency lists, with different jobs:

| | Library (published) | Application (deployed) |
|---|---|---|
| Goal | work with many versions | reproduce one exact environment |
| Pins | **loose ranges** (`httpx>=0.27`) | **exact** versions, ideally with hashes |
| Lives in | `pyproject.toml` `dependencies` | lock file / `requirements.txt` from a resolver |

Direct dependencies are what you import; **transitive** dependencies come along with them. A lock file records the
whole resolved tree so builds are repeatable (`uv lock`, `pip-compile`, `poetry.lock`). Without pins, a new
release of a transitive package can break a model service on the next deploy for no code change.

Practices: separate runtime from dev dependencies (`pytest`, `ruff`, `mypy` go in an optional group), update
deliberately (Dependabot/Renovate plus tests), audit (`pip-audit`), and keep GPU-specific wheels (torch)
documented because they need a special index URL. Dependency conflicts ("A needs x<2, B needs x>=2") are
resolved by the resolver or by changing a version; they cannot both be satisfied in one environment.

## pyproject.toml
<!-- topics: pyproject -->

`pyproject.toml` is the single standard config file (PEP 517/518/621). It declares **how to build** the project and
its **metadata and dependencies**, and holds tool config (ruff, pytest, mypy).

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ragkit"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["httpx>=0.27", "pydantic>=2"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[project.scripts]
ragkit = "ragkit.cli:main"          # installs a `ragkit` command

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`[build-system]` tells pip which backend builds the wheel; `[project]` is the standard metadata; `[project.scripts]`
creates console commands; `[tool.*]` tables configure tools. It is plain TOML, which Python 3.11+ reads with
`tomllib`:

```python
import tomllib

text = """
[project]
name = "ragkit"
dependencies = ["httpx>=0.27"]
[project.optional-dependencies]
dev = ["pytest"]
"""
cfg = tomllib.loads(text)
print(cfg["project"]["name"], cfg["project"]["dependencies"])   # prints: ragkit ['httpx>=0.27']
print(cfg["project"]["optional-dependencies"]["dev"])            # prints: ['pytest']
```

It replaces `setup.py`/`setup.cfg` for most projects. A **src layout** (`src/ragkit/...`) prevents accidentally
importing the uninstalled working copy; install with `pip install -e .` during development.

## requirements.txt
<!-- topics: requirements -->

A plain list of requirement specifiers, one per line, consumed by `pip install -r`. It is the older, simpler,
still ubiquitous way to pin an application's environment.

```text
httpx==0.27.2
pydantic>=2,<3
-r base.txt                  # include another file
--index-url https://...      # custom index (private or CUDA wheels)
```

Typical flow: keep loose top-level requirements in `pyproject.toml` (or `requirements.in`), compile them to a fully
pinned `requirements.txt` with `uv pip compile` or `pip-compile`, and install that in Docker/CI. `pip freeze >
requirements.txt` works but captures *everything*, including tools you installed by hand, and does not distinguish
direct from transitive packages.

| `pyproject.toml` | `requirements.txt` |
|---|---|
| metadata, build config, abstract deps | concrete install list |
| needed to publish or `pip install -e .` | enough to deploy an app |
| standard, tool-readable | simple, line-based, no metadata |

> **AI use** A Dockerfile for an inference service does `pip install --no-cache-dir -r requirements.txt` from a
> compiled, hash-pinned file so every build gets the same transformers and torch versions.

## Quick recap
- A module is a file run once on first import and cached in `sys.modules`; a package is a directory of modules.
- Import steps: check `sys.modules`, find via `sys.path`, register then execute, bind the name.
- Circular imports fail because the partially executed module is already in `sys.modules`; restructure or import lazily.
- Absolute imports are the default; relative imports need a package context, so run with `python -m pkg.mod`.
- `__init__.py` defines the package namespace: re-export the public API, keep it light.
- `if __name__ == "__main__":` separates script behaviour from import behaviour (and protects `multiprocessing`).
- `sys.path` order: script dir, `PYTHONPATH`, stdlib, site-packages; local files can shadow stdlib modules.
- One venv per project; use `python -m pip` so pip matches the interpreter; never install into system Python.
- Libraries declare loose ranges; applications pin exact versions through a lock file.
- `pyproject.toml` holds build backend, metadata, dependencies, scripts and tool config; `requirements.txt` is a pinned install list.
