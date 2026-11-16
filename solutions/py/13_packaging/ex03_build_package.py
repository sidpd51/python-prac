"""
Exercise 13.3 - Lay out a real package on disk                   level: ★★☆
topics: packaging.packages, packaging.init_py, packaging.abs_vs_rel,
        packaging.main_guard, packaging.modules, packaging.imports

Problem
-------
write_package(root: Path) -> Path
    Create this package under `root` (write the files; return root / "ragkit"):

    ragkit/
    ├── __init__.py        __version__ = "0.1.0"
    │                      re-exports chunk and embed;  __all__ = ["chunk", "embed", "__version__"]
    │                      must NOT import ragkit.cli (importing the library must stay cheap)
    ├── _internal.py       SECRET = 42      (private: not re-exported)
    ├── embed.py           embed(text: str) -> list[int]: the length of each chunk of size 4,
    │                      e.g. embed("abcdefghij") == [4, 4, 2]; gets `chunk` via a
    │                      RELATIVE import
    ├── text/
    │   ├── __init__.py    re-exports chunk (relative import)
    │   └── chunking.py    chunk(text: str, size: int) -> list[str]  (fixed-size slices;
    │                      size <= 0 -> ValueError)
    ├── cli.py             main(argv: list[str] | None = None) -> int
    │                        prints f"ragkit {__version__}" (version imported relatively),
    │                        returns 0; plus a __name__ == "__main__" guard that does
    │                        raise SystemExit(main())
    └── __main__.py        makes `python -m ragkit` call cli.main() the same way

Rules the tests check (behaviour, not your exact text):
    * `import ragkit` works with root on sys.path, and does not import ragkit.cli
    * `from ragkit import *` binds exactly chunk, embed, __version__
    * every intra-package import is relative: a copy of the directory renamed to
      "ragkit_fork" must import and work WITHOUT the original "ragkit" importable
    * importing ragkit.cli prints nothing; running it as a script / with -m prints the version
    * `python -m ragkit` prints "ragkit 0.1.0" and exits with code 0

Interviewers look for
---------------------
* Module = one .py file; package = a directory with __init__.py (regular) - or without
  (namespace package). __init__.py runs on first import of the package: keep it light.
* `__all__` controls `from pkg import *` and documents the public API.
* Relative imports (`from .text.chunking import chunk`) only work inside a package
  (`__package__` set) - running a module file directly breaks them; use `python -m`.
* `if __name__ == "__main__":` - the module's name is "__main__" only when executed as
  the entry point; `__main__.py` is what `python -m package` runs.
"""

from pathlib import Path


def _sources() -> dict[str, str]:
    # in a private helper so the stub doesn't ship the answer
    return {
        "__init__.py": '''\
"""ragkit - a toy RAG toolkit."""

from .embed import embed
from .text import chunk

__version__ = "0.1.0"
__all__ = ["chunk", "embed", "__version__"]
''',
        "_internal.py": "SECRET = 42\n",
        "embed.py": """\
from .text.chunking import chunk


def embed(text: str) -> list[int]:
    return [len(c) for c in chunk(text, 4)]
""",
        "text/__init__.py": """\
from .chunking import chunk

__all__ = ["chunk"]
""",
        "text/chunking.py": """\
def chunk(text: str, size: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be > 0")
    return [text[i : i + size] for i in range(0, len(text), size)]
""",
        "cli.py": """\
from . import __version__


def main(argv: list[str] | None = None) -> int:
    print(f"ragkit {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        "__main__.py": """\
from .cli import main

raise SystemExit(main())
""",
    }


def write_package(root: Path) -> Path:
    pkg = Path(root) / "ragkit"
    for rel, source in _sources().items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return pkg
