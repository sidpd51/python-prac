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


def write_package(root: Path) -> Path:
    raise NotImplementedError
