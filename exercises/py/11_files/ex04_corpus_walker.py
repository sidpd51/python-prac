"""
Exercise 11.4 - Walking a document corpus with pathlib           level: ★★☆
topics: files.pathlib, files.dir_ops, files.read_write

Problem
-------
Collect the documents of a repo/knowledge base for indexing.

1. find_documents(root, suffixes=(".md", ".txt"), ignore=DEFAULT_IGNORE) -> list[Path]
     * recursive; returns paths RELATIVE to root, sorted by their POSIX string
     * files only - a *directory* called "notes.md" is not a document
     * suffix match is case-insensitive ("README.MD" counts); suffixes are given with dot
     * ignore: fnmatch-style patterns matched against EVERY path component
       (directory names and the file name), e.g. ".*" drops hidden files and dirs,
       "*.draft.md" drops drafts, "node_modules" drops the whole subtree.
       Don't descend into ignored directories at all (prune the walk - big
       node_modules / .venv trees are why indexing jobs are slow)
     * root that is not a directory -> NotADirectoryError (missing -> FileNotFoundError)

2. corpus_stats(root, suffixes=..., ignore=...) -> dict[str, tuple[int, int]]
     lower-cased suffix -> (file count, total size in bytes), over find_documents()

3. export_tree(src_root, dst_root, files, transform, new_suffix=".txt") -> list[Path]
     * for each relative path in `files`: read src_root/rel as UTF-8, write
       transform(text) to dst_root/rel with its suffix replaced by new_suffix,
       creating missing parent directories; returns the written paths in order
     * SECURITY: a relative path that would end up outside dst_root
       (e.g. "../../etc/passwd", or an absolute path) -> ValueError, nothing written
       for that entry

Interviewers look for
---------------------
* pathlib over string concatenation: `/`, .suffix/.stem/.with_suffix, .relative_to,
  .is_relative_to, .mkdir(parents=True, exist_ok=True), .stat().st_size.
* Pruning os.walk / Path.walk (3.12+) by editing the dirnames list in place vs rglob,
  which can't skip a subtree.
* Path traversal: never join untrusted relative paths without resolving and checking
  containment.
* Deterministic output order (sorted) - os.listdir/scandir order is filesystem-dependent.
"""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

DEFAULT_IGNORE = (".*", "node_modules", "__pycache__", "*.draft.md")


def find_documents(
    root: str | Path, suffixes: Sequence[str] = (".md", ".txt"), ignore: Sequence[str] = DEFAULT_IGNORE
) -> list[Path]:
    raise NotImplementedError


def corpus_stats(
    root: str | Path, suffixes: Sequence[str] = (".md", ".txt"), ignore: Sequence[str] = DEFAULT_IGNORE
) -> dict[str, tuple[int, int]]:
    raise NotImplementedError


def export_tree(
    src_root: str | Path,
    dst_root: str | Path,
    files: Iterable[str | Path],
    transform: Callable[[str], str],
    new_suffix: str = ".txt",
) -> list[Path]:
    raise NotImplementedError
