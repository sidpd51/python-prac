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

import fnmatch
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

DEFAULT_IGNORE = (".*", "node_modules", "__pycache__", "*.draft.md")


def _ignored(name: str, ignore: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in ignore)


def find_documents(
    root: str | Path,
    suffixes: Sequence[str] = (".md", ".txt"),
    ignore: Sequence[str] = DEFAULT_IGNORE,
) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    wanted = {s.lower() for s in suffixes}
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _ignored(d, ignore)]  # prune in place
        for name in filenames:
            if _ignored(name, ignore) or Path(name).suffix.lower() not in wanted:
                continue
            found.append(Path(dirpath, name).relative_to(root))
    return sorted(found, key=lambda p: p.as_posix())


def corpus_stats(
    root: str | Path,
    suffixes: Sequence[str] = (".md", ".txt"),
    ignore: Sequence[str] = DEFAULT_IGNORE,
) -> dict[str, tuple[int, int]]:
    root = Path(root)
    stats: dict[str, tuple[int, int]] = {}
    for rel in find_documents(root, suffixes, ignore):
        suffix = rel.suffix.lower()
        count, size = stats.get(suffix, (0, 0))
        stats[suffix] = (count + 1, size + (root / rel).stat().st_size)
    return stats


def export_tree(
    src_root: str | Path,
    dst_root: str | Path,
    files: Iterable[str | Path],
    transform: Callable[[str], str],
    new_suffix: str = ".txt",
) -> list[Path]:
    src_root, dst_root = Path(src_root), Path(dst_root).resolve()
    written: list[Path] = []
    for rel in files:
        target = (dst_root / rel).with_suffix(new_suffix).resolve()
        if not target.is_relative_to(dst_root):
            raise ValueError(f"refusing to write outside {dst_root}: {rel}")
        text = (src_root / rel).read_text(encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(transform(text), encoding="utf-8")
        written.append(target)
    return written
