import os
from pathlib import Path

import pytest

from study.loader import load

ex = load(__file__, "ex04_corpus_walker")


def make_tree(root: Path) -> None:
    files = {
        "README.MD": "# readme",
        "docs/guide.md": "guide text",
        "docs/api/ref.txt": "reference",
        "docs/api/ref.draft.md": "draft",
        "docs/image.png": "png",
        "notes.md/inner.txt": "inside a dir named notes.md",
        ".git/config.txt": "git",
        "src/node_modules/pkg/readme.md": "vendored",
        "src/.hidden.md": "hidden",
        "src/main.py": "print()",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


@pytest.mark.topic("py.files.pathlib", "py.files.dir_ops")
def test_find_documents_defaults(tmp_path):
    make_tree(tmp_path)
    found = ex.find_documents(tmp_path)
    assert found == [
        Path("README.MD"),
        Path("docs/api/ref.txt"),
        Path("docs/guide.md"),
        Path("notes.md/inner.txt"),
    ]
    assert all(not p.is_absolute() for p in found)


@pytest.mark.topic("py.files.pathlib")
def test_custom_suffixes_and_ignore(tmp_path):
    make_tree(tmp_path)
    assert ex.find_documents(tmp_path, suffixes=(".py", ".PNG"), ignore=()) == [
        Path("docs/image.png"),
        Path("src/main.py"),
    ]
    got = ex.find_documents(tmp_path, suffixes=(".md",), ignore=("docs",))
    assert Path("src/node_modules/pkg/readme.md") in got
    assert Path("src/.hidden.md") in got
    assert not any(p.parts[0] == "docs" for p in got)


@pytest.mark.topic("py.files.dir_ops", "py.files.large_files")
def test_ignored_directories_are_pruned(tmp_path, monkeypatch):
    make_tree(tmp_path)
    scanned = []
    real_scandir = os.scandir

    def spy(path="."):
        scanned.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", spy)
    ex.find_documents(tmp_path)
    assert not any("node_modules" in s or ".git" in s for s in scanned), "prune, don't filter"


@pytest.mark.topic("py.files.dir_ops")
def test_bad_roots(tmp_path):
    (tmp_path / "f.md").write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        ex.find_documents(tmp_path / "f.md")
    with pytest.raises(FileNotFoundError):
        ex.find_documents(tmp_path / "missing")


@pytest.mark.topic("py.files.pathlib", "py.files.dir_ops")
def test_corpus_stats(tmp_path):
    make_tree(tmp_path)
    assert ex.corpus_stats(tmp_path) == {
        ".md": (2, len("# readme") + len("guide text")),
        ".txt": (2, len("reference") + len("inside a dir named notes.md")),
    }


@pytest.mark.topic("py.files.dir_ops", "py.files.read_write")
def test_export_tree(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "out"
    make_tree(src)
    files = ex.find_documents(src)
    written = ex.export_tree(src, dst, files, str.upper)
    assert written == [
        (dst / "README.txt").resolve(),
        (dst / "docs/api/ref.txt").resolve(),
        (dst / "docs/guide.txt").resolve(),
        (dst / "notes.md/inner.txt").resolve(),
    ]
    assert (dst / "docs/guide.txt").read_text(encoding="utf-8") == "GUIDE TEXT"
    assert (src / "docs/guide.md").read_text(encoding="utf-8") == "guide text"


@pytest.mark.topic("py.files.pathlib")
@pytest.mark.parametrize("evil", ["../escaped.md", "docs/../../escaped.md", "ABSOLUTE"])
def test_export_tree_blocks_path_traversal(tmp_path, evil):
    src, dst = tmp_path / "src", tmp_path / "out"
    if evil == "ABSOLUTE":
        evil = str(tmp_path / "escaped.md")
    src.mkdir()
    (tmp_path / "escaped.md").write_text("original", encoding="utf-8")
    with pytest.raises(ValueError):
        ex.export_tree(src, dst, [evil], str.upper)
    assert (tmp_path / "escaped.md").read_text(encoding="utf-8") == "original"
    assert not (tmp_path / "escaped.txt").exists()
