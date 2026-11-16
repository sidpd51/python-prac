import json
import os
from pathlib import Path

import pytest

from study.loader import load

ex = load(__file__, "ex05_atomic_write")


def listing(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


@pytest.mark.topic("py.files.tempfiles", "py.files.dir_ops")
def test_atomic_write_text_creates_and_replaces(tmp_path):
    p = tmp_path / "state.txt"
    ex.atomic_write_text(p, "v1 ✓")
    assert p.read_text(encoding="utf-8") == "v1 ✓"
    ex.atomic_write_text(p, "v2")
    assert p.read_text(encoding="utf-8") == "v2"
    assert listing(tmp_path) == ["state.txt"], "no temp files left behind"


@pytest.mark.topic("py.files.tempfiles", "py.files.open_close")
def test_temp_file_is_in_same_directory_and_fsynced(tmp_path, monkeypatch):
    p = tmp_path / "state.txt"
    replaced, synced = [], []
    real_replace, real_fsync = os.replace, os.fsync

    def spy_replace(src, dst, *a, **kw):
        replaced.append((Path(src), Path(dst)))
        return real_replace(src, dst, *a, **kw)

    def spy_fsync(fd):
        synced.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    ex.atomic_write_text(p, "data")
    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == p
    assert src.parent == p.parent, "temp file must be on the same filesystem (same dir)"
    assert src != p
    assert synced, "fsync the temp file before replacing"


@pytest.mark.topic("py.files.tempfiles")
def test_failure_during_replace_keeps_original(tmp_path, monkeypatch):
    p = tmp_path / "state.txt"
    p.write_text("original", encoding="utf-8")

    def boom(*a, **kw):
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk on fire"):
        ex.atomic_write_text(p, "new")
    assert p.read_text(encoding="utf-8") == "original"
    assert listing(tmp_path) == ["state.txt"]


@pytest.mark.topic("py.files.tempfiles")
def test_encoding_failure_keeps_original(tmp_path):
    p = tmp_path / "state.txt"
    p.write_text("original", encoding="utf-8")
    with pytest.raises(UnicodeEncodeError):
        ex.atomic_write_text(p, "café", encoding="ascii")
    assert p.read_text(encoding="utf-8") == "original"
    assert listing(tmp_path) == ["state.txt"]


@pytest.mark.topic("py.files.json", "py.files.tempfiles")
def test_atomic_write_json(tmp_path):
    p = tmp_path / "state.json"
    ex.atomic_write_json(p, {"done": [1, 2], "model": "text-embedding-3"})
    assert json.loads(p.read_text(encoding="utf-8")) == {
        "done": [1, 2],
        "model": "text-embedding-3",
    }
    with pytest.raises(TypeError):
        ex.atomic_write_json(p, {"bad": {1, 2}})  # sets aren't JSON
    assert json.loads(p.read_text(encoding="utf-8"))["done"] == [1, 2]
    assert listing(tmp_path) == ["state.json"]


@pytest.mark.topic("py.files.json", "py.files.with")
def test_checkpoint(tmp_path):
    cp = ex.Checkpoint(tmp_path / "ckpt.json")
    assert cp.load() == {}
    assert not (tmp_path / "ckpt.json").exists()
    cp.save({"offset": 0})
    assert cp.update(offset=128, batch=3) == {"offset": 128, "batch": 3}
    assert ex.Checkpoint(tmp_path / "ckpt.json").load() == {"offset": 128, "batch": 3}
    assert listing(tmp_path) == ["ckpt.json"]
