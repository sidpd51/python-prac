"""
Exercise 11.5 - Crash-safe checkpoints (atomic writes)           level: ★★☆
topics: files.tempfiles, files.dir_ops, files.json, files.with, files.open_close

Problem
-------
A long embedding job saves its progress to state.json. If the process dies mid-write
(OOM kill, Ctrl-C, full disk) a plain open(path, "w") leaves a truncated, unparsable
file - and the job can't resume.

1. atomic_write_text(path, text, *, encoding="utf-8") -> None
     * write to a NEW temporary file in the SAME directory as `path`
       (tempfile.NamedTemporaryFile(dir=..., delete=False) or tempfile.mkstemp),
       flush + os.fsync it, close it, then os.replace(tmp, path)
     * readers only ever see the old content or the complete new content
     * if anything fails (writing, encoding, fsync, os.replace): delete the temp file and
       re-raise; the original file is untouched and no temp files are left behind

2. atomic_write_json(path, obj) -> None
     same guarantees; a non-JSON-serializable obj -> TypeError, original intact

3. class Checkpoint(path)
     * load() -> dict: {} if the file doesn't exist yet, else the parsed JSON
     * save(state: dict) -> None: atomic
     * update(**changes) -> dict: load, merge, save, return the merged state

Interviewers look for
---------------------
* os.replace is atomic on POSIX and Windows *within one filesystem* - that's why the temp
  file must live in the target's directory, not in /tmp (cross-device rename fails or
  degrades to copy).
* flush() empties Python's buffer, fsync() asks the OS to persist to disk - different
  things. (Fully durable also fsyncs the directory; mention it.)
* tempfile module: NamedTemporaryFile, mkstemp, TemporaryDirectory, SpooledTemporaryFile
  - unique names, no races, automatic cleanup.
* try/except BaseException + re-raise is legitimate here: cleanup must also run on Ctrl-C.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed explicitly below
        "w",
        encoding=encoding,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    # serialize first: a TypeError then never even creates a temp file
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


class Checkpoint:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as f:
                state: dict[str, Any] = json.load(f)
                return state
        except FileNotFoundError:
            return {}

    def save(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.path, state)

    def update(self, **changes: Any) -> dict[str, Any]:
        state = self.load() | changes
        self.save(state)
        return state
