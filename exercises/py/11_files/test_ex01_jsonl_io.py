import json
import os
import types

import pytest

from study.loader import load

ex = load(__file__, "ex01_jsonl_io")

RECORDS = [
    {"prompt": "Say hi", "completion": "hi"},
    {"prompt": "Café?", "completion": "☕", "meta": {"lang": "fr"}},
]


def open_fds():
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None


@pytest.mark.topic("py.files.json", "py.files.read_write", "py.files.encoding")
def test_write_jsonl_format(tmp_path):
    p = tmp_path / "train.jsonl"
    assert ex.write_jsonl(p, RECORDS) == 2
    raw = p.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 2
    assert "Café".encode() in raw and "☕".encode() in raw, "ensure_ascii=False, UTF-8"
    assert b"\r\n" not in raw
    assert [json.loads(line) for line in raw.decode("utf-8").splitlines()] == RECORDS


@pytest.mark.topic("py.files.modes")
def test_write_truncates_and_append_appends(tmp_path):
    p = tmp_path / "evals.jsonl"
    ex.write_jsonl(p, RECORDS)
    ex.write_jsonl(p, RECORDS[:1])
    assert p.read_text(encoding="utf-8").count("\n") == 1, "'w' truncates"
    ex.append_jsonl(p, {"prompt": "x"})
    ex.append_jsonl(tmp_path / "new.jsonl", {"a": 1})  # creates the file
    assert list(ex.iter_jsonl(p)) == [RECORDS[0], {"prompt": "x"}]
    assert list(ex.iter_jsonl(tmp_path / "new.jsonl")) == [{"a": 1}]


@pytest.mark.topic("py.files.read_write", "py.files.large_files")
def test_iter_is_lazy_and_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"i": 1}\n\n   \n{"i": 2}\nnot json at all\n', encoding="utf-8")
    it = ex.iter_jsonl(p)
    assert isinstance(it, types.GeneratorType) or iter(it) is it
    assert next(it) == {"i": 1}
    assert next(it) == {"i": 2}  # the broken line further down hasn't been read yet
    with pytest.raises(ValueError, match=r"data.jsonl:5"):
        next(it)


@pytest.mark.topic("py.files.json")
def test_invalid_json_is_chained(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"i": 1}\n{"i": \n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2") as info:
        list(ex.iter_jsonl(p))
    assert isinstance(info.value.__cause__, json.JSONDecodeError)


@pytest.mark.topic("py.files.json")
def test_non_object_lines_are_invalid(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"i": 1}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2"):
        list(ex.iter_jsonl(p))


@pytest.mark.topic("py.files.json")
def test_skip_invalid_collects_errors(tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text('{"i": 1}\n{oops}\n"str"\n{"i": 2}\n', encoding="utf-8")
    errors = []
    assert list(ex.iter_jsonl(p, skip_invalid=True, errors=errors)) == [{"i": 1}, {"i": 2}]
    assert errors == [(2, "{oops}"), (3, '"str"')]
    assert list(ex.iter_jsonl(p, skip_invalid=True)) == [{"i": 1}, {"i": 2}]


@pytest.mark.topic("py.files.open")
def test_missing_file_fails_at_call_time(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.iter_jsonl(tmp_path / "nope.jsonl")  # note: no next()/list()


@pytest.mark.topic("py.files.open_close", "py.files.with")
@pytest.mark.skipif(open_fds() is None, reason="needs /proc/self/fd")
def test_file_closed_after_exhaustion_and_early_close(tmp_path):
    p = tmp_path / "data.jsonl"
    ex.write_jsonl(p, [{"i": i} for i in range(5)])
    before = open_fds()
    for _ in range(20):
        assert len(list(ex.iter_jsonl(p))) == 5
    assert open_fds() == before
    it = ex.iter_jsonl(p)
    next(it)
    it.close()
    assert open_fds() == before
