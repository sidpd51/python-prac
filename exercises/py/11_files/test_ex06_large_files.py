import tracemalloc

import pytest

from study.loader import load

ex = load(__file__, "ex06_large_files")


@pytest.fixture
def big_log(tmp_path):
    """~12 MB log: 120k lines of 100 bytes."""
    p = tmp_path / "requests.log"
    line = b"x" * 99 + b"\n"
    with open(p, "wb") as f:
        for i in range(120):
            f.write(line * 999 + f"{i:099d}\n".encode())
    return p


@pytest.mark.topic("py.files.large_files", "py.files.text_vs_binary")
@pytest.mark.parametrize(
    ("data", "expected"),
    [(b"", 0), (b"a", 1), (b"a\n", 1), (b"a\nb", 2), (b"\n\n", 2), (b"a\r\nb\r\n", 2)],
)
def test_count_lines_small(tmp_path, data, expected):
    p = tmp_path / "f.log"
    p.write_bytes(data)
    assert ex.count_lines(p) == expected
    assert ex.count_lines(p, chunk_size=1) == expected


@pytest.mark.topic("py.files.large_files")
def test_count_lines_big_with_bounded_memory(big_log):
    tracemalloc.start()
    try:
        assert ex.count_lines(big_log, chunk_size=64 * 1024) == 120_000
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000, f"peak {peak} bytes - are you reading the whole file?"


@pytest.mark.topic("py.files.large_files", "py.files.read_write")
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 8192])
def test_tail_small(tmp_path, chunk_size):
    p = tmp_path / "f.log"
    p.write_bytes(b"one\ntwo\nthree\nfour\n")
    assert ex.tail(p, 2, chunk_size) == ["three", "four"]
    assert ex.tail(p, 10, chunk_size) == ["one", "two", "three", "four"]
    assert ex.tail(p, 0, chunk_size) == []
    p.write_bytes(b"no newline at end\nlast")
    assert ex.tail(p, 1, chunk_size) == ["last"]
    p.write_bytes(b"")
    assert ex.tail(p, 3, chunk_size) == []


@pytest.mark.topic("py.files.text_vs_binary")
@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_tail_multibyte_across_chunk_boundary(tmp_path, chunk_size):
    p = tmp_path / "f.log"
    p.write_text("héllo\n日本語テキスト\n✓ done\n", encoding="utf-8")
    assert ex.tail(p, 2, chunk_size) == ["日本語テキスト", "✓ done"]


@pytest.mark.topic("py.files.large_files", "py.files.open")
def test_tail_big_reads_from_the_end(big_log):
    tracemalloc.start()
    try:
        last = ex.tail(big_log, 3)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert last == ["x" * 99, "x" * 99, f"{119:099d}"]
    assert peak < 1_000_000, f"peak {peak} bytes - seek to the end instead of reading it all"


@pytest.mark.topic("py.files.large_files", "py.files.text_vs_binary")
def test_read_new_lines_resumes_and_skips_partial(tmp_path):
    p = tmp_path / "stream.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": ')
    lines, off = ex.read_new_lines(p)
    assert lines == ['{"a": 1}', '{"b": 2}']
    assert off == len(b'{"a": 1}\n{"b": 2}\n')
    # nothing new yet
    assert ex.read_new_lines(p, off) == ([], off)
    # writer finishes the line and adds another (with non-ASCII)
    with open(p, "ab") as f:
        f.write('3}\n{"d": "é"}\n'.encode())
    lines, off2 = ex.read_new_lines(p, off)
    assert lines == ['{"c": 3}', '{"d": "é"}']
    assert off2 == p.stat().st_size


@pytest.mark.topic("py.files.open")
def test_read_new_lines_offset_beyond_end(tmp_path):
    p = tmp_path / "rotated.log"
    p.write_bytes(b"short\n")
    with pytest.raises(ValueError):
        ex.read_new_lines(p, 1000)
    assert ex.read_new_lines(p, 6) == ([], 6)
